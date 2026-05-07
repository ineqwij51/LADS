#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diffusion_3_Gpt.py  (v3: viz++ / subject-split saving / dense_flow adapter)

新增：
- 可视化更丰富：
  • lag 曲线带平滑、有效时间掩码、均值标注；
  • 对齐热图叠加 top-1 对齐路径与窗口边界；
  • 自相似对比保持；
  • 新增：训练曲线（CE 与 Diff）、raw vs learned 的 2D PCA 散点对比、
          扩散重建误差分布(ASD/TD)与单样本时间曲线。
- 划分记录：除 indices 外，同时保存每折 Train/Val/Test 的 subject IDs（严格检查无交叉）
  • <exp_dir>/splits_subjects.json   （全折）
  • <fold_dir>/subjects_split.json   （该折）
- 针对 dense_flow 的特征前端适配器：FeatureAdapter（多尺度 depthwise-temporal conv + LN + 残差）
  • --use_adapter 1 | 0
  • --adapter_strength {auto,light,strong}
  • 默认：若特征名包含 "flow"，自动启用 strong；否则 light。

兼容三种外部划分 JSON：
(A) indices:  {"folds":[{"train":[…],"val":[…],"test":[…]},…]}
(B) tv/test:  {"folds":[{"trainval_idx":[…],"test_idx":[…]},…]}  -> 内部再做一次分层切 val
(C) subjects: {"folds":[{"train_subjects":[…],"val_subjects":[…],"test_subjects":[…]},…]} -> 自动映射索引

其它核心训练逻辑沿用上一版。
"""

import os, re, json, math, time, argparse
from dataclasses import dataclass
from typing import Optional, List, Dict
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 尝试复用 Diffusion_3 的数据与可能的划分
try:
    import Diffusion_3 as d3
    has_d3 = True
except Exception as _e:
    print("[WARN] Could not import Diffusion_3.py; place this script alongside it. ->", _e)
    has_d3 = False
    d3 = None


# ---------- Utils ----------

def _extract_test_indices(fold_data):
    """Robustly extract test indices from different fold_data formats."""
    # dict-like with common keys
    if isinstance(fold_data, dict):
        for k in ("test_idx", "test", "test_indices"):
            if k in fold_data:
                return fold_data[k]
        # 有些实现会把 (train_idx, val_idx, test_idx) 放在一个键里
        if "indices" in fold_data and isinstance(fold_data["indices"], (list, tuple)):
            cand = fold_data["indices"]
            if len(cand) >= 1:
                return cand[-1]
        raise KeyError("fold_data dict has no 'test_idx'/'test'/'test_indices'.")
    # tuple/list: (train, val, test) 或 (trainval, test)
    if isinstance(fold_data, (list, tuple)):
        if len(fold_data) == 3:
            return fold_data[2]
        if len(fold_data) == 2:
            return fold_data[1]
    raise KeyError("Unsupported fold_data type/shape for test indices.")

def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_csv(rows, path, header=None):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header: w.writerow(header)
        for r in rows: w.writerow(r)

def moving_avg(x, k=5):
    x = np.asarray(x)
    if x.size == 0:
        return x
    if x.ndim != 1:
        x = x.reshape(-1)
    k = max(1, int(k))
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(x, kernel, mode='same')


# ---------- Subject helpers ----------

SUBJECT_KEYS = ['subject_id','sub_id','sid','id','subject','subid','participant','pid','uid']

def _extract_subject_id(sample: dict, subject_key: Optional[str] = None) -> Optional[str]:
    if subject_key and subject_key in sample:
        v = sample[subject_key]
        if torch.is_tensor(v): v = v.item() if v.numel()==1 else str(v.tolist())
        return str(v)
    for k in SUBJECT_KEYS:
        if k in sample:
            v = sample[k]
            if torch.is_tensor(v): v = v.item() if v.numel()==1 else str(v.tolist())
            return str(v)
    pat = re.compile(r'\b(\d{2,5})\b')
    for k, v in sample.items():
        if isinstance(v, str):
            m = pat.search(v)
            if m: return m.group(1)
        elif isinstance(v, (list,tuple)) and v and isinstance(v[0], str):
            for s in v:
                m = pat.search(s)
                if m: return m.group(1)
    return None

def _build_subject_index_map(dataset, subject_key: Optional[str] = None) -> Dict[str, List[int]]:
    sub2idx = {}
    for i in range(len(dataset)):
        s = dataset[i]
        sid = _extract_subject_id(s, subject_key)
        if sid is None: continue
        sub2idx.setdefault(str(sid), []).append(i)
    return sub2idx

def _indices_to_subjects(dataset, indices, subject_key: Optional[str] = None):
    subs = []
    for i in indices:
        s = dataset[i]
        sid = _extract_subject_id(s, subject_key)
        if sid is not None: subs.append(str(sid))
    return sorted(list(set(subs)))


# ---------- Collate ----------

def default_collate_with_padding(batch):
    B = len(batch)
    T_e = [s['exp'].shape[0] for s in batch]
    T_s = [s['sub'].shape[0] for s in batch]
    F_e = batch[0]['exp'].shape[-1]
    F_s = batch[0]['sub'].shape[-1]
    assert F_e == F_s, "exp/sub feature dims must match"
    T_e_max, T_s_max = max(T_e), max(T_s)

    def pad_time(x, T_max):
        if x.shape[0] == T_max:
            return x
        pad = torch.zeros((T_max - x.shape[0], x.shape[1]), dtype=x.dtype, device='cpu')
        return torch.cat([x.cpu(), pad], dim=0)

    exp = torch.stack([pad_time(s['exp'], T_e_max) for s in batch], dim=0)
    sub = torch.stack([pad_time(s['sub'], T_s_max) for s in batch], dim=0)
    label = torch.tensor([int(s['label']) for s in batch], dtype=torch.long)

    def make_mask(T_cur, T_max):
        m = torch.zeros((T_max,), dtype=torch.bool)
        m[:T_cur] = True
        return m

    exp_mask = torch.stack([make_mask(T_e[i], T_e_max) for i in range(B)], dim=0)
    sub_mask = torch.stack([make_mask(T_s[i], T_s_max) for i in range(B)], dim=0)

    if 'exp_mask' in batch[0]:
        def norm(m, T_max):
            m = m.bool()
            if m.dim() == 2 and m.size(0) == 1: m = m[0]
            if m.size(0) < T_max: m = torch.cat([m, torch.zeros(T_max-m.size(0), dtype=torch.bool)])
            elif m.size(0) > T_max: m = m[:T_max]
            return m
        exp_mask = torch.stack([norm(s['exp_mask'], T_e_max) for s in batch], dim=0)
    if 'sub_mask' in batch[0]:
        sub_mask = torch.stack([norm(s['sub_mask'], T_s_max) for s in batch], dim=0)

    return {'exp': exp, 'sub': sub, 'label': label, 'exp_mask': exp_mask, 'sub_mask': sub_mask}


# ---------- Model ----------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


def _to_bt_mask(mask: Optional[torch.Tensor], T: Optional[int] = None) -> Optional[torch.Tensor]:
    if mask is None: return None
    m = mask
    if m.dtype != torch.bool: m = m != 0
    if m.dim() == 3:
        if m.size(1) == 1: m = m[:, 0, :]
        elif m.size(2) == 1: m = m[:, :, 0]
    elif m.dim() > 3: m = m.squeeze()
    if m.dim() == 1: m = m.unsqueeze(0)
    if m.dim() != 2: raise ValueError(f"mask must end up (B,T), got {tuple(m.shape)}")
    if T is not None and m.size(1) != T:
        if m.size(1) > T: m = m[:, :T]
        else:
            pad = torch.zeros((m.size(0), T - m.size(1)), dtype=torch.bool, device=m.device)
            m = torch.cat([m, pad], dim=1)
    return m

def _invert_mask(mask: Optional[torch.Tensor], T: Optional[int] = None) -> Optional[torch.Tensor]:
    if mask is None: return None
    return ~_to_bt_mask(mask, T)


class TemporalEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int, nhead: int, num_layers: int, dropout: float,
                 use_adapter: int = 0, adapter_strength: str = "light"):
        super().__init__()
        self.adapter = None  # Adapter removed
        self.input_proj = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True,
            dropout=dropout, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)
        self.in_norm = nn.LayerNorm(d_model)
        self.out_norm = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = self.in_norm(x)
        x = self.pos(x)
        kpm = _invert_mask(mask, T=x.size(1))
        x = self.encoder(x, src_key_padding_mask=kpm)
        x = self.out_norm(x)
        return x


class LagAwareAligner(nn.Module):
    def __init__(self, d_model: int, window: int, dropout: float = 0.1, temperature: float = 1.0,
                 use_gauss: bool = True, learn_temp: int = 1):
        super().__init__()
        self.window = window
        self.log_temp = nn.Parameter(torch.tensor(math.log(max(1e-3, float(temperature)))) if int(learn_temp)==1 else torch.tensor(math.log(max(1e-3, float(temperature)))), requires_grad=(int(learn_temp)==1))
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.use_gauss = use_gauss
        if use_gauss:
            self.summary = nn.Sequential(nn.Linear(d_model*2, d_model), nn.GELU(), nn.Linear(d_model, 3))

    def _build_mask(self, B, T_s, T_e, device, exp_mask, sub_mask):
        u = torch.arange(T_s, device=device).unsqueeze(1)
        t = torch.arange(T_e, device=device).unsqueeze(0)
        base = (t <= u) & ((u - t) <= self.window)
        base = base.unsqueeze(0).expand(B, -1, -1)
        em = _to_bt_mask(exp_mask, T_e)
        sm = _to_bt_mask(sub_mask, T_s)
        if em is not None: base = base & em.unsqueeze(1).expand(-1, T_s, -1)
        if sm is not None: base = base & sm.unsqueeze(2).expand(-1, -1, T_e)
        any_valid = base.any(dim=-1)
        if not any_valid.all():
            b_idx, u_idx = (~any_valid).nonzero(as_tuple=True)
            if b_idx.numel() > 0:
                t_fb = torch.clamp(u_idx, min=0, max=T_e-1)
                base[b_idx, u_idx, t_fb] = True
        add = torch.zeros((B, T_s, T_e), device=device)
        add[~base] = float('-inf')
        return add, base

    def forward(self, E, S, exp_mask=None, sub_mask=None):
        B, T_e, C = E.shape
        _, T_s, _ = S.shape
        device = E.device
        temperature = self.log_temp.exp()
        Q = self.q(S); K = self.k(E); V = self.v(E)
        scores = torch.matmul(Q, K.transpose(1, 2)) / (math.sqrt(C) * temperature)
        add, base = self._build_mask(B, T_s, T_e, device, exp_mask, sub_mask)
        scores = scores + add

        attn = F.softmax(scores, dim=-1)
        sum_w = attn.sum(dim=-1, keepdim=True)
        good = torch.isfinite(sum_w) & (sum_w > 1e-8)
        denom = torch.where(good, sum_w, torch.ones_like(sum_w))
        attn = attn / denom
        bad_rows = ~good.squeeze(-1)
        if bad_rows.any():
            b_idx, u_idx = bad_rows.nonzero(as_tuple=True)
            first_valid = base[b_idx, u_idx].float().argmax(dim=-1)
            attn[b_idx, u_idx, :] = 0.0
            attn[b_idx, u_idx, first_valid] = 1.0

        if self.use_gauss:
            S_g = S.mean(dim=1); E_g = E.mean(dim=1)
            g = torch.cat([S_g, E_g], dim=-1)
            mu_logsig_beta = self.summary(g)
            mu = F.softplus(mu_logsig_beta[:, 0:1])
            logsig = mu_logsig_beta[:, 1:2].clamp(-2.5, 3.0)
            sigma = torch.exp(logsig) + 1.0
            beta = 5.0 * torch.sigmoid(mu_logsig_beta[:, 2:3])

            u = torch.arange(T_s, device=device).view(1, T_s, 1).float()
            t = torch.arange(T_e, device=device).view(1, 1, T_e).float()
            center = (u - mu.view(B,1,1)).clamp(min=0)
            gauss = -0.5 * ((t - center) / sigma.view(B,1,1)).pow(2)
            bias = torch.exp(gauss) * beta.view(B,1,1)

            attn = attn * bias
            sum_w = attn.sum(dim=-1, keepdim=True)
            good = torch.isfinite(sum_w) & (sum_w > 1e-8)
            denom = torch.where(good, sum_w, torch.ones_like(sum_w))
            attn = attn / denom
            bad_rows = ~good.squeeze(-1)
            if bad_rows.any():
                b_idx, u_idx = bad_rows.nonzero(as_tuple=True)
                first_valid = base[b_idx, u_idx].float().argmax(dim=-1)
                attn[b_idx, u_idx, :] = 0.0
                attn[b_idx, u_idx, first_valid] = 1.0

        attn = self.drop(attn)
        Ew = torch.matmul(attn, V)
        Ew = self.o(Ew)

        t_idx = torch.arange(T_e, device=device).float().view(1, 1, T_e)
        exp_t = (attn * t_idx).sum(dim=-1)
        u_idx = torch.arange(T_s, device=device).float().view(1, T_s)  # (1, T_s)
        lag = (u_idx - exp_t).clamp(min=0)                              # (B, T_s)
        sm = _to_bt_mask(sub_mask, T_s)
        if sm is not None:
            lag = lag * sm.float()
        return Ew, attn, lag


class ImitationLagAwareClassifier(nn.Module):
    def __init__(self, feature_dim: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2,
                 window: int = 12, num_classes: int = 2, dropout: float = 0.1,
                 use_gaussian_bias: int = 1,
                 use_dynamics: int = 1,
                 use_cos: int = 1,
                 use_lag_norm: int = 1,
                 pooling: str = "attn_mean_max",
                 align_mode: str = "lagaware",
                 use_adapter: int = 0,
                 adapter_strength: str = "light",
                 learn_temp: int = 1):
        super().__init__()
        self.use_cos = int(use_cos) == 1
        self.use_dynamics = int(use_dynamics) == 1
        self.use_lag_norm = int(use_lag_norm) == 1
        self.pooling = pooling
        self.align_mode = align_mode

        self.exp_enc = TemporalEncoder(feature_dim, d_model, nhead, num_layers, dropout,
                                       use_adapter=use_adapter, adapter_strength=adapter_strength)
        self.sub_enc = TemporalEncoder(feature_dim, d_model, nhead, num_layers, dropout,
                                       use_adapter=use_adapter, adapter_strength=adapter_strength)
        self.aligner = LagAwareAligner(d_model, window=window, dropout=dropout, temperature=1.0,
                                       use_gauss=(int(use_gaussian_bias)==1), learn_temp=learn_temp)

        in_feats = 0
        in_feats += 4 * d_model  # S, Ew, |.|, ⊙
        if self.use_dynamics:
            in_feats += 3 * d_model  # ΔS, ΔEw, |Δ.|
        if self.use_cos:
            in_feats += 1
        if self.use_lag_norm:
            in_feats += 1

        self.disc_proj = nn.Sequential(
            nn.Linear(in_feats, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.GELU()
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.pool_attn = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        pooled_dim = 0
        if pooling in ("attn", "attn_mean_max","attn_mean_max_std"): pooled_dim += d_model
        if pooling in ("mean", "attn_mean_max","attn_mean_max_std"): pooled_dim += d_model
        if pooling in ("max", "attn_mean_max","attn_mean_max_std"): pooled_dim += d_model
        if pooling in ("attn_mean_max_std",): pooled_dim += d_model  # std
        self.classifier = nn.Linear(pooled_dim, num_classes)

    @staticmethod
    def _temporal_diff(X):
        d = X[:, 1:, :] - X[:, :-1, :]
        zero = torch.zeros_like(d[:, :1, :])
        return torch.cat([zero, d], dim=1)

    def _make_discrepancy(self, S, Ew, lag):
        feats = []
        diff = S - Ew
        prod = S * Ew
        absd = diff.abs()
        feats += [S, Ew, absd, prod]
        if self.use_dynamics:
            dS = self._temporal_diff(S); dE = self._temporal_diff(Ew)
            dabs = (dS - dE).abs()
            feats += [dS, dE, dabs]
        if self.use_cos:
            Sn = F.normalize(S, dim=-1); En = F.normalize(Ew, dim=-1)
            cos = (Sn * En).sum(dim=-1, keepdim=True)
            feats += [cos]
        if self.use_lag_norm:
            if lag.dim() == 3:
                lag = lag.squeeze()
                if lag.dim() == 1: lag = lag.unsqueeze(0)
            elif lag.dim() > 3:
                lag = lag.squeeze()
            lag_feat = lag.unsqueeze(-1)
            lag_norm = (lag_feat / (self.aligner.window + 1e-6)).clamp(0, 1)
            feats += [lag_norm]
        x = torch.cat(feats, dim=-1)
        return self.disc_proj(x)

    def _attn_pool(self, H, mask):
        scores = self.pool_attn(H).squeeze(-1)
        m = _to_bt_mask(mask, T=H.size(1))
        if m is not None: scores = scores.masked_fill(~m, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        return torch.bmm(attn.unsqueeze(1), H).squeeze(1)

    def _direct_align(self, E, S):
        B, T_e, C = E.shape
        T_s = S.size(1)
        if T_e == T_s: return E
        if T_e > T_s:  return E[:, :T_s, :]
        pad = torch.zeros(B, T_s - T_e, C, device=E.device, dtype=E.dtype)
        return torch.cat([E, pad], dim=1)

    def extract_embedding(self, exp, sub, exp_mask=None, sub_mask=None):
        logits, inter = self.forward(exp, sub, exp_mask, sub_mask, return_intermediates=True)
        H = self.final_norm(self._make_discrepancy(inter["S"], inter["Ew"], inter["lag"]))
        pooled = []
        if self.pooling in ("attn", "attn_mean_max","attn_mean_max_std"):
            pooled.append(self._attn_pool(H, sub_mask))
        if self.pooling in ("mean", "attn_mean_max","attn_mean_max_std"):
            pooled.append(H.mean(dim=1))
        if self.pooling in ("max", "attn_mean_max","attn_mean_max_std"):
            pooled.append(H.max(dim=1).values)
        if self.pooling in ("attn_mean_max_std",):
            stdp = H.std(dim=1)
            pooled.append(stdp)
        Z = torch.cat(pooled, dim=-1)
        return Z

    def forward(self, exp, sub, exp_mask=None, sub_mask=None, return_intermediates=False):
        E = self.exp_enc(exp, exp_mask)
        S = self.sub_enc(sub, sub_mask)

        if self.align_mode == "direct":
            Ew = self._direct_align(E, S)
            attn = None
            T_e = exp.size(1); T_s = sub.size(1)
            if T_e >= T_s:
                exp_t = torch.arange(T_s, device=E.device).float().view(1, T_s)
            else:
                exp_t = torch.cat([torch.arange(T_e, device=E.device).float(),
                                torch.full((T_s - T_e,), T_e - 1, device=E.device).float()]).view(1, T_s)
            exp_t = exp_t.expand(E.size(0), -1)  # (B, T_s)
            u_idx = torch.arange(T_s, device=E.device).float().view(1, T_s).expand(E.size(0), -1)
            lag = (u_idx - exp_t).clamp(min=0)   # (B, T_s)
        else:
            Ew, attn, lag = self.aligner(E, S, exp_mask, sub_mask)

        H = self._make_discrepancy(S, Ew, lag)
        H = self.final_norm(H)

        pooled = []
        if self.pooling in ("attn", "attn_mean_max","attn_mean_max_std"):
            pooled.append(self._attn_pool(H, sub_mask))
        if self.pooling in ("mean", "attn_mean_max","attn_mean_max_std"):
            pooled.append(H.mean(dim=1))
        if self.pooling in ("max", "attn_mean_max","attn_mean_max_std"):
            pooled.append(H.max(dim=1).values)
        if self.pooling in ("attn_mean_max_std",):
            pooled.append(H.std(dim=1))

        Z = torch.cat(pooled, dim=-1)
        logits = self.classifier(Z)

        if return_intermediates:
            return logits, {"attn": attn, "lag": lag, "Ew": Ew, "S": S, "E": E}
        return logits


# ----- Optional TD diffusion aux head -----

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / half))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1: emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

class FiLMBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cond_dim, channels*2), nn.GELU(), nn.Linear(channels*2, channels*2))
    def forward(self, x, cond):
        gb = self.mlp(cond); C = x.size(1)
        gamma, beta = gb[:, :C], gb[:, C:]
        return x * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)

class TemporalUNetSmall(nn.Module):
    def __init__(self, channels: int, cond_dim: int, hidden: int = 256):
        super().__init__()
        self.in_conv = nn.Conv1d(channels, hidden, 3, padding=1)
        self.film1 = FiLMBlock(hidden, cond_dim)
        self.down = nn.Conv1d(hidden, hidden, 4, stride=2, padding=1)
        self.mid = nn.Sequential(nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(), nn.Conv1d(hidden, hidden, 3, padding=1))
        self.film2 = FiLMBlock(hidden, cond_dim)
        self.up = nn.ConvTranspose1d(hidden, hidden, 4, stride=2, padding=1)
        self.out = nn.Conv1d(hidden, channels, 3, padding=1)
    def forward(self, x, cond):
        h = self.in_conv(x); h = self.film1(h, cond); d = self.down(h)
        m = self.mid(d); m = self.film2(m, cond); u = self.up(m)
        if u.size(-1) != h.size(-1):
            if u.size(-1) > h.size(-1): u = u[:, :, :h.size(-1)]
            else: u = F.pad(u, (0, h.size(-1)-u.size(-1)))
        return self.out(h + u)

class TDConditionalDiffusionHead(nn.Module):
    def __init__(self, d_model: int, T: int = 200):
        super().__init__()
        self.T = T
        self.time_embed = SinusoidalTimeEmbedding(d_model)
        self.cond_proj = nn.Linear(d_model*2 + d_model, d_model)
        self.denoiser = TemporalUNetSmall(channels=d_model, cond_dim=d_model, hidden=max(128, d_model*2))
        beta_start, beta_end = 1e-4, 2e-2
        betas = torch.linspace(beta_start, beta_end, T).float()
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas); self.register_buffer('alphas', alphas); self.register_buffer('alpha_bar', alpha_bar)
    def _q_sample(self, x0, t, noise):
        ab = self.alpha_bar[t].view(-1,1,1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise
    def reconstruct_x0(self, x_t, t, eps_hat):
        ab = self.alpha_bar[t].view(-1,1,1)
        return (x_t - (1 - ab).sqrt() * eps_hat) / (ab.sqrt() + 1e-8)
    def loss(self, S_enc, Ew_enc, mask=None):
        B, T, C = S_enc.shape; device = S_enc.device
        t = torch.randint(0, self.T, (B,), device=device)
        noise = torch.randn_like(S_enc)
        x_t = self._q_sample(S_enc, t, noise)
        x_ct = x_t.transpose(1,2).contiguous()
        S_g = S_enc.mean(dim=1); Ew_g = Ew_enc.mean(dim=1); t_emb = self.time_embed(t)
        cond = self.cond_proj(torch.cat([S_g, Ew_g, t_emb], dim=-1))
        eps_hat = self.denoiser(x_ct, cond).transpose(1,2).contiguous()
        if mask is not None:
            m = _to_bt_mask(mask, T=T).unsqueeze(-1).float()
            mse = ((eps_hat - noise) ** 2 * m).sum() / (m.sum() * C + 1e-8)
        else:
            mse = F.mse_loss(eps_hat, noise)
        return mse


# ---------- Metrics ----------

def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: Optional[int] = None) -> float:
    if y_true.size == 0: return 0.0
    if num_classes is None:
        num_classes = int(max(int(y_true.max()), int(y_pred.max()))) + 1 if y_true.size > 0 else 2
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = (2*tp + fp + fn)
        f1s.append(0.0 if denom == 0 else (2*tp) / denom)
    return float(np.mean(f1s))

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 2):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


# ---------- Train / Eval ----------

@dataclass
class TrainCfg:
    epochs: int = 60
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    lag_window: int = 12
    use_diffusion: int = 0
    diffusion_lambda: float = 0.05
    diffusion_steps: int = 200
    td_label: int = 0
    td_ratio: float = 1.0
    clip_grad: float = 5.0
    val_ratio: float = 0.15
    use_gaussian_bias: int = 1
    use_dynamics: int = 1
    use_cos: int = 1
    use_lag_norm: int = 1
    pooling: str = "attn_mean_max"
    align_mode: str = "lagaware"
    viz_samples: int = 1
    use_adapter: int = 0
    adapter_strength: str = "light"
    learn_temp: int = 1
    subject_key: Optional[str] = None


def run_fold(model, train_loader, val_loader, test_loader, device, cfg: TrainCfg, save_dir: str, log_every: int = 5):
    ensure_dir(save_dir)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss()

    hist_path = os.path.join(save_dir, "history.jsonl")
    best_state = None
    best_key = -1.0
    best_epoch = -1

    def eval_loader(loader, return_preds=False):
        model.eval()
        n_correct = n_total = 0
        ys, ps = [], []
        with torch.no_grad():
            for b in loader:
                xE, xS = b['exp'].to(device), b['sub'].to(device)
                mE = b.get('exp_mask', None); mS = b.get('sub_mask', None)
                if mE is not None: mE = mE.to(device)
                if mS is not None: mS = mS.to(device)
                logits = model(xE, xS, mE, mS)
                pred = logits.argmax(dim=-1)
                lab = b['label'].to(device).view(-1).long()
                n_correct += (pred == lab).sum().item(); n_total += lab.numel()
                ys.append(lab.cpu().numpy()); ps.append(pred.cpu().numpy())
        acc = n_correct / max(1, n_total)
        y_true = np.concatenate(ys) if len(ys) > 0 else np.array([])
        y_pred = np.concatenate(ps) if len(ps) > 0 else np.array([])
        f1 = macro_f1_score(y_true, y_pred)
        if return_preds:
            return acc, f1, y_true, y_pred
        return acc, f1

    with open(hist_path, "w", encoding="utf-8") as f:
        for epoch in range(1, cfg.epochs+1):
            model.train()
            t0 = time.time()
            loss_sum = ce_sum = diff_sum = 0.0
            n = 0
            for b in train_loader:
                xE, xS = b['exp'].to(device), b['sub'].to(device)
                mE = b.get('exp_mask', None); mS = b.get('sub_mask', None)
                if mE is not None: mE = mE.to(device)
                if mS is not None: mS = mS.to(device)
                lab = b['label'].to(device).view(-1).long()

                logits = model(xE, xS, mE, mS)
                loss_ce = ce(logits, lab)

                loss_diff = torch.tensor(0.0, device=device)
                if cfg.use_diffusion and hasattr(model, "cls"):
                    E = model.cls.exp_enc(xE, mE); S = model.cls.sub_enc(xS, mS)
                    Ew, _, _ = model.cls.aligner(E, S, mE, mS)
                    if hasattr(model, "diff"):
                        loss_diff = model.diff.loss(S, Ew, mS)

                loss = loss_ce + cfg.diffusion_lambda * loss_diff
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.clip_grad and cfg.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.clip_grad)
                optimizer.step()

                bs = lab.size(0)
                loss_sum += float(loss.item()) * bs
                ce_sum += float(loss_ce.item()) * bs
                diff_sum += float(loss_diff.item()) * bs
                n += bs

            va_acc, va_f1 = eval_loader(val_loader)
            te_acc, te_f1 = eval_loader(test_loader)

            rec = dict(
                epoch=epoch,
                train_loss=loss_sum/max(1,n),
                train_ce=ce_sum/max(1,n),
                train_diff=diff_sum/max(1,n),
                val_acc=va_acc, val_f1=va_f1,
                test_acc=te_acc, test_f1=te_f1,
                epoch_sec=round(time.time()-t0,2)
            )
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()

            key = va_f1 if not math.isnan(va_f1) else va_acc
            if key > best_key:
                best_key = key
                best_epoch = epoch
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save(best_state, os.path.join(save_dir, "best_model.pth"))

            if (epoch % log_every) == 0 or epoch == 1 or epoch == cfg.epochs:
                print(json.dumps({"fold_dir": save_dir, **rec}, ensure_ascii=False))

    if best_state is not None:
        model.load_state_dict(best_state)
    te_acc, te_f1, y_true, y_pred = eval_loader(test_loader, return_preds=True)

    save_csv(list(zip(y_true.tolist(), y_pred.tolist())), os.path.join(save_dir, "preds.csv"), header=["y_true","y_pred"])
    cm = confusion_matrix(y_true, y_pred, num_classes=2)
    save_csv(cm.tolist(), os.path.join(save_dir, "confusion_matrix.csv"))
    plt.figure(figsize=(3.5,3.2))
    plt.imshow(cm, cmap="Blues"); plt.colorbar(fraction=0.046)
    plt.xlabel("Pred"); plt.ylabel("True")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i,j]), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=180); plt.close()

    save_json({"best_val": best_key, "best_epoch": best_epoch, "final_test_acc": te_acc, "final_test_f1": te_f1},
              os.path.join(save_dir, "summary.json"))
    return best_key, te_acc, te_f1


# ---------- Build loaders & splits ----------

def build_dataloaders(dataset, idx_train, idx_val, idx_test, batch_size: int):
    if has_d3 and hasattr(d3, "collate_fn_with_padding"):
        collate = d3.collate_fn_with_padding
    else:
        collate = default_collate_with_padding
    train_loader = DataLoader(Subset(dataset, idx_train), batch_size=batch_size, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(Subset(dataset, idx_val),   batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(Subset(dataset, idx_test),  batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader, test_loader

def count_by_class(labels, indices):
    arr = np.array([int(labels[i]) for i in indices])
    uniq, cnt = np.unique(arr, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq, cnt)}

def stratified_kfold_indices(labels: List[int], k: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    labels = np.array(labels).astype(int)
    uniq = np.unique(labels)
    cls2idx = {c: np.where(labels == c)[0].tolist() for c in uniq}
    for c in uniq: rng.shuffle(cls2idx[c])
    folds_by_cls = {c: np.array_split(cls2idx[c], k) for c in uniq}
    folds = []
    for i in range(k):
        test_idx = np.concatenate([folds_by_cls[c][i] for c in uniq]).tolist()
        trainval_idx = np.setdiff1d(np.arange(len(labels)), test_idx).tolist()
        folds.append((trainval_idx, test_idx))
    return folds

def try_get_d3_splits(dataset, k: int):
    if not has_d3: return None
    candidates = ["SPLITS", "FOLDS", "KFOLDS", "kfold_indices", "fold_indices", "KF_5", "K5_SPLITS"]
    for name in candidates:
        if hasattr(d3, name):
            obj = getattr(d3, name)
            try:
                if isinstance(obj, (list, tuple)) and len(obj) == k:
                    ok = True
                    for it in obj:
                        if not (isinstance(it, (list, tuple)) and len(it) == 2):
                            ok = False; break
                    if ok:
                        print(f"[INFO] Using splits from Diffusion_3.{name}")
                        return obj
            except Exception: pass
    fn_candidates = ["get_kfold_splits", "build_kfold_indices", "make_kfold_splits", "get_folds"]
    for fn in fn_candidates:
        if hasattr(d3, fn):
            try:
                obj = getattr(d3, fn)(dataset, k)
                if isinstance(obj, (list, tuple)) and len(obj) == k:
                    print(f"[INFO] Using splits from Diffusion_3.{fn}()")
                    return obj
            except Exception as e:
                print(f"[WARN] Calling Diffusion_3.{fn} failed:", e)
    return None

def build_splits(dataset, k: int, seed: int, use_d3_splits: int, splits_json: Optional[str], val_ratio: float, subject_key: Optional[str] = None):
    # labels
    labels = []
    for i in range(len(dataset)):
        li = dataset[i]['label']
        labels.append(int(li.item()) if torch.is_tensor(li) else int(li))

    # explicit json
    if splits_json and os.path.exists(splits_json):
        with open(splits_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        src = f"external:{splits_json}"
        folds_out = []
        sub2idx = None

        for fold in obj.get("folds", []):
            if all(key in fold for key in ("train","val","test")):
                tr, va, te = list(map(int, fold["train"])), list(map(int, fold["val"])), list(map(int, fold["test"]))
                rep = {"fold": len(folds_out), "source": src + " (indices)",
                       "train_counts": count_by_class(labels, tr), "val_counts": count_by_class(labels, va),
                       "test_counts": count_by_class(labels, te), "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
                folds_out.append({"report": rep, "train": tr, "val": va, "test": te}); continue

            if ("trainval_idx" in fold) and ("test_idx" in fold):
                trainval_idx = list(map(int, fold["trainval_idx"])); test_idx = list(map(int, fold["test_idx"]))
                rng = np.random.default_rng(seed)
                y_tv = np.array([labels[j] for j in trainval_idx])
                idx_by_c = defaultdict(list)
                for j, idx in enumerate(trainval_idx): idx_by_c[y_tv[j]].append(idx)
                for c in idx_by_c: rng.shuffle(idx_by_c[c])
                idx_train, idx_val = [], []
                for c, lst in idx_by_c.items():
                    n = len(lst); n_val = max(1, int(round(val_ratio * n)))
                    idx_val += lst[:n_val]; idx_train += lst[n_val:]
                rep = {"fold": len(folds_out), "source": src + " (trainval+test -> split val)",
                       "train_counts": count_by_class(labels, idx_train), "val_counts": count_by_class(labels, idx_val),
                       "test_counts": count_by_class(labels, test_idx),
                       "n_train": len(idx_train), "n_val": len(idx_val), "n_test": len(test_idx)}
                folds_out.append({"report": rep, "train": idx_train, "val": idx_val, "test": test_idx}); continue

            if all(key in fold for key in ("train_subjects","val_subjects","test_subjects")):
                if sub2idx is None: sub2idx = _build_subject_index_map(dataset, subject_key=subject_key)
                def s2i(lst):
                    out, miss = [], []
                    for sid in lst:
                        sid = str(sid)
                        if sid in sub2idx: out += sub2idx[sid]
                        else: miss.append(sid)
                    return sorted(out), miss
                tr_idx, miss_tr = s2i(fold["train_subjects"])
                va_idx, miss_va = s2i(fold["val_subjects"])
                te_idx, miss_te = s2i(fold["test_subjects"])
                rep = {"fold": len(folds_out), "source": src + " (subjects -> indices)",
                       "train_counts": count_by_class(labels, tr_idx), "val_counts": count_by_class(labels, va_idx),
                       "test_counts": count_by_class(labels, te_idx),
                       "n_train": len(tr_idx), "n_val": len(va_idx), "n_test": len(te_idx),
                       "missing_subjects": {"train": miss_tr, "val": miss_va, "test": miss_te}}
                folds_out.append({"report": rep, "train": tr_idx, "val": va_idx, "test": te_idx}); continue

        if not folds_out:
            raise KeyError(f"[splits_json] No recognized fold format in {splits_json}.")
        return folds_out, labels

    # Diffusion_3 or internal
    folds = try_get_d3_splits(dataset, k) if use_d3_splits else None
    src = "Diffusion_3" if folds is not None else "internal_stratified_kfold"
    if folds is None:
        folds = stratified_kfold_indices(labels, k=k, seed=seed)

    rng = np.random.default_rng(seed)
    data = []
    for i, (trainval_idx, test_idx) in enumerate(folds):
        y_trainval = np.array([labels[j] for j in trainval_idx])
        idx_by_c = defaultdict(list)
        for j, idx in enumerate(trainval_idx): idx_by_c[y_trainval[j]].append(idx)
        for c in idx_by_c: rng.shuffle(idx_by_c[c])
        idx_train, idx_val = [], []
        for c, lst in idx_by_c.items():
            n = len(lst); n_val = max(1, int(round(val_ratio * n)))
            idx_val += lst[:n_val]; idx_train += lst[n_val:]
        rep = {"fold": i, "source": src,
               "train_counts": count_by_class(labels, idx_train), "val_counts": count_by_class(labels, idx_val),
               "test_counts": count_by_class(labels, test_idx),
               "n_train": len(idx_train), "n_val": len(idx_val), "n_test": len(test_idx)}
        data.append({"report": rep, "train": idx_train, "val": idx_val, "test": test_idx})
    return data, labels


# ---------- Visualization ----------

@torch.no_grad()
def visualize_samples(model, dataset, fold_split, device, out_dir, max_per_class=1, subject_key: Optional[str]=None):
    ensure_dir(out_dir)
    # 1) 训练曲线（若 history.jsonl 存在于上级 fold 目录）
    fold_dir = os.path.dirname(out_dir.rstrip("/"))
    hist_path = os.path.join(fold_dir, "history.jsonl")
    if os.path.exists(hist_path):
        ep, tr, ce, df, va, te = [], [], [], [], [], []
        with open(hist_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                ep.append(obj["epoch"]); tr.append(obj["train_loss"]); ce.append(obj["train_ce"]); df.append(obj["train_diff"]); va.append(obj["val_f1"]); te.append(obj["test_f1"])
        plt.figure(figsize=(6.5,3.2))
        plt.plot(ep, tr, label="train_loss"); plt.plot(ep, ce, label="train_ce")
        if np.max(np.array(df)) > 0: plt.plot(ep, df, label="train_diff")
        plt.plot(ep, va, label="val_f1"); plt.plot(ep, te, label="test_f1")
        plt.legend(); plt.xlabel("epoch"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=180); plt.close()

    # 2) 采样 ASD/TD
    asd_idxs, td_idxs = [], []
    for idx in fold_split["test"]:
        lab = int(dataset[idx]['label']) if not torch.is_tensor(dataset[idx]['label']) else int(dataset[idx]['label'].item())
        if lab == 1 and len(asd_idxs) < max_per_class: asd_idxs.append(idx)
        if lab == 0 and len(td_idxs) < max_per_class: td_idxs.append(idx)
        if len(asd_idxs) >= max_per_class and len(td_idxs) >= max_per_class: break
    picks = [("ASD", i) for i in asd_idxs] + [("TD", i) for i in td_idxs]

    def _forward_intermediates(sample):
        exp = sample['exp'].unsqueeze(0).to(device)
        sub = sample['sub'].unsqueeze(0).to(device)
        mE = sample.get('exp_mask', None); mS = sample.get('sub_mask', None)
        if mE is not None: mE = mE.unsqueeze(0).to(device)
        if mS is not None: mS = mS.unsqueeze(0).to(device)
        logits, inter = model(exp, sub, mE, mS, return_intermediates=True)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return inter, probs, (mS.squeeze(0).cpu().numpy() if mS is not None else None)

    for tag, idx in picks:
        sample = dataset[idx]
        inter, probs, mS = _forward_intermediates(sample)
        S = inter["S"].squeeze(0).cpu().numpy()
        Ew = inter["Ew"].squeeze(0).cpu().numpy()
        lag = inter["lag"].squeeze(0).cpu().numpy()
        attn = inter["attn"]

        # 2.1 lag 曲线（只画有效时间步，叠加平滑与均值）
        if mS is None: mS = np.ones((len(lag),), dtype=bool)
        t = np.arange(len(lag))[mS.astype(bool)]
        y = lag[mS.astype(bool)]
        plt.figure(figsize=(6,3))
        if len(t) > 0:
            plt.plot(t, y, alpha=0.35, label="lag")
            ys = np.asarray(y, dtype=float).reshape(-1)
            k = max(3, len(ys) // 30)
            plt.plot(t, moving_avg(ys, k=k), label="smoothed")
            plt.hlines([0], t.min(), t.max(), linestyles='dashed', colors='gray', alpha=0.5)
            plt.hlines([np.mean(y)], t.min(), t.max(), linestyles='dotted', colors='C3', label=f"mean={np.mean(y):.2f}")
        plt.title(f"lag curve [{tag}] idx={idx}  prob(TD)={probs[0]:.2f}, prob(ASD)={probs[1]:.2f}")
        plt.xlabel("SUB time"); plt.ylabel("lag (frames)"); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"lag_curve_{tag}_{idx}.png"), dpi=180); plt.close()

        # 2.2 对齐热图 + top1 路径 + 窗口边界
        if attn is not None:
            A = inter["attn"].squeeze(0).cpu().numpy()  # (T_s, T_e)
            plt.figure(figsize=(6.0,4.8))
            plt.imshow(A, aspect='auto', origin='lower')
            plt.colorbar(fraction=0.046); plt.xlabel("EXP time"); plt.ylabel("SUB time")
            cols = np.argmax(A, axis=1); rows = np.arange(A.shape[0])
            plt.plot(cols, rows, 'w-', linewidth=1.0, alpha=0.9, label="argmax path")
            # 窗口上/下边界（te in [u-window, u]）
            w = max(1, getattr(model.aligner, "window", 12))
            up = rows
            low = np.maximum(0, rows - w)
            plt.plot(up, rows, 'w--', linewidth=0.8, alpha=0.6, label="diag (u)")
            plt.plot(low, rows, 'w--', linewidth=0.8, alpha=0.6, label=f"u-{w}")
            plt.title(f"alignment heatmap [{tag}] idx={idx}")
            plt.legend(loc='upper right', fontsize=8)
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"alignment_heatmap_{tag}_{idx}.png"), dpi=200); plt.close()

        # 2.3 自相似一致性
        def sim_mat(X):
            X = torch.tensor(X)
            Xn = F.normalize(X, dim=-1)
            M = torch.einsum('td,Sd->tS', Xn, Xn).cpu().numpy()
            return M
        S_sim = sim_mat(S); Ew_sim = sim_mat(Ew)
        vmax = np.percentile(np.concatenate([S_sim, Ew_sim]).ravel(), 99)
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1); plt.imshow(S_sim, vmin=0, vmax=vmax); plt.title(f"SUB self-sim [{tag}]"); plt.xlabel('t'); plt.ylabel('t'); plt.colorbar(fraction=0.046)
        plt.subplot(1,2,2); plt.imshow(Ew_sim, vmin=0, vmax=vmax); plt.title(f"Aligned-EXP self-sim [{tag}]"); plt.xlabel('t'); plt.ylabel('t'); plt.colorbar(fraction=0.046)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"sim_consistency_{tag}_{idx}.png"), dpi=200); plt.close()

    # 3) raw vs learned 2D PCA 散点：取该折测试集全部样本
    def pca_2d(X):
        X = np.asarray(X, dtype=np.float64)
        Xc = X - X.mean(0, keepdims=True)
        U, Sg, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Xc @ Vt[:2].T

    # 收集嵌入与原始粗特征
    embs, raws, labs = [], [], []
    for idx in fold_split["test"]:
        s = dataset[idx]
        lab = int(s['label']) if not torch.is_tensor(s['label']) else int(s['label'].item())
        with torch.no_grad():
            exp = s['exp'].unsqueeze(0).to(device); sub = s['sub'].unsqueeze(0).to(device)
            mE = s.get('exp_mask', None); mS = s.get('sub_mask', None)
            if mE is not None: mE = mE.unsqueeze(0).to(device)
            if mS is not None: mS = mS.unsqueeze(0).to(device)
            z = model.extract_embedding(exp, sub, mE, mS).squeeze(0).cpu().numpy()
            # raw baseline：时间均值 + 绝对差均值 + cos 均值
            Ee = model.exp_enc(s['exp'].unsqueeze(0).to(device), mE).squeeze(0).cpu().numpy()
            Se = model.sub_enc(s['sub'].unsqueeze(0).to(device), mS).squeeze(0).cpu().numpy()
            Te = min(Ee.shape[0], Se.shape[0])
            d = np.abs(Se[:Te] - Ee[:Te]).mean(axis=0)
            cos = (Se[:Te]/(np.linalg.norm(Se[:Te],axis=1,keepdims=True)+1e-6) * (Ee[:Te]/(np.linalg.norm(Ee[:Te],axis=1,keepdims=True)+1e-6))).sum(axis=1).mean()
            rawv = np.concatenate([Se.mean(axis=0), Ee.mean(axis=0), d, np.array([cos])], axis=0)
        embs.append(z); raws.append(rawv); labs.append(lab)
    embs = np.stack(embs); raws = np.stack(raws); labs = np.array(labs)
    e2 = pca_2d(embs); r2 = pca_2d(raws)
    plt.figure(figsize=(10,4))
    for i,(name,pts) in enumerate([("RAW", r2), ("LEARNED", e2)]):
        plt.subplot(1,2,i+1)
        plt.scatter(pts[labs==0,0], pts[labs==0,1], s=16, alpha=0.7, label="TD")
        plt.scatter(pts[labs==1,0], pts[labs==1,1], s=16, alpha=0.7, label="ASD")
        plt.title(f"{name} 2D PCA"); plt.axis('equal'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "scatter_raw_vs_learned.png"), dpi=200); plt.close()

    # 4) 扩散可视化：误差直方图 + 单样本时间曲线
    if hasattr(model, "diff") and hasattr(model, "cls"):
        errs_td, errs_asd = [], []
        for idx in fold_split["test"]:
            s = dataset[idx]
            lab = int(s['label']) if not torch.is_tensor(s['label']) else int(s['label'].item())
            exp = s['exp'].unsqueeze(0).to(device); sub = s['sub'].unsqueeze(0).to(device)
            mE = s.get('exp_mask', None); mS = s.get('sub_mask', None)
            if mE is not None: mE = mE.unsqueeze(0).to(device)
            if mS is not None: mS = mS.unsqueeze(0).to(device)
            with torch.no_grad():
                E = model.cls.exp_enc(exp, mE); S = model.cls.sub_enc(sub, mS)
                Ew, _, _ = model.cls.aligner(E, S, mE, mS)
                B,T,C = S.shape
                t = torch.randint(0, model.diff.T, (B,), device=E.device)
                noise = torch.randn_like(S)
                x_t = model.diff._q_sample(S, t, noise)
                cond = model.diff.cond_proj(torch.cat([S.mean(1), Ew.mean(1), model.diff.time_embed(t)], dim=-1))
                eps_hat = model.diff.denoiser(x_t.transpose(1,2), cond).transpose(1,2)
                x0_hat = model.diff.reconstruct_x0(x_t, t, eps_hat)
                mse = F.mse_loss(x0_hat, S, reduction='none').mean(dim=(1,2)).item()
                if lab==0: errs_td.append(mse)
                else: errs_asd.append(mse)
        if len(errs_td)+len(errs_asd) > 0:
            plt.figure(figsize=(6,3.2))
            if len(errs_td)>0: plt.hist(errs_td, bins=20, alpha=0.6, label='TD')
            if len(errs_asd)>0: plt.hist(errs_asd, bins=20, alpha=0.6, label='ASD')
            plt.xlabel("Diffusion recon MSE"); plt.ylabel("count"); plt.legend()
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, "diffusion_mse_hist.png"), dpi=180); plt.close()


# ---------- Orchestration ----------

def run_combo(data_path: str, feature: str, cfg: TrainCfg, root_out: str, seed: int,
              use_d3_splits: int, splits_json: Optional[str]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    assert has_d3 and hasattr(d3, "AutismDataset"), "AutismDataset not found in Diffusion_3.py"
    dataset = d3.AutismDataset(data_path, feature_type=feature)

    # feature 特殊默认：flow 系列自动打开适配器
    use_adapter = cfg.use_adapter
    adapter_strength = cfg.adapter_strength
    fe_lower = feature.lower()
    use_adapter = 0; adapter_strength = cfg.adapter_strength

    tag = f"lag{cfg.lag_window}_diff{cfg.use_diffusion}_gb{cfg.use_gaussian_bias}_dyn{cfg.use_dynamics}_cos{cfg.use_cos}_lag{cfg.use_lag_norm}_pool-{cfg.pooling}_align-{cfg.align_mode}_adapt{use_adapter}-{adapter_strength}"
    exp_dir = os.path.join(root_out, feature, tag, f"seed{seed}")
    ensure_dir(exp_dir)

    split_list, labels = build_splits(dataset, k=5, seed=seed, use_d3_splits=use_d3_splits,
                                      splits_json=splits_json, val_ratio=cfg.val_ratio, subject_key=cfg.subject_key)

    # 保存 splits indices + subjects（检查无交叉）
    all_subjects = []
    for d in split_list:
        tr_sub = _indices_to_subjects(dataset, d["train"], cfg.subject_key)
        va_sub = _indices_to_subjects(dataset, d["val"], cfg.subject_key)
        te_sub = _indices_to_subjects(dataset, d["test"], cfg.subject_key)
        inter = sorted(list((set(tr_sub)&set(va_sub)) | (set(tr_sub)&set(te_sub)) | (set(va_sub)&set(te_sub))))
        d["subjects"] = {"train_subjects": tr_sub, "val_subjects": va_sub, "test_subjects": te_sub, "intersection": inter}
        all_subjects.append({"train_subjects": tr_sub, "val_subjects": va_sub, "test_subjects": te_sub, "intersection": inter})
    save_json({"folds": all_subjects}, os.path.join(exp_dir, "splits_subjects.json"))

    splits_overall = {
        "source": split_list[0]["report"]["source"],
        "folds": [{"fold": d["report"]["fold"], "train": d["train"], "val": d["val"], "test": d["test"],
                   "report": d["report"], "subjects": d["subjects"]} for d in split_list]
    }
    save_json(splits_overall, os.path.join(exp_dir, "splits_overall.json"))

    all_res = []
    for fold_data in split_list:
        i = fold_data["report"]["fold"]
        fold_dir = os.path.join(exp_dir, f"fold_{i}")
        ensure_dir(fold_dir)
        save_json(fold_data["report"], os.path.join(fold_dir, "split_report.json"))
        save_json(fold_data["subjects"], os.path.join(fold_dir, "subjects_split.json"))

        # 交叉检查：如有 subject 交叉，强制报错
        if fold_data["subjects"]["intersection"]:
            raise RuntimeError(f"[LeakCheck] subjects overlap in fold {i}: {fold_data['subjects']['intersection']}")

        train_loader, val_loader, test_loader = build_dataloaders(
            dataset, fold_data["train"], fold_data["val"], fold_data["test"], cfg.batch_size
        )

        in_dim = dataset[0]['exp'].shape[-1]
        base_model = ImitationLagAwareClassifier(
            feature_dim=in_dim, d_model=128, window=cfg.lag_window,
            num_classes=2, dropout=cfg.dropout,
            use_gaussian_bias=cfg.use_gaussian_bias,
            use_dynamics=cfg.use_dynamics,
            use_cos=cfg.use_cos,
            use_lag_norm=cfg.use_lag_norm,
            pooling=cfg.pooling,
            align_mode=cfg.align_mode,
            use_adapter=use_adapter,
            adapter_strength=adapter_strength,
            learn_temp=cfg.learn_temp
        )
        if cfg.use_diffusion:
            diff_head = TDConditionalDiffusionHead(d_model=128, T=cfg.diffusion_steps)
            class _Wrap(nn.Module):
                def __init__(self, cls, diff): super().__init__(); self.cls = cls; self.diff = diff
                def forward(self, *a, **kw): return self.cls(*a, **kw)
                def state_dict(self, *a, **kw): 
                    sd = {"cls."+k:v for k,v in self.cls.state_dict().items()}
                    sd.update({"diff."+k:v for k,v in self.diff.state_dict().items()})
                    return sd
                def load_state_dict(self, sd):
                    cls_sd = {k.replace("cls.",""):v for k,v in sd.items() if k.startswith("cls.")}
                    diff_sd = {k.replace("diff.",""):v for k,v in sd.items() if k.startswith("diff.")}
                    self.cls.load_state_dict(cls_sd, strict=False); self.diff.load_state_dict(diff_sd, strict=False)
        else:
            class _Wrap(nn.Module):
                def __init__(self, cls): super().__init__(); self.cls = cls
                def forward(self, *a, **kw): return self.cls(*a, **kw)
                def state_dict(self, *a, **kw): return {"cls."+k:v for k,v in self.cls.state_dict().items()}
                def load_state_dict(self, sd):
                    cls_sd = {k.replace("cls.",""):v for k,v in sd.items() if k.startswith("cls.")}
                    self.cls.load_state_dict(cls_sd, strict=False)

        model = _Wrap(base_model, diff_head) if cfg.use_diffusion else _Wrap(base_model)

        best_val, te_acc, te_f1 = run_fold(model, train_loader, val_loader, test_loader, device, cfg, fold_dir)
        all_res.append({"fold": i, "best_val": best_val, "test_acc": te_acc, "test_f1": te_f1})

        if i == 0:
            viz_dir = os.path.join(exp_dir, "viz")
            visualize_samples(base_model.to(device), dataset, fold_data, device, viz_dir, max_per_class=cfg.viz_samples, subject_key=cfg.subject_key)

    accs = [r["test_acc"] for r in all_res]
    f1s = [r["test_f1"] for r in all_res]
    summary = {
        "feature": feature, "lag_window": cfg.lag_window, "use_diffusion": cfg.use_diffusion,
        "use_gaussian_bias": cfg.use_gaussian_bias, "use_dynamics": cfg.use_dynamics,
        "use_cos": cfg.use_cos, "use_lag_norm": cfg.use_lag_norm,
        "pooling": cfg.pooling, "align_mode": cfg.align_mode,
        "use_adapter": use_adapter, "adapter_strength": adapter_strength, "learn_temp": cfg.learn_temp,
        "seed": seed, "folds": all_res,
        "mean_test_acc": float(np.mean(accs)), "std_test_acc": float(np.std(accs)),
        "mean_test_f1": float(np.mean(f1s)), "std_test_f1": float(np.std(f1s))
    }
    save_json(summary, os.path.join(exp_dir, "summary_overall.json"))
    print(json.dumps({"combo_dir": exp_dir, **summary}, ensure_ascii=False))

    from classwise_viz import classwise_viz

    test_indices = _extract_test_indices(fold_data)
    viz_dir = os.path.join(exp_dir, "viz_class_avg")
    classwise_viz(model, dataset, test_indices, device, viz_dir,
                asd_label=1, grid=128, curve_len=200, lag_window=cfg.lag_window)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="autism_multimodal_dataset_20250726.pkl", help="Path used by Diffusion_3.AutismDataset")
    ap.add_argument("--features", type=str, default="skeleton,dense_flow,heatmap,sparse_flow", help="Comma-separated feature types")
    ap.add_argument("--lag_windows", type=str, default="12", help="Comma-separated lag windows (frames)")
    ap.add_argument("--use_diff", type=str, default="1", help="Comma-separated 0/1; if 1, enable TD diffusion aux loss")
    ap.add_argument("--kfolds", type=int, default=5)

    # training
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--diffusion_lambda", type=float, default=0.05)
    ap.add_argument("--diffusion_steps", type=int, default=200)
    ap.add_argument("--td_label", type=int, default=0)
    ap.add_argument("--td_ratio", type=float, default=1.0)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--root_out", type=str, default="./runs_gpt")
    ap.add_argument("--seed", type=int, default=42)

    # splits
    ap.add_argument("--use_d3_splits", type=int, default=1)
    ap.add_argument("--splits_json", type=str, default="splits_subjects_template.json")
    ap.add_argument("--subject_key", type=str, default="")  # optional

    # ablations / model
    ap.add_argument("--use_gaussian_bias", type=int, default=1)
    ap.add_argument("--use_dynamics", type=int, default=1)
    ap.add_argument("--use_cos", type=int, default=1)
    ap.add_argument("--use_lag_norm", type=int, default=1)
    ap.add_argument("--pooling", type=str, default="attn_mean_max", choices=["attn","mean","max","attn_mean_max","attn_mean_max_std"])
    ap.add_argument("--align_mode", type=str, default="lagaware", choices=["lagaware","direct"])
    ap.add_argument("--viz_samples", type=int, default=1)
    ap.add_argument("--use_adapter", type=int, default=0)
    ap.add_argument("--adapter_strength", type=str, default="light", choices=["auto","light","strong"])
    ap.add_argument("--learn_temp", type=int, default=1)

    args = ap.parse_args()
    set_seed(args.seed); ensure_dir(args.root_out)

    features = [s.strip() for s in args.features.split(",") if s.strip()]
    lag_windows = [int(s.strip()) for s in args.lag_windows.split(",") if s.strip()]
    use_diffs = [int(s.strip()) for s in args.use_diff.split(",") if s.strip()]
    subj_key = args.subject_key if args.subject_key else None

    manifest = []
    t0 = time.time()
    for feat in features:
        for lw in lag_windows:
            for ud in use_diffs:
                cfg = TrainCfg(
                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
                    dropout=args.dropout, lag_window=lw, use_diffusion=ud, diffusion_lambda=args.diffusion_lambda,
                    diffusion_steps=args.diffusion_steps, td_label=args.td_label, td_ratio=args.td_ratio,
                    val_ratio=args.val_ratio, use_gaussian_bias=args.use_gaussian_bias,
                    use_dynamics=args.use_dynamics, use_cos=args.use_cos, use_lag_norm=args.use_lag_norm,
                    pooling=args.pooling, align_mode=args.align_mode, viz_samples=args.viz_samples,
                    use_adapter=args.use_adapter, adapter_strength=args.adapter_strength, learn_temp=args.learn_temp,
                    subject_key=subj_key
                )
                run_combo(args.data_path, feat, cfg, args.root_out, args.seed,
                          use_d3_splits=args.use_d3_splits, splits_json=(args.splits_json or None))
                manifest.append({"feature": feat, "lag_window": lw, "use_diff": ud})

    save_json({"total_sec": round(time.time()-t0,2), "combos": manifest}, os.path.join(args.root_out, "manifest.json"))
    print("[DONE] All combos finished. Root:", args.root_out)


if __name__ == "__main__":
    main()


@torch.no_grad()
def visualize_classwise_alignment(model, dataset, fold_split, device, out_dir,
                                  grid: int = 128, curve_len: int = 200,
                                  asd_label: int = 1, lag_window: int = None):
    ensure_dir(out_dir)
    model.eval()
    # Helper
    def _to_tensor(x): return torch.as_tensor(x, dtype=torch.float32, device=device)

    def _encode_and_align(sample):
        exp = _to_tensor(sample['exp']).unsqueeze(0)
        sub = _to_tensor(sample['sub']).unsqueeze(0)
        mE  = sample.get('exp_mask', None)
        mS  = sample.get('sub_mask', None)
        if mE is None: mE = torch.ones(exp.shape[1], dtype=torch.bool)
        if mS is None: mS = torch.ones(sub.shape[1], dtype=torch.bool)
        mE = mE[None,:].to(device); mS = mS[None,:].to(device)
        E  = model.exp_enc(exp, mE); S = model.sub_enc(sub, mS)
        Ew, attn, lag = model.aligner(E, S, mE, mS)
        return attn[0].detach().cpu(), lag[0].detach().cpu()

    def _interp_curve(u_src, y_src, n=200):
        u = np.asarray(u_src).reshape(-1); y = np.asarray(y_src).reshape(-1)
        if u.size < 2: return np.linspace(0,1,n), np.zeros(n)
        xi = np.linspace(0,1,n); yi = np.interp(xi, u, y); return xi, yi

    heat = {'ASD': np.zeros((grid,grid), float), 'TD': np.zeros((grid,grid), float)}
    cnt  = {'ASD': 0, 'TD': 0}
    lag_curv, path_curv, lag_means = {'ASD': [], 'TD': []}, {'ASD': [], 'TD': []}, {'ASD': [], 'TD': []}

    test_idx = fold_split['test']
    for idx in test_idx:
        s = dataset[idx]
        cls = 'ASD' if int(s['label']) == asd_label else 'TD'
        attn, lag = _encode_and_align(s)
        Ts, Te = attn.shape

        a = attn.clamp(min=0); ssum = float(a.sum().item())
        if ssum <= 1e-12: continue
        a = (a / ssum).unsqueeze(0).unsqueeze(0)
        aG = F.interpolate(a, size=(grid, grid), mode='bilinear', align_corners=True)[0,0].numpy()
        heat[cls] += aG; cnt[cls] += 1

        attn_np = attn.numpy()
        denom = attn_np.sum(axis=1, keepdims=True) + 1e-8
        exp_t = (attn_np * np.linspace(0,1,Te)[None,:]).sum(axis=1) / denom.squeeze(1)
        u = np.linspace(0,1,Ts)
        _, pc = _interp_curve(u, exp_t, n=curve_len)
        path_curv[cls].append(pc)

        lag_np = lag.numpy().astype(float)
        if lag_window and lag_window > 0: lag_np = lag_np / float(lag_window)
        _, lc = _interp_curve(u, lag_np, n=curve_len)
        lag_curv[cls].append(lc); lag_means[cls].append(float(lag_np.mean()))

    for k in ('ASD','TD'):
        if cnt[k] > 0: heat[k] /= cnt[k]

    # heatmaps
    for cls in ('ASD','TD'):
        if cnt[cls] == 0: continue
        plt.figure(figsize=(6.4,5.4))
        vmax = float(heat[cls].max() + 1e-12)
        plt.imshow(heat[cls], origin='lower', cmap='viridis', vmin=0.0, vmax=vmax, extent=(0,1,0,1), aspect='auto')
        plt.colorbar(label='avg attention density')
        u = np.linspace(0,1,200); plt.plot(u,u,'--',color='white',alpha=0.85,linewidth=1.2,label='diag (u)')
        plt.title(f'class-averaged alignment heatmap [{cls}] (N={cnt[cls]})')
        plt.xlabel('EXP (normalized time)'); plt.ylabel('SUB (normalized time)')
        plt.legend(loc='upper left', framealpha=0.85); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'class_avg_alignment_heatmap_{cls}.png'), dpi=180); plt.close()

    def _plot_curves(curves, title, ylabel, fname):
        plt.figure(figsize=(9.2,3.8)); x = np.linspace(0,1,curve_len)
        for cls, color in (('ASD','#D32F2F'), ('TD','#1976D2')):
            if len(curves[cls]) == 0: continue
            arr = np.stack(curves[cls], axis=0); mean = arr.mean(0); std = arr.std(0)
            plt.fill_between(x, mean-std, mean+std, color=color, alpha=0.18, linewidth=0)
            plt.plot(x, mean, color=color, linewidth=2.0, label=f'{cls} (N={arr.shape[0]})')
        plt.xlabel('SUB (normalized time)'); plt.ylabel(ylabel); plt.title(title)
        plt.legend(loc='best'); plt.grid(alpha=0.25); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=180); plt.close()

    _plot_curves(lag_curv,  'Class-averaged lag curve (mean ± std)', 'lag (normalized)' if lag_window else 'lag (frames)', 'class_avg_lag_curve.png')
    _plot_curves(path_curv, 'Class-averaged expected EXP time (mean ± std)', 'E[t|attn] (normalized)', 'class_avg_path_curve.png')

    stats = {
        'counts': cnt,
        'lag_mean': {k: (float(np.mean(lag_means[k])) if len(lag_means[k]) else None) for k in ('ASD','TD')},
        'lag_std':  {k: (float(np.std(lag_means[k]))  if len(lag_means[k]) else None) for k in ('ASD','TD')},
        'notes': 'Averaged on normalized grids; curves are resampled to equal length.'
    }
    with open(os.path.join(out_dir, 'class_avg_alignment_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    return stats
