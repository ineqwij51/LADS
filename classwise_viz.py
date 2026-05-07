# -*- coding: utf-8 -*-
import os, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

__all__ = ["classwise_viz"]

@torch.no_grad()
def classwise_viz(model, dataset, test_indices, device, save_dir,
                  asd_label=1, grid=128, curve_len=200, lag_window=None):
 
    os.makedirs(save_dir, exist_ok=True)
    m = getattr(model, "cls", model)
    m.eval()

    def _to_tensor(x): return torch.as_tensor(x, dtype=torch.float32, device=device)

    def _encode_and_align(sample):
        exp = _to_tensor(sample["exp"]).unsqueeze(0)
        sub = _to_tensor(sample["sub"]).unsqueeze(0)
        em  = sample.get("exp_mask", None)
        sm  = sample.get("sub_mask", None)
        if em is None: em = np.ones((exp.shape[1],), dtype=bool)
        if sm is None: sm = np.ones((sub.shape[1],), dtype=bool)
        em = torch.as_tensor(em, device=device)[None, :]
        sm = torch.as_tensor(sm, device=device)[None, :]
        E   = m.exp_enc(exp, mask=em)
        S   = m.sub_enc(sub, mask=sm)
        Ew, attn, lag = m.aligner(E, S, exp_mask=em, sub_mask=sm)
        return attn[0].detach().cpu(), lag[0].detach().cpu()

    def _interp_curve(u_src, y_src, n=200):
        u = np.asarray(u_src).reshape(-1); y = np.asarray(y_src).reshape(-1)
        if u.size < 2: return np.linspace(0,1,n), np.zeros(n)
        xi = np.linspace(0,1,n); yi = np.interp(xi, u, y); return xi, yi

    heat = {"ASD": np.zeros((grid, grid), float), "TD": np.zeros((grid, grid), float)}
    cnt  = {"ASD": 0, "TD": 0}
    lag_curv, path_curv, lag_means = {"ASD": [], "TD": []}, {"ASD": [], "TD": []}, {"ASD": [], "TD": []}

    for idx in test_indices:
        s = dataset[idx]
        cls = "ASD" if int(s["label"]) == asd_label else "TD"
        attn, lag = _encode_and_align(s)
        Ts, Te = attn.shape

        a = attn.clamp(min=0); ssum = float(a.sum().item())
        if ssum <= 1e-12: continue
        a = (a / ssum).unsqueeze(0).unsqueeze(0)
        aG = F.interpolate(a, size=(grid, grid), mode="bilinear", align_corners=True)[0,0].numpy()
        heat[cls] += aG; cnt[cls] += 1

        attn_np = attn.numpy()
        denom   = attn_np.sum(axis=1, keepdims=True) + 1e-8
        exp_t   = (attn_np * np.linspace(0,1,Te)[None,:]).sum(axis=1) / denom.squeeze(1)
        u = np.linspace(0,1,Ts)
        _, pc = _interp_curve(u, exp_t, n=curve_len)
        path_curv[cls].append(pc)

        lag_np = lag.numpy().astype(float)
        if lag_window and lag_window > 0: lag_np = lag_np / float(lag_window)
        _, lc = _interp_curve(u, lag_np, n=curve_len)
        lag_curv[cls].append(lc); lag_means[cls].append(float(lag_np.mean()))

    for k in ("ASD","TD"):
        if cnt[k] > 0: heat[k] /= cnt[k]

    # heatmaps
    for cls in ("ASD","TD"):
        if cnt[cls] == 0: 
            # placeholder
            plt.figure(figsize=(6.4,5.4))
            plt.imshow(np.zeros((grid,grid)), origin="lower", cmap="viridis", vmin=0.0, vmax=1.0, extent=(0,1,0,1), aspect="auto")
            u = np.linspace(0,1,200); plt.plot(u,u,"--",color="white",alpha=0.85,linewidth=1.2,label="diag (u)")
            plt.title(f"class-averaged alignment heatmap [{cls}]")
            plt.xlabel("EXP (normalized time)"); plt.ylabel("SUB (normalized time)")
            plt.legend(loc="upper left", framealpha=0.85); plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"class_avg_alignment_heatmap_{cls}.png"), dpi=180); plt.close()
            continue
        plt.figure(figsize=(6.4,5.4))
        vmax = float(heat[cls].max() + 1e-12)
        plt.imshow(heat[cls], origin="lower", cmap="viridis", vmin=0.0, vmax=vmax, extent=(0,1,0,1), aspect="auto")
        plt.colorbar(label="avg attention density")
        u = np.linspace(0,1,200); plt.plot(u,u,"--",color="white",alpha=0.85,linewidth=1.2,label="diag (u)")
        plt.title(f"class-averaged alignment heatmap [{cls}]")
        plt.xlabel("EXP (normalized time)"); plt.ylabel("SUB (normalized time)")
        plt.legend(loc="upper left", framealpha=0.85); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"class_avg_alignment_heatmap_{cls}.png"), dpi=180); plt.close()

    # curve plots
    def _plot_curves(curves, title, ylabel, fname):
        plt.figure(figsize=(9.2,3.8)); x = np.linspace(0,1,curve_len)
        for cls, color in (("ASD","#D32F2F"), ("TD","#1976D2")):
            if len(curves[cls]) == 0: continue
            arr = np.stack(curves[cls], axis=0); mean = arr.mean(0); std = arr.std(0)
            plt.fill_between(x, mean-std, mean+std, color=color, alpha=0.18, linewidth=0)
            plt.plot(x, mean, color=color, linewidth=2.0, label=f"{cls}")
        plt.xlabel("SUB (normalized time)"); plt.ylabel(ylabel); plt.title(title)
        plt.legend(loc="best"); plt.grid(alpha=0.25); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, fname), dpi=180); plt.close()

    _plot_curves(lag_curv,  "Class-averaged lag curve (mean ± std)", "lag (normalized)" if lag_window else "lag (frames)", "class_avg_lag_curve.png")
    _plot_curves(path_curv, "Class-averaged expected EXP time (mean ± std)", "E[t|attn] (normalized)", "class_avg_path_curve.png")

    stats = {
        "counts": cnt,
        "lag_mean": {k: (float(np.mean(lag_means[k])) if len(lag_means[k]) else None) for k in ("ASD","TD")},
        "lag_std":  {k: (float(np.std(lag_means[k]))  if len(lag_means[k]) else None) for k in ("ASD","TD")},
        "notes": "Averaged on normalized grids; curves are resampled to equal length."
    }
    with open(os.path.join(save_dir, "class_avg_alignment_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[classwise_viz] Done. ASD, TD  ->  {save_dir}")
