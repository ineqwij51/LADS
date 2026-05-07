#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
from typing import Dict, Tuple, Optional, List
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
from datetime import datetime
import gc
import os

warnings.filterwarnings('ignore')

# ===================== 数据集与工具函数 =====================

def collate_fn_with_padding(batch):
    """自定义的批处理函数，处理变长序列"""
    exp_seqs = []
    sub_seqs = []
    labels = []
    sub_ids = []
    exp_lengths = []
    sub_lengths = []
    for sample in batch:
        exp_seqs.append(sample['exp'])
        sub_seqs.append(sample['sub'])
        labels.append(sample['label'])
        sub_ids.append(sample['sub_id'])
        exp_lengths.append(len(sample['exp']))
        sub_lengths.append(len(sample['sub']))
    exp_padded = pad_sequence(exp_seqs, batch_first=True, padding_value=0)
    sub_padded = pad_sequence(sub_seqs, batch_first=True, padding_value=0)
    max_exp_len = exp_padded.size(1)
    max_sub_len = sub_padded.size(1)
    exp_mask = torch.zeros(len(batch), max_exp_len, dtype=torch.bool)
    sub_mask = torch.zeros(len(batch), max_sub_len, dtype=torch.bool)
    for i, (exp_len, sub_len) in enumerate(zip(exp_lengths, sub_lengths)):
        exp_mask[i, :exp_len] = True
        sub_mask[i, :sub_len] = True
    labels = torch.stack(labels)
    return {
        'exp': exp_padded,
        'sub': sub_padded,
        'exp_mask': exp_mask,
        'sub_mask': sub_mask,
        'exp_lengths': torch.tensor(exp_lengths),
        'sub_lengths': torch.tensor(sub_lengths),
        'label': labels,
        'sub_id': sub_ids
    }

class AutismDataset(Dataset):
    """自闭症数据集加载器（支持变长序列）"""
    def __init__(self, data_path, feature_type='skeleton', transform=None):
        self.data_path = Path(data_path)
        self.feature_type = feature_type
        self.transform = transform
        if self.data_path.suffix == '.pkl':
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = torch.load(self.data_path)
        if 'exp_features' in self.data:
            self._convert_pytorch_format()
    def _convert_pytorch_format(self):
        converted_data = []
        for i in range(len(self.data['labels'])):
            sample = {
                'exp': {'combined': self.data['exp_features'][i]},
                'sub': {'combined': self.data['sub_features'][i]},
                'metadata': self.data['metadata'][i] if 'metadata' in self.data else {
                    'label': 'ASD' if self.data['labels'][i] == 1 else 'TD',
                    'sub_id': str(i)
                }
            }
            converted_data.append(sample)
        self.data = converted_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.feature_type in sample['exp'] and self.feature_type in sample['sub']:
            exp_features = sample['exp'][self.feature_type]
            exp_features = exp_features.reshape(exp_features.shape[0], -1)
            sub_features = sample['sub'][self.feature_type]
            sub_features = sub_features.reshape(sub_features.shape[0], -1)
        elif 'combined' in sample['exp']:
            exp_features = sample['exp']['combined']
            sub_features = sample['sub']['combined']
        else:
            feature_keys = list(sample['exp'].keys())
            if feature_keys:
                exp_features = sample['exp'][feature_keys[0]]
                sub_features = sample['sub'][feature_keys[0]]
            else:
                raise ValueError(f"No features found for sample {idx}")
        exp_tensor = torch.FloatTensor(exp_features)
        sub_tensor = torch.FloatTensor(sub_features)
        label = 1 if sample['metadata']['label'] == 'ASD' else 0
        label_tensor = torch.LongTensor([label]).squeeze()
        if self.transform:
            exp_tensor = self.transform(exp_tensor)
            sub_tensor = self.transform(sub_tensor)
        return {
            'exp': exp_tensor,
            'sub': sub_tensor,
            'label': label_tensor,
            'sub_id': sample['metadata']['sub_id']
        }

class SubjectIndependentKFold:
    """受试者独立的K折交叉验证（支持三集划分）"""
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, dataset):
        """生成训练/验证/测试索引（被试独立）"""
        # 收集所有受试者ID和对应的样本索引
        subject_indices = {}
        for idx in range(len(dataset)):
            sub_id = dataset[idx]['sub_id']
            if sub_id not in subject_indices:
                subject_indices[sub_id] = []
            subject_indices[sub_id].append(idx)
        # 获取所有被试ID列表
        subject_ids = list(subject_indices.keys())
        print(f"总被试数: {len(subject_ids)}")
        print(f"被试ID列表: {sorted(subject_ids)}")
        # 设置随机种子以确保可重复性
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(subject_ids)
        # 计算每个fold应该包含的被试数量
        n_subjects = len(subject_ids)
        subjects_per_fold = n_subjects // self.n_splits
        remainder = n_subjects % self.n_splits
        print(f"每个fold的被试数: {subjects_per_fold}, 余数: {remainder}")
        # 生成每个fold的被试划分
        fold_subjects = []
        start_idx = 0
        for i in range(self.n_splits):
            # 前remainder个fold多分配一个被试
            fold_size = subjects_per_fold + (1 if i < remainder else 0)
            end_idx = start_idx + fold_size
            fold_subjects.append(subject_ids[start_idx:end_idx])
            start_idx = end_idx
        print(f"各fold的被试划分:")
        for i, subjects in enumerate(fold_subjects):
            print(f"  Fold {i+1}: {sorted(subjects)} (共{len(subjects)}个被试)")
        # 生成每个fold的训练/验证/测试索引
        for fold_idx in range(self.n_splits):
            # 测试集：当前fold的被试
            test_subjects = fold_subjects[fold_idx]
            test_indices = []
            for sub_id in test_subjects:
                test_indices.extend(subject_indices[sub_id])
            # 训练+验证集：其他所有fold的被试
            trainval_subjects = []
            for i in range(self.n_splits):
                if i != fold_idx:
                    trainval_subjects.extend(fold_subjects[i])
            # 将训练+验证集按被试独立原则划分为训练集和验证集
            np.random.seed(self.random_state + fold_idx)
            np.random.shuffle(trainval_subjects)
            n_trainval_subjects = len(trainval_subjects)
            n_train_subjects = int(0.8 * n_trainval_subjects)
            train_subjects = trainval_subjects[:n_train_subjects]
            val_subjects = trainval_subjects[n_train_subjects:]
            # 根据被试ID获取对应的样本索引
            train_indices = []
            for sub_id in train_subjects:
                train_indices.extend(subject_indices[sub_id])
            val_indices = []
            for sub_id in val_subjects:
                val_indices.extend(subject_indices[sub_id])
            print(f"\nFold {fold_idx+1} 三集被试独立划分:")
            print(f"  训练被试: {sorted(train_subjects)} (共{len(train_subjects)}个被试)")
            print(f"  验证被试: {sorted(val_subjects)} (共{len(val_subjects)}个被试)")
            print(f"  测试被试: {sorted(test_subjects)} (共{len(test_subjects)}个被试)")
            print(f"  训练样本: {len(train_indices)}个")
            print(f"  验证样本: {len(val_indices)}个")
            print(f"  测试样本: {len(test_indices)}个")
            yield train_indices, val_indices, test_indices

# ===================== VGG-style 1D CNN =====================
class VGGStyle1DCNN(nn.Module):
    """VGG-style 1D CNN 适用于 (B, T, F) 时序特征，只使用 SUB 数据"""
    def __init__(self, in_channels: int, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x, mask=None):
        # x shape: [B, T, F]  -> 转为 [B, F, T]
        x = x.transpose(1, 2)
        x = self.features(x)
        return self.classifier(x)

# ===================== ResNet1D (代表性实现) =====================
class BasicBlock1d(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, kernel_size=7, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet1d(nn.Module):
    def __init__(self, block, layers, in_channels, num_classes=2, kernel_size=7, dropout=0.0):
        super().__init__()
        self.inplanes = 64
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._init_weights()
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size=self.kernel_size, dropout=self.dropout))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=self.kernel_size, dropout=self.dropout))
        return nn.Sequential(*layers)
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet34_1d(**kwargs):
    return ResNet1d(BasicBlock1d, [3, 4, 6, 3], **kwargs)

# ===================== DIM模型实现 (直接使用开源代码) =====================
import math
import numpy as np
from einops import rearrange

# 从开源代码直接导入的辅助函数
def get_activation(activation):
    """获取激活函数"""
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "swish":
        return lambda x: x * torch.sigmoid(x)
    else:
        raise ValueError(f"Unknown activation: {activation}")

def get_shape_list(tensor):
    """获取张量的形状列表"""
    shape = list(tensor.shape)
    return shape

# 从开源代码直接导入的VectorQuantizer
class VectorQuantizer(nn.Module):
    """
    从开源代码直接导入的VectorQuantizer
    见 https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    """
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        z_flattened = z.view(-1, self.e_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z) 
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # 直接返回[B, T, embedding_dim]格式
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

# 从开源代码直接导入的基础模型组件
class Norm(nn.Module):
    """ Norm Layer """
    def __init__(self, fn, size):
        super().__init__()
        self.norm = nn.LayerNorm(size, eps=1e-5)
        self.fn = fn

    def forward(self, x_data):
        if type(x_data) is dict:
            x_norm = self.fn({'x_a':x_data['x_a'], 'x_b':self.norm(x_data['x_b'])})
            return x_norm
        else:
            x, mask_info = x_data
            x_norm, _ = self.fn((self.norm(x), mask_info))
            return (x_norm, mask_info)

class Residual(nn.Module):
    """ Residual Layer """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x_data):
        if type(x_data) is dict:
            x_resid = self.fn(x_data)['x_b']
            return {'x_a':x_data['x_a'], 'x_b':x_resid+x_data['x_b']}
        else:
            x, mask_info = x_data
            x_resid, _ = self.fn(x_data)
            return (x_resid + x, mask_info)

class MLP(nn.Module):
    """ MLP Layer """
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.activation = get_activation("gelu")
        self.l2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_data):
        if type(x_data) is dict:
            out = self.l2(self.activation(self.l1(x_data['x_b'])))
            return {'x_a':x_data['x_a'], 'x_b':out}
        else:
            x, mask_info = x_data
            out = self.l2(self.activation(self.l1(x)))
            return (out, mask_info)

class CrossModalAttention(nn.Module):
    """ Cross Modal Attention Layer """
    def __init__(self, in_dim, dim, heads=8, in_dim2=None):
        super().__init__()
        self.heads = heads
        self.scale = dim**-0.5

        if in_dim2 is not None:
            self.to_kv = nn.Linear(in_dim2, in_dim2 * 2, bias=False)
        else:
            self.to_kv = nn.Linear(in_dim, dim * 2, bias=False)
        self.to_q = nn.Linear(in_dim, dim, bias=False)
        if in_dim2 is not None:
            dim2 = int((in_dim + in_dim2*2) / 3)
        else:
            dim2 = dim
        self.to_out = nn.Linear(dim2, dim)

        self.rearrange_qkv = lambda x: rearrange(x, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
        self.rearrange_out = lambda x: rearrange(x, "b h n d -> b n (h d)")

    def forward(self, x_data):
        x_a = x_data['x_a']
        x_b = x_data['x_b']

        kv = self.to_kv(x_b)
        q = self.to_q(x_a)

        qkv = torch.cat((q, kv), dim=-1)
        qkv = self.rearrange_qkv(qkv)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = F.softmax(dots, dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = self.rearrange_out(out)
        out = self.to_out(out)
        return {'x_a':x_a, 'x_b':out}

class Attention(nn.Module):
    """ Attention Layer """
    def __init__(self, in_dim, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim**-0.5

        self.to_qkv = nn.Linear(in_dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

        self.rearrange_qkv = lambda x: rearrange(x, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
        self.rearrange_out = lambda x: rearrange(x, "b h n d -> b n (h d)")

    def forward(self, x_data):
        x, mask_info = x_data
        max_mask = mask_info['max_mask']
        mask = mask_info['mask']
        
        qkv = self.to_qkv(x)
        qkv = self.rearrange_qkv(qkv)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        if max_mask is not None:
            dots[:,:,:max_mask,:max_mask] = \
                dots[:,:,:max_mask,:max_mask].masked_fill(mask == 0., float('-inf'))

        attn = F.softmax(dots, dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = self.rearrange_out(out)
        out = self.to_out(out)
        return (out, mask_info)

class Transformer(nn.Module):
    """ Transformer class """
    def __init__(self,
                 in_size=50,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 cross_modal=False,
                 in_dim2=None):
        super().__init__()
        blocks = []
        attn = False

        self.cross_modal = cross_modal
        if cross_modal:
            for i in range(num_hidden_layers):
                blocks.extend([
                    Residual(Norm(CrossModalAttention(in_size, hidden_size,
                                                      heads=num_attention_heads,
                                                      in_dim2=in_dim2), hidden_size)),
                    Residual(Norm(MLP(hidden_size, hidden_size, intermediate_size),
                                      hidden_size))
                ])
        else:
            for i in range(num_hidden_layers):
                blocks.extend([
                    Residual(Norm(Attention(in_size, hidden_size,
                                            heads=num_attention_heads), hidden_size)),
                    Residual(Norm(MLP(hidden_size, hidden_size, intermediate_size),
                                      hidden_size))
                ])
        self.net = torch.nn.Sequential(*blocks)

    def forward(self, x_data):
        if self.cross_modal:
            assert type(x_data) is dict
            x_data = self.net(x_data)
            x = x_data['x_b']
        else:
            x, mask_info = x_data
            x, _ = self.net((x, mask_info))
        return x

class LinearEmbedding(nn.Module):
    """ Linear Layer """
    def __init__(self, size, dim):
        super().__init__()
        self.net = nn.Linear(size, dim)

    def forward(self, x):
        return self.net(x)

class PositionEmbedding(nn.Module):
    """Postion Embedding Layer"""
    def __init__(self, seq_length, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(seq_length, dim))

    def forward(self, x):
        # x: [B, T, D] -> 添加位置嵌入
        return x + self.pos_embedding[:x.size(1), :]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 从开源代码直接导入的TransformerEncoder
class TransformerEncoder(nn.Module):
    """ 从开源代码直接导入的TransformerEncoder """
    def __init__(self, in_dim, hidden_size=768, num_hidden_layers=6, 
                 num_attention_heads=8, intermediate_size=3072, 
                 quant_factor=1, neg=0.2, INaffine=True, face_quan_num=1, zquant_dim=256):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.quant_factor = quant_factor
        self.neg = neg
        self.INaffine = INaffine
        self.face_quan_num = face_quan_num
        self.zquant_dim = zquant_dim
        
        size = self.in_dim
        dim = self.hidden_size
        self.vertice_mapping = nn.Sequential(nn.Linear(size,dim), nn.LeakyReLU(self.neg, True))
        
        if quant_factor == 0:
            layers = [nn.Sequential(
                        nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                    padding_mode='replicate'),
                        nn.LeakyReLU(self.neg, True),
                        nn.InstanceNorm1d(dim, affine=INaffine)
                        )]
        else:
            layers = [nn.Sequential(
                        nn.Conv1d(dim,dim,5,stride=2,padding=2,
                                    padding_mode='replicate'),
                        nn.LeakyReLU(self.neg, True),
                        nn.InstanceNorm1d(dim, affine=INaffine)
                        )] 

            for _ in range(1, quant_factor):
                layers += [nn.Sequential(
                            nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                        padding_mode='replicate'),
                            nn.LeakyReLU(self.neg, True),
                            nn.InstanceNorm1d(dim, affine=INaffine), 
                            nn.MaxPool1d(2)
                            )] 
        self.squasher = nn.Sequential(*layers)
        self.encoder_transformer = Transformer(
            in_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size)
        self.encoder_pos_embedding = PositionalEncoding(
            self.hidden_size)
        self.encoder_linear_embedding = LinearEmbedding(
            self.hidden_size,
            self.hidden_size)
        
        # turn the channel number back to args.face_quan_num*args.zquant_dim
        self.encoder_linear_embedding_post = LinearEmbedding(
            self.hidden_size,
            self.face_quan_num*self.zquant_dim)

    def forward(self, inputs):
        ## downdample into path-wise length seq before passing into transformer
        dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
        inputs = self.vertice_mapping(inputs)
        inputs = self.squasher(inputs.permute(0,2,1)).permute(0,2,1) # [N L C]

        encoder_features = self.encoder_linear_embedding(inputs)
        encoder_features = self.encoder_pos_embedding(encoder_features)
        encoder_features = self.encoder_transformer((encoder_features, dummy_mask))
        encoder_features = self.encoder_linear_embedding_post(encoder_features)
        return encoder_features

# 从开源代码直接导入的CrossModalLayer
class CrossModalLayer(nn.Module):
    """Cross Modal Layer inspired by FACT [Li 2021]"""
    def __init__(self, in_dim, hidden_size=768, num_hidden_layers=6, 
                 num_attention_heads=8, intermediate_size=3072, 
                 sequence_length=100, out_dim=256):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.sequence_length = sequence_length
        self.out_dim = out_dim
        
        self.transformer_layer = Transformer(
            in_size=hidden_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size)

        self.cross_norm_layer = nn.LayerNorm(in_dim)
        self.cross_output_layer = nn.Linear(
                                        in_dim,
                                        out_dim,
                                        bias=False)

        self.cross_pos_embedding = PositionEmbedding(
                sequence_length, in_dim)

    def forward(self, modal_a_sequences, modal_b_sequences, mask_info):
        """
        Parameters
        ----------
        modal_a_sequences : tensor
            the first modality (e.g. Listener motion embedding)
        modal_b_sequences : tensor
            the second modality (e.g. Speaker motion+audio embedding)
        mask_info: dict
            specifies the binary mask that is applied to the Transformer attention
        """
        _, _, modal_a_width = modal_a_sequences.shape
        merged_sequences = modal_a_sequences
        if modal_b_sequences is not None:
            _, _, modal_b_width = modal_b_sequences.shape
            if modal_a_width != modal_b_width:
                raise ValueError(
                    "The modal_a hidden size (%d) should be the same with the modal_b"
                    "hidden size (%d)" % (modal_a_width, modal_b_width))
            merged_sequences = torch.cat([merged_sequences, modal_b_sequences],
                                          dim=1)

        merged_sequences = self.cross_pos_embedding(merged_sequences)
        merged_sequences = self.transformer_layer((merged_sequences, mask_info))
        merged_sequences = self.cross_norm_layer(merged_sequences)
        logits = self.cross_output_layer(merged_sequences)
        return logits

# 重新设计的DIM模型 - 直接使用开源组件
class DIMTransformer(nn.Module):
    """
    DIM Transformer: 基于开源代码重新设计
    直接集成开源代码的核心模块，确保架构一致性
    适用于ASD/TD分类任务
    """
    def __init__(self, feature_dim, hidden_dim=768, num_heads=8, num_layers=6,
                 num_embeddings=512, embedding_dim=256, num_classes=2, dropout=0.1,
                 quant_factor=1, sequence_length=100):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        # 确保embedding_dim和hidden_dim一致
        self.embedding_dim = hidden_dim  # 强制使用hidden_dim
        self.num_embeddings = num_embeddings
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        
        # 从开源代码直接导入的VectorQuantizer
        self.speaker_quantizer = VectorQuantizer(num_embeddings, self.embedding_dim, beta=0.25)
        self.listener_quantizer = VectorQuantizer(num_embeddings, self.embedding_dim, beta=0.25)
        
        # 从开源代码直接导入的TransformerEncoder
        self.speaker_encoder = TransformerEncoder(
            in_dim=feature_dim,
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_dim * 4,
            quant_factor=quant_factor,
            face_quan_num=1,
            zquant_dim=self.embedding_dim
        )
        
        self.listener_encoder = TransformerEncoder(
            in_dim=feature_dim,
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_dim * 4,
            quant_factor=quant_factor,
            face_quan_num=1,
            zquant_dim=self.embedding_dim
        )
        
        # 使用标准的Transformer进行跨模态融合
        self.cross_modal_transformer = Transformer(
            in_size=self.embedding_dim,  # 使用embedding_dim作为输入维度
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_dim * 4,
            cross_modal=False  # 使用标准Transformer
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, exp_seq, sub_seq, exp_mask=None, sub_mask=None):
        """
        exp_seq: [B, T, F] - 说话者序列
        sub_seq: [B, T, F] - 听者序列
        exp_mask: [B, T] - 说话者掩码
        sub_mask: [B, T] - 听者掩码
        """
        B = exp_seq.size(0)
        
        # 使用开源TransformerEncoder编码
        speaker_features = self.speaker_encoder(exp_seq)  # [B, T//2, embedding_dim] (下采样后)
        listener_features = self.listener_encoder(sub_seq)  # [B, T//2, embedding_dim] (下采样后)
        
        # 向量量化
        speaker_quantized, speaker_vq_loss, speaker_info = self.speaker_quantizer(speaker_features)  # [B, T//2, embedding_dim]
        listener_quantized, listener_vq_loss, listener_info = self.listener_quantizer(listener_features)  # [B, T//2, embedding_dim]
        
        # 拼接后送入Transformer
        combined_features = torch.cat([speaker_quantized, listener_quantized], dim=1)  # [B, T, embedding_dim]
        
        # 使用Transformer处理，输出为[B, T, hidden_dim]
        dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
        cross_modal_features = self.cross_modal_transformer((combined_features, dummy_mask))  # [B, T, hidden_dim]
        
        # 全局平均池化
        if exp_mask is not None and sub_mask is not None:
            # 掩码也需要下采样以匹配特征维度
            # 使用平均池化下采样掩码
            exp_mask_downsampled = self._downsample_mask(exp_mask)  # [B, T//2]
            sub_mask_downsampled = self._downsample_mask(sub_mask)  # [B, T//2]
            combined_mask = torch.cat([exp_mask_downsampled, sub_mask_downsampled], dim=1)  # [B, T]
            cross_modal_features = self._masked_mean(cross_modal_features, combined_mask)
        else:
            cross_modal_features = cross_modal_features.mean(dim=1)  # [B, hidden_dim]
        
        # 分类
        logits = self.classifier(cross_modal_features)
        
        # 总损失
        vq_loss = speaker_vq_loss + listener_vq_loss
        
        return logits, vq_loss
    
    def _downsample_mask(self, mask):
        """下采样掩码以匹配特征维度"""
        # 使用平均池化下采样掩码
        # mask: [B, T] -> [B, T//2]
        B, T = mask.shape
        
        # 处理奇数长度的情况
        if T % 2 == 1:
            # 如果是奇数，先补一个False到末尾
            mask = torch.cat([mask, torch.zeros(B, 1, dtype=torch.bool, device=mask.device)], dim=1)
            T = T + 1
        
        # 现在T一定是偶数
        mask_reshaped = mask.view(B, T//2, 2)  # [B, T//2, 2]
        # 如果两个时间步中任何一个为True，则下采样后的掩码为True
        downsampled_mask = torch.any(mask_reshaped, dim=2)  # [B, T//2]
        return downsampled_mask
    
    def _masked_mean(self, x, mask):
        """掩码加权平均"""
        mask_expanded = mask.unsqueeze(-1).float()
        masked_x = x * mask_expanded
        summed = torch.sum(masked_x, dim=1)
        lengths = torch.sum(mask, dim=1, keepdim=True).float()
        lengths = torch.clamp(lengths, min=1.0)
        return summed / lengths

# ===================== BlockGCN实现 (直接使用开源代码) =====================
import math
import numpy as np
from einops import rearrange

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MultiScale_TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 dilations=[1,2,3,4], residual=False, residual_kernel_size=1):
        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'
        
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
            
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(branch_channels, branch_channels, kernel_size=ks, stride=stride, dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        self.apply(weights_init)

    def forward(self, x):
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
        out = torch.cat(branch_outs, dim=1)
        out += res
        return out

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True, alpha=False):
        super(unit_gcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_heads = 8 if in_channels > 8 else 1
        self.fc1 = nn.Parameter(torch.stack([torch.stack([torch.eye(A.shape[-1]) for _ in range(self.num_heads)], dim=0) for _ in range(3)], dim=0), requires_grad=True)
        self.fc2 = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 1, groups=self.num_heads) for _ in range(3)])

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        
        # k-hop - 简化版本，避免复杂的矩阵运算
        self.hops = torch.zeros(A.shape[-1], A.shape[-1], dtype=torch.long)
        # 直接使用邻接矩阵作为1-hop连接
        self.hops = (A > 0).long()
        self.rpe = nn.Parameter(torch.zeros((3, self.num_heads, self.hops.max() + 1,)))
        self.in_channels = in_channels
        self.hidden_channels = in_channels if in_channels > 3 else 64

        if alpha:
            self.alpha = nn.Parameter(torch.ones(1, self.num_heads, 1, 1, 1))
        else:
            self.alpha = 1

    def L2_norm(self, weight):
        weight_norm = torch.norm(weight, 2, dim=-2, keepdim=True) + 1e-4
        return weight_norm

    def forward(self, x):
        N, C, T, V = x.size()
        y = None
        pos_emb = self.rpe[:, :, self.hops]
        
        for i in range(3):
            weight_norm = self.L2_norm(self.fc1[i])
            w1 = self.fc1[i]
            w1 = w1/weight_norm
            w1 = w1 + pos_emb[i]/self.L2_norm(pos_emb[i])
            x_in = x.view(N, self.num_heads, C//self.num_heads, T, V)
            z = torch.einsum("nhctv, hvw->nhctw", (x_in, w1)).contiguous().view(N, -1, T, V)
            z = self.fc2[i](z)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2], num_point=25, num_heads=16, alpha=False):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive, alpha=alpha)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations, residual=False)
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y

class BlockGCN(nn.Module):
    """
    BlockGCN: 直接使用开源代码的核心模块
    基于CVPR 2024官方实现 https://github.com/ZhouYuxuanYX/BlockGCN
    适配33关节骨骼数据，适用于ASD/TD分类
    """
    def __init__(self, num_joints=33, num_features=4, num_classes=2, 
                 hidden_dim=128, dropout=0.1, adaptive=True, alpha=False):
        super().__init__()
        self.num_joints = num_joints
        self.num_features = num_features
        self.num_classes = num_classes
        
        # 构建邻接矩阵 - 基于33关节人体骨骼
        self.register_buffer('A', self._build_skeleton_graph())
        
        # 数据归一化层
        self.data_bn = nn.BatchNorm1d(128 * num_joints)
        
        # 关节嵌入层
        self.to_joint_embedding = nn.Linear(num_features, 128)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_joints, 128))
        
        # TCN-GCN单元层 - 直接使用开源实现
        self.l1 = TCN_GCN_unit(128, 128, self.A, adaptive=adaptive, alpha=alpha)
        self.l2 = TCN_GCN_unit(128, 128, self.A, adaptive=adaptive, alpha=alpha)
        self.l3 = TCN_GCN_unit(128, 128, self.A, adaptive=adaptive, alpha=alpha)
        self.l4 = TCN_GCN_unit(128, 128, self.A, adaptive=adaptive, alpha=alpha)
        self.l5 = TCN_GCN_unit(128, 256, self.A, stride=2, adaptive=adaptive, alpha=alpha)
        self.l6 = TCN_GCN_unit(256, 256, self.A, adaptive=adaptive, alpha=alpha)
        self.l7 = TCN_GCN_unit(256, 256, self.A, adaptive=adaptive, alpha=alpha)
        self.l8 = TCN_GCN_unit(256, 256, self.A, stride=2, adaptive=adaptive, alpha=alpha)
        self.l9 = TCN_GCN_unit(256, 256, self.A, adaptive=adaptive, alpha=alpha)
        self.l10 = TCN_GCN_unit(256, 256, self.A, adaptive=adaptive, alpha=alpha)
        
        # 分类头
        self.fc = nn.Linear(256, num_classes)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_classes))
        bn_init(self.data_bn, 1)
        
        # Dropout
        if dropout:
            self.drop_out = nn.Dropout(dropout)
        else:
            self.drop_out = lambda x: x
    
    def _build_skeleton_graph(self):
        """构建33关节人体骨骼图的邻接矩阵"""
        # 基于33个关键点的人体骨骼连接
        adjacency = torch.zeros(self.num_joints, self.num_joints)
        
        # 定义骨骼连接 - 扩展自NTU RGB+D的25关节到33关节
        connections = [
            # 头部连接 (0-4)
            (0, 1), (1, 2), (2, 3), (3, 4),
            # 躯干连接 (5-9)
            (5, 6), (6, 7), (7, 8), (8, 9),
            # 左臂连接 (10-14)
            (10, 11), (11, 12), (12, 13), (13, 14),
            # 右臂连接 (15-19)
            (15, 16), (16, 17), (17, 18), (18, 19),
            # 左腿连接 (20-24)
            (20, 21), (21, 22), (22, 23), (23, 24),
            # 右腿连接 (25-29)
            (25, 26), (26, 27), (27, 28), (28, 29),
            # 手部连接 (30-32)
            (30, 31), (31, 32),
        ]
        
        # 设置连接
        for i, j in connections:
            if i < self.num_joints and j < self.num_joints:
                adjacency[i, j] = 1
                adjacency[j, i] = 1  # 无向图
        
        # 添加自连接
        adjacency += torch.eye(self.num_joints)
        
        # 归一化
        degree = adjacency.sum(dim=1, keepdim=True)
        degree = torch.clamp(degree, min=1.0)
        adjacency = adjacency / degree
        
        return adjacency
    
    def forward(self, x, mask=None):
        """
        x: [B, T, num_joints, num_features] - 骨骼点数据
        mask: [B, T] - 时序掩码 (暂时不使用)
        """
        B, T, V, C = x.shape  # 调整为开源代码的格式
        
        # 关节嵌入
        x = x.view(B * T, V, C)  # [B*T, V, C]
        x = self.to_joint_embedding(x)  # [B*T, V, 128]
        x += self.pos_embedding[:, :self.num_joints]  # 添加位置嵌入
        
        # 重塑为开源代码格式
        x = x.view(B, T, V, 128).permute(0, 3, 1, 2)  # [B, 128, T, V]
        
        # 数据归一化
        x = x.reshape(B, 128 * V, T)  # [B, 128*V, T]
        x = self.data_bn(x)  # 归一化
        x = x.reshape(B, V, 128, T).permute(0, 2, 3, 1)  # [B, 128, T, V]
        
        # TCN-GCN单元处理 - 直接使用开源实现
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        
        # 全局平均池化
        x = x.mean(3).mean(2)  # [B, 256]
        x = self.drop_out(x)
        
        # 分类
        logits = self.fc(x)
        return logits


# ===================== LLM实现 (基于DeepSeek大模型) =====================
class LLMClassifier(nn.Module):
    """
    LLM: 基于DeepSeek大语言模型的序列编码和分类
    使用DeepSeek LLM对时序特征序列进行编码，训练ASD/TD分类器
    适用于Skeleton/Flow/Heatmap特征序列
    """
    def __init__(self, feature_dim, seq_len, num_classes=2, 
                 model_name="deepseek-ai/deepseek-llm-7b-base", hidden_dim=4096, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # DeepSeek LLM编码器
        self.llm_encoder = self._load_deepseek_encoder(model_name)
        
        # 特征投影层 - 将时序特征投影到DeepSeek的输入维度
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim)  # DeepSeek的输入维度
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _load_deepseek_encoder(self, model_name):
        """加载DeepSeek LLM编码器"""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            class DeepSeekEncoder(nn.Module):
                def __init__(self, model_name):
                    super().__init__()
                    # 使用DeepSeek LLM模型
                    self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    
                    # 冻结模型参数（用于特征提取）
                    for param in self.model.parameters():
                        param.requires_grad = False
                    
                def forward(self, x, attention_mask=None):
                    # x: [B, T, hidden_dim] - 已经是正确的维度
                    # 使用DeepSeek LLM编码
                    outputs = self.model(inputs_embeds=x, attention_mask=attention_mask)
                    return outputs.last_hidden_state  # [B, T, hidden_dim]
            
            return DeepSeekEncoder(model_name)
            
        except (ImportError, OSError) as e:
            print(f"Error: Could not load DeepSeek model ({e})")
            print("Please ensure:")
            print("1. Internet connection is available")
            print("2. DeepSeek model is properly installed")
            print("3. Transformers library is installed: pip install transformers")
            raise RuntimeError(f"DeepSeek model loading failed: {e}")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, mask=None):
        """
        x: [B, T, F] - 批次、时间、特征
        mask: [B, T] - 时序掩码
        """
        B, T, F = x.shape
        
        # 处理变长序列
        if T < self.seq_len:
            padding = torch.zeros(B, self.seq_len - T, F, device=x.device)
            x = torch.cat([x, padding], dim=1)
            if mask is not None:
                padding_mask = torch.zeros(B, self.seq_len - T, dtype=torch.bool, device=x.device)
                mask = torch.cat([mask, padding_mask], dim=1)
        elif T > self.seq_len:
            x = x[:, :self.seq_len, :]
            if mask is not None:
                mask = mask[:, :self.seq_len]
        
        # 特征投影到DeepSeek输入维度
        x = self.feature_projection(x)  # [B, T, hidden_dim]
        
        # 使用DeepSeek编码器进行特征提取
        if mask is not None:
            attention_mask = mask.long()
        else:
            attention_mask = torch.ones(B, x.size(1), dtype=torch.long, device=x.device)
        
        x = self.llm_encoder(x, attention_mask=attention_mask)
        
        # 全局平均池化
        x = x.transpose(1, 2)  # [B, hidden_dim, T]
        x = self.global_pool(x)  # [B, hidden_dim, 1]
        x = x.squeeze(-1)  # [B, hidden_dim]
        
        # 分类
        logits = self.classifier(x)
        return logits

# ===================== get_model函数主干 =====================
def get_model(model_name, feature_dim, args):
    """根据模型名称创建相应的模型"""
    if model_name == 'dim':
        return DIMTransformer(
            feature_dim=feature_dim,
            hidden_dim=768,
            num_heads=8,
            num_layers=6,
            num_embeddings=512,
            embedding_dim=256,
            num_classes=2,
            dropout=0.1
        )
    elif model_name == 'resnet':
        return resnet34_1d(in_channels=feature_dim, num_classes=2, kernel_size=7, dropout=0.1)
    elif model_name == 'vgg':
        return VGGStyle1DCNN(in_channels=feature_dim, num_classes=2, dropout=0.5)
    elif model_name == 'blockgcn':
        # BlockGCN - CVPR2024 SOTA动作识别方法
        # 适用于骨骼点数据 (T, 33, 4) 形状
        return BlockGCN(
            num_joints=33,
            num_features=4,
            num_classes=2,
            hidden_dim=128,   # 建议用128，和开源一致
            dropout=0.1
        )
    elif model_name == 'llm':
        # LLM - 基于DeepSeek大语言模型的序列编码和分类
        # 使用DeepSeek LLM对时序特征序列进行编码
        return LLMClassifier(
            feature_dim=feature_dim,
            seq_len=args['seq_len'],
            num_classes=2,
            model_name=args.get('llm_model_name', 'deepseek-ai/deepseek-llm-7b-base'),
            hidden_dim=4096,
            dropout=0.1
        )
    elif model_name == 'patchtst':
        try:
            from tsai.models.PatchTST import PatchTST
            # 创建一个包装类来处理PatchTST的输出
            class PatchTSTWrapper(nn.Module):
                def __init__(self, feature_dim, seq_len, num_classes=2):
                    super().__init__()
                    # 计算合适的patch参数，确保patch数量合理
                    patch_len = min(16, seq_len // 4)  # 确保patch_len不超过序列长度的1/4
                    stride = max(1, patch_len // 2)    # stride为patch_len的一半
                    
                    # 确保序列长度能被patch_len整除，或者调整序列长度
                    if seq_len % patch_len != 0:
                        # 调整序列长度到最近的能被patch_len整除的数
                        adjusted_seq_len = ((seq_len // patch_len) + 1) * patch_len
                        print(f"调整序列长度从 {seq_len} 到 {adjusted_seq_len} 以适应PatchTST")
                        seq_len = adjusted_seq_len
                    
                    self.patchtst = PatchTST(
                        c_in=feature_dim,
                        c_out=feature_dim,  # 输出特征维度
                        seq_len=seq_len,
                        patch_len=patch_len,
                        stride=stride
                    )
                    # 添加分类头
                    self.classifier = nn.Sequential(
                        nn.AdaptiveAvgPool1d(1),  # 全局平均池化
                        nn.Flatten(),
                        nn.Linear(feature_dim, num_classes)
                    )
                    self.seq_len = seq_len
                
                def forward(self, x):
                    # x: [B, C, T]
                    B, C, T = x.shape
                    
                    # 如果输入序列长度与期望的不匹配，进行调整
                    if T != self.seq_len:
                        if T < self.seq_len:
                            # 如果序列太短，进行padding
                            padding = torch.zeros(B, C, self.seq_len - T, device=x.device)
                            x = torch.cat([x, padding], dim=2)
                        else:
                            # 如果序列太长，进行截断
                            x = x[:, :, :self.seq_len]
                    
                    features = self.patchtst(x)  # [B, C, T]
                    logits = self.classifier(features)  # [B, num_classes]
                    return logits
            
            return PatchTSTWrapper(feature_dim, args['seq_len'], 2)
        except ImportError:
            raise ImportError('请安装tsai库以使用PatchTST: pip install tsai')
    else:
        raise ValueError(f'Unknown model: {model_name}')

# ===================== 训练与评估函数 =====================
def train_epoch(model, train_loader, optimizer, device, model_name, lambda_aux=0.1):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    progress_bar = tqdm(train_loader, desc='Training')
    for batch in progress_bar:
        exp_seq = batch['exp'].to(device)
        sub_seq = batch['sub'].to(device)
        exp_mask = batch['exp_mask'].to(device)
        sub_mask = batch['sub_mask'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        
        # 根据模型类型处理输入
        if model_name == 'dim':
            # DIM模型用双输入
            logits, aux_loss = model(exp_seq, sub_seq, exp_mask, sub_mask)
            cls_loss = F.cross_entropy(logits, labels)
            loss = cls_loss + lambda_aux * aux_loss
        elif model_name == 'blockgcn':
            # BlockGCN需要骨骼点数据格式 [B, T, J, F]
            # 将 [B, T, F] 重塑为 [B, T, 33, 4]
            B, T, feat_dim = sub_seq.shape
            if feat_dim == 132:  # 33 * 4 = 132
                sub_seq_reshaped = sub_seq.view(B, T, 33, 4)
            else:
                # 如果特征维度不匹配，进行padding或截断
                target_features = 132
                if feat_dim < target_features:
                    padding = torch.zeros(B, T, target_features - feat_dim, device=sub_seq.device)
                    sub_seq = torch.cat([sub_seq, padding], dim=2)
                else:
                    sub_seq = sub_seq[:, :, :target_features]
                sub_seq_reshaped = sub_seq.view(B, T, 33, 4)
            
            logits = model(sub_seq_reshaped, sub_mask)
            loss = F.cross_entropy(logits, labels)
        elif model_name == 'llm':
            # LLM模型支持mask
            logits = model(sub_seq, sub_mask)
            loss = F.cross_entropy(logits, labels)
        elif model_name == 'patchtst':
            # PatchTST模型不支持mask参数，且需要 [B, C, T] 格式
            # 当前数据格式: [B, T, F] -> 需要转换为 [B, F, T]
            sub_seq_patchtst = sub_seq.transpose(1, 2)  # [B, T, F] -> [B, F, T]
            logits = model(sub_seq_patchtst)
            loss = F.cross_entropy(logits, labels)
        else:
            # 其他模型（VGG, ResNet等）支持mask
            logits = model(sub_seq, sub_mask)
            loss = F.cross_entropy(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * exp_seq.size(0)
        pred = torch.argmax(logits, dim=1)
        total_correct += (pred == labels).sum().item()
        total_samples += exp_seq.size(0)
        progress_bar.set_postfix({'loss': loss.item(), 'acc': total_correct / total_samples})
    return total_loss / total_samples, total_correct / total_samples

def evaluate(model, val_loader, device, model_name, lambda_aux=0.1):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            exp_seq = batch['exp'].to(device)
            sub_seq = batch['sub'].to(device)
            exp_mask = batch['exp_mask'].to(device)
            sub_mask = batch['sub_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 根据模型类型处理输入
            if model_name == 'dim':
                logits, aux_loss = model(exp_seq, sub_seq, exp_mask, sub_mask)
                cls_loss = F.cross_entropy(logits, labels)
                loss = cls_loss + lambda_aux * aux_loss
            elif model_name == 'blockgcn':
                # BlockGCN需要骨骼点数据格式 [B, T, J, F]
                # 将 [B, T, F] 重塑为 [B, T, 33, 4]
                B, T, feat_dim = sub_seq.shape
                if feat_dim == 132:  # 33 * 4 = 132
                    sub_seq_reshaped = sub_seq.view(B, T, 33, 4)
                else:
                    # 如果特征维度不匹配，进行padding或截断
                    target_features = 132
                    if feat_dim < target_features:
                        padding = torch.zeros(B, T, target_features - feat_dim, device=sub_seq.device)
                        sub_seq = torch.cat([sub_seq, padding], dim=2)
                    else:
                        sub_seq = sub_seq[:, :, :target_features]
                    sub_seq_reshaped = sub_seq.view(B, T, 33, 4)
                
                logits = model(sub_seq_reshaped, sub_mask)
                loss = F.cross_entropy(logits, labels)
            elif model_name == 'llm':
                # LLM模型支持mask
                logits = model(sub_seq, sub_mask)
                loss = F.cross_entropy(logits, labels)
            elif model_name == 'patchtst':
                # PatchTST模型不支持mask参数，且需要 [B, C, T] 格式
                # 当前数据格式: [B, T, F] -> 需要转换为 [B, F, T]
                sub_seq_patchtst = sub_seq.transpose(1, 2)  # [B, T, F] -> [B, F, T]
                logits = model(sub_seq_patchtst)
                loss = F.cross_entropy(logits, labels)
            else:
                # 其他模型（VGG, ResNet等）支持mask
                logits = model(sub_seq, sub_mask)
                loss = F.cross_entropy(logits, labels)
            
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)
            total_loss += loss.item() * exp_seq.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    
    # 计算加权平均的precision和f1，但不使用weighted recall（因为与accuracy相同）
    precision, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    # 计算每个类别的指标
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    # 计算更有意义的recall指标
    # 1. ASD类别的recall（敏感度）- 检测ASD的能力
    asd_recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # 也称为敏感度(Sensitivity)
    
    # 2. TD类别的recall（特异度）- 检测TD的能力  
    td_recall = tn / (tn + fp) if (tn + fp) > 0 else 0   # 也称为特异度(Specificity)
    
    # 3. 宏平均recall - 两个类别recall的简单平均
    macro_recall = (asd_recall + td_recall) / 2
    
    # 4. 使用ASD作为positive class的recall（医学诊断中常用）
    asd_as_positive_recall = asd_recall

    metrics = {
        'loss': total_loss / len(val_loader.dataset),
        'accuracy': accuracy,
        'precision': precision,
        'f1': f1,
        # 新的recall指标
        'asd_recall': asd_recall,           # ASD类别的recall（敏感度）
        'td_recall': td_recall,             # TD类别的recall（特异度）
        'macro_recall': macro_recall,       # 宏平均recall
        'asd_as_positive_recall': asd_as_positive_recall,  # 以ASD为正类的recall
        # 每个类别的详细指标
        'precision_td': precision_per_class[0],
        'precision_asd': precision_per_class[1],
        'recall_td': recall_per_class[0],
        'recall_asd': recall_per_class[1],
        'f1_td': f1_per_class[0],
        'f1_asd': f1_per_class[1],
        # 混淆矩阵相关
        'sensitivity': asd_recall,  # 敏感度（ASD recall）
        'specificity': td_recall,   # 特异度（TD recall）
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }
    return metrics

# ===================== 可视化函数 =====================
def visualize_results(train_losses, val_metrics_history, model_name, args, fold_idx):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].plot(train_losses)
    axes[0, 0].set_title(f'{model_name.upper()} - Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    val_accs = [m['accuracy'] for m in val_metrics_history]
    val_f1s = [m['f1'] for m in val_metrics_history]
    axes[0, 1].plot(val_accs, label='Accuracy')
    axes[0, 1].plot(val_f1s, label='F1')
    axes[0, 1].set_title('Validation Metrics')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    last_metrics = val_metrics_history[-1]
    cm = confusion_matrix(last_metrics['labels'], last_metrics['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2])
    axes[0, 2].set_title('Confusion Matrix')
    axes[0, 2].set_xlabel('Predicted')
    axes[0, 2].set_ylabel('Actual')
    val_prec = [m['precision'] for m in val_metrics_history]
    val_asd_recall = [m['asd_recall'] for m in val_metrics_history]
    val_td_recall = [m['td_recall'] for m in val_metrics_history]
    axes[1, 0].plot(val_prec, label='Precision')
    axes[1, 0].plot(val_asd_recall, label='ASD Recall (Sensitivity)')
    axes[1, 0].plot(val_td_recall, label='TD Recall (Specificity)')
    axes[1, 0].set_title('Precision & Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    # 添加ASD和TD的F1分数对比
    val_f1_asd = [m['f1_asd'] for m in val_metrics_history]
    val_f1_td = [m['f1_td'] for m in val_metrics_history]
    axes[1, 1].plot(val_f1_asd, label='ASD F1')
    axes[1, 1].plot(val_f1_td, label='TD F1')
    axes[1, 1].set_title('F1 Score by Class')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 添加宏平均召回率
    val_macro_recall = [m['macro_recall'] for m in val_metrics_history]
    axes[1, 2].plot(val_macro_recall, label='Macro Recall', color='purple')
    axes[1, 2].set_title('Macro Average Recall')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Macro Recall')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    save_path = f"{args['save_dir']}/{model_name}_results_fold{fold_idx}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Results saved to: {save_path}")

# ===================== 主流程 main =====================
def main():
    # ===== 参数设置 =====
    args = {
        'data_path': 'autism_multimodal_dataset_20250726.pkl',
        'feature_type': 'dense_flow',  # skeleton, sparse_flow, dense_flow, heatmap
        'batch_size': 4,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'num_folds': 5,
        'test_fold': -1,  # -1表示全部，>=0表示只运行指定fold
        'device': 'cuda:0',
        'save_dir': 'baseline_results_dense_flow_dim_0815/',
        'seed': 42,
        'model': 'dim',  # vgg, resnet, blockgcn, llm, patchtst, dim
        'lambda_aux': 0.1,
        'llm_model_name': 'deepseek-ai/deepseek-llm-7b-base'  
    }
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    Path(args['save_dir']).mkdir(exist_ok=True)
    results_file = f"{args['save_dir']}/experiment_results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("自闭症儿童模仿行为 Baseline 分类模型实验结果\n")
        f.write("=" * 80 + "\n")
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据集: {args['data_path']}\n")
        f.write(f"特征类型: {args['feature_type']}\n")
        f.write(f"批次大小: {args['batch_size']}\n")
        f.write(f"学习率: {args['learning_rate']}\n")
        f.write(f"训练轮数: {args['num_epochs']}\n")
        f.write(f"K折数: {args['num_folds']}\n")
        f.write(f"模型: {args['model']}\n")
        f.write(f"随机种子: {args['seed']}\n")
        f.write("=" * 80 + "\n\n")
    print(f"加载数据集: {args['data_path']}")
    dataset = AutismDataset(args['data_path'], feature_type=args['feature_type'])
    print(f"数据集大小: {len(dataset)}")
    sample = dataset[0]
    if len(sample['exp'].shape) == 3:
        feature_dim = sample['exp'].shape[1] * sample['exp'].shape[2]
    elif len(sample['exp'].shape) == 4:
        feature_dim = sample['exp'].shape[1] * sample['exp'].shape[2] * sample['exp'].shape[3]
    elif len(sample['exp'].shape) == 2:
        feature_dim = sample['exp'].shape[1]
    else:
        raise ValueError(f"Unsupported feature shape: {sample['exp'].shape}")
    seq_len = sample['sub'].shape[0]
    args['seq_len'] = seq_len
    print(f"特征维度: {feature_dim}")
    print(f"样本序列长度 - EXP: {sample['exp'].shape[0]}, SUB: {sample['sub'].shape[0]}")
    kfold = SubjectIndependentKFold(n_splits=args['num_folds'])
    all_fold_results = []
    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(kfold.split(dataset)):
        if args['test_fold'] >= 0 and fold_idx != args['test_fold']:
            continue
        print(f"\n训练折 {fold_idx + 1}/{args['num_folds']}")
        print(f"训练样本数: {len(train_idx)}, 验证样本数: {len(val_idx)}, 测试样本数: {len(test_idx)}")
        train_sub_ids = set(dataset[idx]['sub_id'] for idx in train_idx)
        val_sub_ids = set(dataset[idx]['sub_id'] for idx in val_idx)
        test_sub_ids = set(dataset[idx]['sub_id'] for idx in test_idx)
        overlap_train_val = train_sub_ids.intersection(val_sub_ids)
        overlap_train_test = train_sub_ids.intersection(test_sub_ids)
        overlap_val_test = val_sub_ids.intersection(test_sub_ids)
        print(f"训练集被试ID: {sorted(list(train_sub_ids))}")
        print(f"验证集被试ID: {sorted(list(val_sub_ids))}")
        print(f"测试集被试ID: {sorted(list(test_sub_ids))}")
        print(f"训练集与验证集重叠被试: {overlap_train_val}")
        print(f"训练集与测试集重叠被试: {overlap_train_test}")
        print(f"验证集与测试集重叠被试: {overlap_val_test}")
        if overlap_train_val or overlap_train_test or overlap_val_test:
            raise ValueError(f"被试独立性验证失败！存在重叠被试: {overlap_train_val}, {overlap_train_test}, {overlap_val_test}")
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(f"Fold {fold_idx + 1} 划分信息\n")
            f.write("-" * 60 + "\n")
            f.write(f"训练样本数: {len(train_idx)}\n")
            f.write(f"验证样本数: {len(val_idx)}\n")
            f.write(f"测试样本数: {len(test_idx)}\n")
            f.write(f"训练被试ID: {sorted(list(train_sub_ids))}\n")
            f.write(f"验证被试ID: {sorted(list(val_sub_ids))}\n")
            f.write(f"测试被试ID: {sorted(list(test_sub_ids))}\n")
            if overlap_train_val:
                f.write(f"[ERROR] 训练/验证存在重叠被试: {overlap_train_val}\n")
            if overlap_train_test:
                f.write(f"[ERROR] 训练/测试存在重叠被试: {overlap_train_test}\n")
            if overlap_val_test:
                f.write(f"[ERROR] 验证/测试存在重叠被试: {overlap_val_test}\n")
            else:
                f.write("被试独立划分验证通过\n")
            f.write("-" * 60 + "\n\n")
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        test_subset = torch.utils.data.Subset(dataset, test_idx)
        train_loader = DataLoader(
            train_subset,
            batch_size=args['batch_size'],
            shuffle=True,
            collate_fn=collate_fn_with_padding,
            num_workers=4
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=args['batch_size'],
            shuffle=False,
            collate_fn=collate_fn_with_padding,
            num_workers=4
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=args['batch_size'],
            shuffle=False,
            collate_fn=collate_fn_with_padding,
            num_workers=4
        )
        model = get_model(args['model'], feature_dim, args).to(args['device'])
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args['learning_rate'],
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args['num_epochs']
        )
        best_val_f1 = 0
        train_losses = []
        val_metrics_history = []
        for epoch in range(args['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{args['num_epochs']}")
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, args['device'], args['model'], args['lambda_aux']
            )
            train_losses.append(train_loss)
            val_metrics = evaluate(model, val_loader, args['device'], args['model'], args['lambda_aux'])
            val_metrics_history.append(val_metrics)
            scheduler.step()
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}, "
                  f"ASD Recall: {val_metrics['asd_recall']:.4f}, F1: {val_metrics['f1']:.4f}")
            print(f"TD Recall: {val_metrics['td_recall']:.4f}, "
                  f"Macro Recall: {val_metrics['macro_recall']:.4f}")
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'args': args
                }, f"{args['save_dir']}/best_model_{args['model']}_fold{fold_idx}.pt")
                print(f"保存最佳模型 (F1: {best_val_f1:.4f})")
        visualize_results(train_losses, val_metrics_history, args['model'], args, fold_idx)
        # 重新加载最佳模型进行最终评估
        print(f"重新加载最佳模型进行最终评估...")
        checkpoint = torch.load(f"{args['save_dir']}/best_model_{args['model']}_fold{fold_idx}.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        # 使用最佳模型在测试集上重新评估
        best_model_metrics = evaluate(model, test_loader, args['device'], args['model'], args['lambda_aux'])
        print(f"最佳模型最终测试集评估结果:")
        print(f"  F1: {best_model_metrics['f1']:.4f}")
        print(f"  Accuracy: {best_model_metrics['accuracy']:.4f}")
        print(f"  ASD Recall: {best_model_metrics['asd_recall']:.4f}")
        # 保存fold的详细结果 - 使用测试集的结果
        final_metrics = best_model_metrics
        fold_result = {
            'fold': fold_idx + 1,
            'train_sub_ids': sorted(list(train_sub_ids)),
            'val_sub_ids': sorted(list(val_sub_ids)),
            'test_sub_ids': sorted(list(test_sub_ids)),
            'best_f1': best_val_f1,
            'final_accuracy': final_metrics['accuracy'],
            'final_precision': final_metrics['precision'],
            'final_asd_recall': final_metrics['asd_recall'],
            'final_td_recall': final_metrics['td_recall'],
            'final_macro_recall': final_metrics['macro_recall'],
            'final_f1': final_metrics['f1'],
            'final_sensitivity': final_metrics['sensitivity'],
            'final_specificity': final_metrics['specificity'],
            'confusion_matrix': confusion_matrix(final_metrics['labels'], final_metrics['predictions']).tolist()
        }
        all_fold_results.append(fold_result)
        # 保存fold结果到文件
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Fold {fold_idx + 1} 最终结果\n")
            f.write(f"{'='*50}\n")
            f.write(f"最佳F1分数: {best_val_f1:.4f}\n")
            f.write(f"最终准确率: {final_metrics['accuracy']:.4f}\n")
            f.write(f"最终精确率: {final_metrics['precision']:.4f}\n")
            f.write(f"ASD召回率(敏感度): {final_metrics['asd_recall']:.4f}\n")
            f.write(f"TD召回率(特异度): {final_metrics['td_recall']:.4f}\n")
            f.write(f"宏平均召回率: {final_metrics['macro_recall']:.4f}\n")
            f.write(f"最终F1分数: {final_metrics['f1']:.4f}\n")
            f.write(f"混淆矩阵:\n")
            cm = confusion_matrix(final_metrics['labels'], final_metrics['predictions'])
            f.write(f"          Predicted\n")
            f.write(f"           TD  ASD\n")
            f.write(f"Actual TD  {cm[0,0]:3d} {cm[0,1]:3d}\n")
            f.write(f"      ASD  {cm[1,0]:3d} {cm[1,1]:3d}\n")
            f.write(f"{'='*50}\n\n")
        print(f"\n折 {fold_idx} 最终结果:")
        print(f"最佳验证F1分数: {best_val_f1:.4f}")
        # 详细评估报告
        from sklearn.metrics import classification_report
        print("\n分类报告:")
        print(classification_report(
            final_metrics['labels'],
            final_metrics['predictions'],
            target_names=['TD', 'ASD']
        ))
        # 保存分类报告到文件
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write("详细分类报告:\n")
            f.write(classification_report(
                final_metrics['labels'],
                final_metrics['predictions'],
                target_names=['TD', 'ASD']
            ))
            f.write("\n")
        # 统计三集被试ID
        print(f"训练被试ID: {sorted(list(train_sub_ids))}")
        print(f"验证被试ID: {sorted(list(val_sub_ids))}")
        print(f"测试被试ID: {sorted(list(test_sub_ids))}")
        # 保存三集被试ID到结果文件
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(f"Fold {fold_idx + 1} 三集被试独立划分\n")
            f.write(f"训练被试ID: {sorted(list(train_sub_ids))}\n")
            f.write(f"验证被试ID: {sorted(list(val_sub_ids))}\n")
            f.write(f"测试被试ID: {sorted(list(test_sub_ids))}\n")
            f.write(f"训练样本数: {len(train_idx)}, 验证样本数: {len(val_idx)}, 测试样本数: {len(test_idx)}\n")
            f.write(f"{'='*50}\n")
        if args['test_fold'] >= 0:
            break
        del model, optimizer, scheduler
        del train_loader, val_loader, test_loader, train_subset, val_subset, test_subset
        gc.collect()
        torch.cuda.empty_cache()
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write("实验总结\n")
        f.write(f"{'='*80}\n")
        f.write(f"实验完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总运行折数: {len(all_fold_results)}\n")
        if args['test_fold'] >= 0:
            f.write(f"运行模式: 单折模式 (Fold {args['test_fold'] + 1})\n")
        else:
            f.write(f"运行模式: 完整{args['num_folds']}折交叉验证\n")
        if all_fold_results:
            avg_f1 = np.mean([r['final_f1'] for r in all_fold_results])
            avg_acc = np.mean([r['final_accuracy'] for r in all_fold_results])
            avg_precision = np.mean([r['final_precision'] for r in all_fold_results])
            avg_asd_recall = np.mean([r['final_asd_recall'] for r in all_fold_results])
            avg_td_recall = np.mean([r['final_td_recall'] for r in all_fold_results])
            avg_macro_recall = np.mean([r['final_macro_recall'] for r in all_fold_results])
            std_f1 = np.std([r['final_f1'] for r in all_fold_results])
            std_acc = np.std([r['final_accuracy'] for r in all_fold_results])
            std_precision = np.std([r['final_precision'] for r in all_fold_results])
            std_asd_recall = np.std([r['final_asd_recall'] for r in all_fold_results])
            std_td_recall = np.std([r['final_td_recall'] for r in all_fold_results])
            std_macro_recall = np.std([r['final_macro_recall'] for r in all_fold_results])
            f.write(f"\n平均性能指标 (基于{len(all_fold_results)}个fold):\n")
            f.write(f"平均F1分数: {avg_f1:.4f} ± {std_f1:.4f}\n")
            f.write(f"平均准确率: {avg_acc:.4f} ± {std_acc:.4f}\n")
            f.write(f"平均精确率: {avg_precision:.4f} ± {std_precision:.4f}\n")
            f.write(f"平均ASD召回率(敏感度): {avg_asd_recall:.4f} ± {std_asd_recall:.4f}\n")
            f.write(f"平均TD召回率(特异度): {avg_td_recall:.4f} ± {std_td_recall:.4f}\n")
            f.write(f"平均宏平均召回率: {avg_macro_recall:.4f} ± {std_macro_recall:.4f}\n")
            f.write(f"\n各Fold详细结果:\n")
            for result in all_fold_results:
                f.write(f"Fold {result['fold']}: F1={result['final_f1']:.4f}, Acc={result['final_accuracy']:.4f}, "
                       f"Precision={result['final_precision']:.4f}, ASD_Recall={result['final_asd_recall']:.4f}, "
                       f"TD_Recall={result['final_td_recall']:.4f}, Macro_Recall={result['final_macro_recall']:.4f}\n")
        f.write(f"{'='*80}\n")
    json_results = {
        'experiment_info': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_path': args['data_path'],
            'feature_type': args['feature_type'],
            'batch_size': args['batch_size'],
            'learning_rate': args['learning_rate'],
            'num_epochs': args['num_epochs'],
            'num_folds': args['num_folds'],
            'model': args['model'],
            'lambda_aux': args['lambda_aux'],
            'seed': args['seed']
        },
        'fold_results': all_fold_results,
        'summary': {}
    }
    if all_fold_results:
        json_results['summary'] = {
            'avg_f1': float(np.mean([r['final_f1'] for r in all_fold_results])),
            'avg_accuracy': float(np.mean([r['final_accuracy'] for r in all_fold_results])),
            'avg_precision': float(np.mean([r['final_precision'] for r in all_fold_results])),
            'avg_asd_recall': float(np.mean([r['final_asd_recall'] for r in all_fold_results])),
            'avg_td_recall': float(np.mean([r['final_td_recall'] for r in all_fold_results])),
            'avg_macro_recall': float(np.mean([r['final_macro_recall'] for r in all_fold_results])),
            'avg_sensitivity': float(np.mean([r['final_sensitivity'] for r in all_fold_results])),
            'avg_specificity': float(np.mean([r['final_specificity'] for r in all_fold_results])),
            'std_f1': float(np.std([r['final_f1'] for r in all_fold_results])),
            'std_accuracy': float(np.std([r['final_accuracy'] for r in all_fold_results])),
            'std_precision': float(np.std([r['final_precision'] for r in all_fold_results])),
            'std_asd_recall': float(np.std([r['final_asd_recall'] for r in all_fold_results])),
            'std_td_recall': float(np.std([r['final_td_recall'] for r in all_fold_results])),
            'std_macro_recall': float(np.std([r['final_macro_recall'] for r in all_fold_results])),
            'std_sensitivity': float(np.std([r['final_sensitivity'] for r in all_fold_results])),
            'std_specificity': float(np.std([r['final_specificity'] for r in all_fold_results]))
        }
    json_file = f"{args['save_dir']}/experiment_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"\n实验结果已保存到: {results_file}")
    print(f"JSON格式结果已保存到: {json_file}")

if __name__ == '__main__':
    main()
