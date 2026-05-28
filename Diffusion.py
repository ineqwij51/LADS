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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
from datetime import datetime
import gc

warnings.filterwarnings('ignore')


def collate_fn_with_padding(batch):
    """Custom batch processing function for variable-length sequences"""
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

    # Pad sequences
    exp_padded = pad_sequence(exp_seqs, batch_first=True, padding_value=0)
    sub_padded = pad_sequence(sub_seqs, batch_first=True, padding_value=0)

    # Create masks
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


class DTWEncoder(nn.Module):
    """DTW-based difference encoder - computes DTW matrix and encodes it as features"""

    def __init__(self, feature_dim, hidden_dim=32, output_dim=16):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Feature transformation for DTW computation
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # DTW matrix encoder - encodes the DTW alignment matrix into features
        self.dtw_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, output_dim)
        )

    def compute_dtw_matrix(self, exp_features, sub_features, exp_mask, sub_mask):
        """Compute DTW distance matrix between EXP and SUB sequences"""
        B, T1, F = exp_features.shape
        _, T2, _ = sub_features.shape

        # Transform features
        exp_transformed = self.feature_transform(exp_features)
        sub_transformed = self.feature_transform(sub_features)

        # Compute pairwise distances
        exp_expanded = exp_transformed.unsqueeze(2)  # (B, T1, 1, F)
        sub_expanded = sub_transformed.unsqueeze(1)  # (B, 1, T2, F)

        # L2 distance
        distances = torch.norm(exp_expanded - sub_expanded, dim=-1)  # (B, T1, T2)

        # Apply masks
        mask_2d = exp_mask.unsqueeze(2) & sub_mask.unsqueeze(1)
        distances = distances.masked_fill(~mask_2d, float('inf'))

        # Normalize distances
        distances = distances / (distances.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)
        distances = distances.masked_fill(~mask_2d, 0)

        return distances

    def forward(self, exp_seq, sub_seq, exp_mask, sub_mask):
        """Forward pass - returns DTW encoding"""
        # Compute DTW matrix
        dtw_matrix = self.compute_dtw_matrix(exp_seq, sub_seq, exp_mask, sub_mask)

        # Encode DTW matrix
        B = dtw_matrix.shape[0]
        dtw_matrix_expanded = dtw_matrix.unsqueeze(1)  # Add channel dimension
        dtw_encoding = self.dtw_encoder(dtw_matrix_expanded)

        return dtw_encoding, dtw_matrix


class ConditionalDiffusionGenerator(nn.Module):
    """Conditional diffusion model to generate ideal SUB sequence from EXP"""

    def __init__(self, feature_dim, hidden_dim=64, num_diffusion_steps=20):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_diffusion_steps = num_diffusion_steps

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Condition encoder (for EXP features)
        # Ensure d_model is divisible by nhead
        nhead = 4
        d_model = feature_dim
        if d_model % nhead != 0:
            # Pad to nearest multiple of nhead
            d_model = ((d_model // nhead) + 1) * nhead
            self.feature_pad = nn.Linear(feature_dim, d_model)
            self.feature_unpad = nn.Linear(d_model, feature_dim)
        else:
            self.feature_pad = nn.Identity()
            self.feature_unpad = nn.Identity()
        
        self.condition_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=hidden_dim,
                dropout=0.2,
                batch_first=True
            ),
            num_layers=1
        )

        # Denoising U-Net style network
        self.denoise_net = nn.ModuleList([
            # Encoder
            ResidualBlock(feature_dim + hidden_dim, hidden_dim),
            # Bottleneck
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=2,
                dim_feedforward=hidden_dim * 2,
                dropout=0.2,
                batch_first=True
            ),
            # Decoder
            ResidualBlock(hidden_dim, feature_dim),
        ])

        # Condition projection layer to match denoising network dimensions
        self.condition_proj = nn.Linear(feature_dim, hidden_dim)

        # Final projection
        self.final_proj = nn.Linear(feature_dim, feature_dim)

        # Initialize noise schedule
        self.register_buffer('betas', self._cosine_beta_schedule(num_diffusion_steps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

        # Frame position embedding
        self.max_seq_len = 2000  # Increased to handle longer sequences
        self.frame_pos_emb = nn.Embedding(self.max_seq_len, d_model)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule for noise"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt()
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt()

        while len(sqrt_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def denoise(self, noisy_seq, exp_condition, t, mask=None):
        """Denoise one step"""
        B, T, F = noisy_seq.shape

        # Time embedding
        t_emb = self.time_mlp(t).unsqueeze(1).expand(B, T, -1)
        
        # Safety check for sequence length
        if T > self.max_seq_len:
            print(f"Warning: Sequence length {T} exceeds max_seq_len {self.max_seq_len}. Truncating.")

        # Condition encoding
        # Pad features if necessary
        exp_condition_padded = self.feature_pad(exp_condition)
        


        # Frame position embedding
        B, T, _ = exp_condition_padded.shape
        # Ensure position indices don't exceed embedding size
        max_pos = min(T, self.max_seq_len)
        idx = torch.arange(max_pos, device=exp_condition_padded.device).unsqueeze(0).expand(B, max_pos)
        
        # Get position embeddings
        pos_emb = self.frame_pos_emb(idx)
        
        # If sequence is longer than max_seq_len, truncate or use modulo
        if T > self.max_seq_len:
            # Option 1: Truncate to max_seq_len
            exp_condition_padded = exp_condition_padded[:, :max_pos, :] + pos_emb
        else:
            exp_condition_padded = exp_condition_padded + pos_emb
        
        if mask is not None:
            # Ensure mask has correct shape and type
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)  # Add batch dimension if needed
            
            # Ensure mask has correct batch size
            if mask.shape[0] != exp_condition_padded.shape[0]:
                mask = mask.expand(exp_condition_padded.shape[0], -1)
            
            # Ensure mask has correct sequence length (after potential truncation)
            if mask.shape[1] != exp_condition_padded.shape[1]:
                # Pad or truncate mask to match sequence length
                if mask.shape[1] < exp_condition_padded.shape[1]:
                    # Pad with False (invalid tokens)
                    pad_size = exp_condition_padded.shape[1] - mask.shape[1]
                    mask = torch.cat([mask, torch.zeros(mask.shape[0], pad_size, dtype=mask.dtype, device=mask.device)], dim=1)
                else:
                    # Truncate
                    mask = mask[:, :exp_condition_padded.shape[1]]
            
            src_key_padding_mask = ~mask
            cond_features_padded = self.condition_encoder(exp_condition_padded, src_key_padding_mask=src_key_padding_mask)
        else:
            cond_features_padded = self.condition_encoder(exp_condition_padded)
        
        # Unpad features back to original dimension
        cond_features = self.feature_unpad(cond_features_padded)

        # Concatenate noisy sequence with time embedding
        x = torch.cat([noisy_seq, t_emb], dim=-1)

        # Pass through denoising network
        for i, layer in enumerate(self.denoise_net):
            if i == 0:  # Encoder block
                x = layer(x)
            elif i == 1:  # Transformer bottleneck
                # Project condition features to match x dimension for conditioning
                cond_features_proj = self.condition_proj(cond_features[:, :x.shape[1], :])

                # Add conditioning via cross-attention
                x = x + cond_features_proj  # Simple addition for conditioning
                if mask is not None:
                    # Ensure mask has correct shape for transformer
                    transformer_mask = ~mask[:, :x.shape[1]]
                    
                    # Ensure mask has correct batch size
                    if transformer_mask.shape[0] != x.shape[0]:
                        transformer_mask = transformer_mask.expand(x.shape[0], -1)
                    
                    # Ensure mask has correct sequence length
                    if transformer_mask.shape[1] != x.shape[1]:
                        if transformer_mask.shape[1] < x.shape[1]:
                            # Pad with True (invalid tokens)
                            pad_size = x.shape[1] - transformer_mask.shape[1]
                            transformer_mask = torch.cat([transformer_mask, torch.ones(transformer_mask.shape[0], pad_size, dtype=transformer_mask.dtype, device=transformer_mask.device)], dim=1)
                        else:
                            # Truncate
                            transformer_mask = transformer_mask[:, :x.shape[1]]
                    
                    x = layer(x, src_key_padding_mask=transformer_mask)
                else:
                    x = layer(x)
            elif i == 2:  # Decoder block
                x = layer(x)

        # Final projection
        predicted_noise = self.final_proj(x)

        # Apply mask
        if mask is not None:
            predicted_noise = predicted_noise * mask.unsqueeze(-1).float()

        return predicted_noise

    def generate_ideal_sub(self, exp_seq, exp_mask=None, num_inference_steps=None):
        """Generate ideal SUB sequence given EXP condition"""
        B, T, F = exp_seq.shape
        device = exp_seq.device

        if num_inference_steps is None:
            num_inference_steps = self.num_diffusion_steps

        # Start from random noise
        x = torch.randn(B, T, F, device=device)

        # Denoising loop
        for t in reversed(range(num_inference_steps)):
            t_batch = torch.full((B,), t, device=device)

            # Predict noise
            predicted_noise = self.denoise(x, exp_seq, t_batch, exp_mask)

            # Update x (simplified DDPM sampling)
            alpha_t = self.alphas_cumprod[t]
            alpha_t_bar = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)

            beta_t = 1 - alpha_t / alpha_t_bar if t > 0 else self.betas[0]

            # Compute mean
            mean = (x - beta_t * predicted_noise / torch.sqrt(1 - alpha_t)) / torch.sqrt(1 - beta_t)

            # Add noise (except for last step)
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean

        return x

    def forward(self, exp_seq, exp_mask=None):
        """Forward pass - generate ideal SUB and return it"""
        try:
            ideal_sub = self.generate_ideal_sub(exp_seq, exp_mask, num_inference_steps=5)  # Fewer steps for training
            return ideal_sub
        except Exception as e:
            print(f"Error in diffusion generator: {e}")
            # Fallback: return a simple copy of exp_seq if generation fails
            print("Using fallback: returning copy of exp_seq")
            return exp_seq.clone()


class DifferenceEncoder(nn.Module):
    """Encode differences between actual SUB, ideal SUB, and DTW features"""

    def __init__(self, feature_dim, dtw_dim=16, hidden_dim=64, output_dim=32):
        super().__init__()
        self.feature_dim = feature_dim

        # Feature projection
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)

        # Difference attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=2,
            dropout=0.2,
            batch_first=True
        )

        # Difference encoder network
        self.diff_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 3 + dtw_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, actual_sub, ideal_sub, dtw_features, mask=None):
        """Encode differences"""
        # Project features
        actual_proj = self.feature_proj(actual_sub)
        ideal_proj = self.feature_proj(ideal_sub)

        # Compute attention-based difference
        attn_diff, _ = self.cross_attention(
            query=actual_proj,
            key=ideal_proj,
            value=ideal_proj,
            key_padding_mask=~mask if mask is not None else None
        )

        # Direct difference
        direct_diff = actual_proj - ideal_proj

        # Pool sequences to get global features
        if mask is not None:
            # Masked mean pooling
            mask_expanded = mask.unsqueeze(-1).float()
            actual_pooled = (actual_proj * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            ideal_pooled = (ideal_proj * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            attn_pooled = (attn_diff * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            actual_pooled = actual_proj.mean(dim=1)
            ideal_pooled = ideal_proj.mean(dim=1)
            attn_pooled = attn_diff.mean(dim=1)

        # Concatenate all features
        combined_features = torch.cat([
            actual_pooled,
            ideal_pooled,
            attn_pooled,
            dtw_features
        ], dim=-1)

        # Encode differences
        diff_encoding = self.diff_encoder(combined_features)

        return diff_encoding


class ImitationDiffusionClassifier(nn.Module):
    """Main model for autism classification using diffusion-based imitation analysis"""

    def __init__(
            self,
            feature_dim,
            hidden_dim=128,
            dtw_output_dim=32,
            diff_output_dim=64,
            num_classes=2,
            num_diffusion_steps=50,
            dropout=0.1,
            use_contrastive=False,
            contrastive_config=None
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_contrastive = use_contrastive

        # DTW encoder
        self.dtw_encoder = DTWEncoder(feature_dim, hidden_dim, dtw_output_dim)

        # Conditional diffusion generator
        self.diffusion_generator = ConditionalDiffusionGenerator(
            feature_dim, hidden_dim, num_diffusion_steps
        )

        # Difference encoder
        self.difference_encoder = DifferenceEncoder(
            feature_dim, dtw_output_dim, hidden_dim, diff_output_dim
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(diff_output_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Contrastive learning module (optional)
        if use_contrastive:
            contrastive_config = contrastive_config or {}
            self.contrastive_module = ContrastiveLearningModule(
                feature_dim=feature_dim,
                projection_dim=contrastive_config.get('projection_dim', 64),
                temperature=contrastive_config.get('temperature', 0.1)
            )
        else:
            self.contrastive_module = None

    def forward(self, exp_seq, sub_seq, exp_mask=None, sub_mask=None, return_intermediates=False, return_contrastive=False):
        """Forward pass"""
        # 1. Compute DTW encoding
        dtw_encoding, dtw_matrix = self.dtw_encoder(exp_seq, sub_seq, exp_mask, sub_mask)

        # 2. Generate ideal SUB sequence using diffusion
        ideal_sub = self.diffusion_generator(exp_seq, exp_mask)

        # 3. Encode differences
        diff_encoding = self.difference_encoder(sub_seq, ideal_sub, dtw_encoding, exp_mask)

        # 4. Classify
        logits = self.classifier(diff_encoding)

        # 5. Contrastive learning (if enabled)
        contrastive_outputs = None
        if self.use_contrastive and self.contrastive_module is not None:
            exp_embeddings, sub_embeddings = self.contrastive_module(
                exp_seq, sub_seq, exp_mask, sub_mask
            )
            contrastive_outputs = {
                'exp_embeddings': exp_embeddings,
                'sub_embeddings': sub_embeddings
            }

        if return_intermediates:
            intermediates = {
                'dtw_matrix': dtw_matrix,
                'dtw_encoding': dtw_encoding,
                'ideal_sub': ideal_sub,
                'diff_encoding': diff_encoding
            }
            if contrastive_outputs:
                intermediates.update(contrastive_outputs)
            return logits, intermediates

        if return_contrastive:
            return logits, contrastive_outputs

        return logits


# Helper modules
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return F.gelu(self.block(x) + self.residual(x))


class ContrastiveLearningModule(nn.Module):
    """Contrastive learning module for EXP-SUB pairs"""
    
    def __init__(self, feature_dim, projection_dim=64, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
        # Projection heads for EXP and SUB features
        self.exp_projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        self.sub_projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # Normalize projections
        self.exp_projection_norm = nn.LayerNorm(projection_dim)
        self.sub_projection_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, exp_features, sub_features, exp_mask=None, sub_mask=None):
        """Project EXP and SUB features to contrastive space"""
        B, T, F = exp_features.shape
        
        # Project features
        exp_proj = self.exp_projection(exp_features)
        sub_proj = self.sub_projection(sub_features)
        
        # Normalize
        exp_proj = self.exp_projection_norm(exp_proj)
        sub_proj = self.sub_projection_norm(sub_proj)
        
        # Masked pooling to get global representations
        if exp_mask is not None and sub_mask is not None:
            # Masked mean pooling
            exp_mask_expanded = exp_mask.unsqueeze(-1).float()
            sub_mask_expanded = sub_mask.unsqueeze(-1).float()
            
            exp_global = (exp_proj * exp_mask_expanded).sum(dim=1) / (exp_mask_expanded.sum(dim=1) + 1e-8)
            sub_global = (sub_proj * sub_mask_expanded).sum(dim=1) / (sub_mask_expanded.sum(dim=1) + 1e-8)
        else:
            exp_global = exp_proj.mean(dim=1)
            sub_global = sub_proj.mean(dim=1)
        
        # L2 normalize for cosine similarity
        exp_global = torch.nn.functional.normalize(exp_global, p=2, dim=1)
        sub_global = torch.nn.functional.normalize(sub_global, p=2, dim=1)
        
        return exp_global, sub_global


class ContrastiveLoss(nn.Module):
    """Contrastive learning loss for EXP-SUB pairs"""
    
    def __init__(self, temperature=0.1, mode='three_way'):
        super().__init__()
        self.temperature = temperature
        self.mode = mode  # 'three_way' or 'binary'
        
    def forward(self, exp_embeddings, sub_embeddings, labels, batch_indices=None):
        """
        Compute contrastive loss
        
        Args:
            exp_embeddings: (N, D) EXP embeddings
            sub_embeddings: (N, D) SUB embeddings  
            labels: (N,) labels (0=TD, 1=ASD)
            batch_indices: (N,) batch indices for hard negative mining
        """
        N = exp_embeddings.shape[0]
        device = exp_embeddings.device
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(exp_embeddings, sub_embeddings.T) / self.temperature
        
        if self.mode == 'three_way':
            return self._three_way_contrastive_loss(similarity_matrix, labels, batch_indices)
        else:
            return self._binary_contrastive_loss(similarity_matrix, labels, batch_indices)
    
    def _three_way_contrastive_loss(self, similarity_matrix, labels, batch_indices=None):
        """Three-way contrastive loss: TD pairs (positive), ASD pairs (negative), unmatched pairs (negative)"""
        N = similarity_matrix.shape[0]
        device = similarity_matrix.device
        
        # Create binary labels: 1 for positive pairs (TD), 0 for negative pairs (ASD + unmatched)
        positive_labels = torch.zeros(N, N, device=device)
        
        # TD pairs are positive
        td_mask = (labels == 0)
        td_indices = torch.where(td_mask)[0]
        for i in td_indices:
            positive_labels[i, i] = 1  # Positive pair
        
        # All other pairs (ASD pairs and unmatched pairs) are negative
        # This is already 0 by default
        
        # Compute InfoNCE-style contrastive loss
        # Positive pairs should have high similarity, negative pairs should have low similarity
        positive_mask = (positive_labels == 1)
        negative_mask = (positive_labels == 0)
        
        if positive_mask.sum() > 0 and negative_mask.sum() > 0:
            positive_similarities = similarity_matrix[positive_mask]
            negative_similarities = similarity_matrix[negative_mask]
            
            # Pull positive pairs closer, push negative pairs apart
            positive_loss = -torch.mean(positive_similarities)
            negative_loss = torch.mean(F.relu(negative_similarities + 1.0))  # Margin of 1.0
            
            loss = positive_loss + negative_loss
        else:
            # Fallback to simple binary cross-entropy if no positive/negative pairs
            loss = F.binary_cross_entropy_with_logits(
                similarity_matrix / self.temperature, 
                positive_labels
            )
        
        return loss
    
    def _binary_contrastive_loss(self, similarity_matrix, labels, batch_indices=None):
        """Binary contrastive loss: TD pairs (positive), others (negative)"""
        N = similarity_matrix.shape[0]
        device = similarity_matrix.device
        
        # Create binary labels: 1 for positive pairs (TD), 0 for negative pairs
        binary_labels = torch.zeros(N, N, device=device)
        
        # TD pairs are positive
        td_mask = (labels == 0)
        td_indices = torch.where(td_mask)[0]
        for i in td_indices:
            binary_labels[i, i] = 1  # Positive pair
        
        # All other pairs are negative
        for i in range(N):
            for j in range(N):
                if i != j or labels[i] == 1:  # ASD pairs or unmatched pairs
                    binary_labels[i, j] = 0  # Negative pair
        
        # Compute InfoNCE-style contrastive loss for better stability
        positive_mask = (binary_labels == 1)
        negative_mask = (binary_labels == 0)
        
        if positive_mask.sum() > 0 and negative_mask.sum() > 0:
            positive_similarities = similarity_matrix[positive_mask]
            negative_similarities = similarity_matrix[negative_mask]
            
            # Pull positive pairs closer, push negative pairs apart
            positive_loss = -torch.mean(positive_similarities)
            negative_loss = torch.mean(F.relu(negative_similarities + 1.0))  # Margin of 1.0
            
            loss = positive_loss + negative_loss
        else:
            # Fallback to binary cross-entropy
            loss = F.binary_cross_entropy_with_logits(
                similarity_matrix / self.temperature, 
                binary_labels
            )
        
        return loss


# Dataset class (simplified from original)
class AutismDataset(Dataset):
    """Autism dataset loader"""

    def __init__(self, data_path, feature_type='skeleton', transform=None):
        self.data_path = Path(data_path)
        self.feature_type = feature_type
        self.transform = transform

        # Load data
        if self.data_path.suffix == '.pkl':
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = torch.load(self.data_path)

        # Convert if needed
        if 'exp_features' in self.data:
            self._convert_pytorch_format()

    def _convert_pytorch_format(self):
        """Convert PyTorch format to unified format"""
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

        # Extract features
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

        # Convert to tensor
        exp_tensor = torch.FloatTensor(exp_features)
        sub_tensor = torch.FloatTensor(sub_features)

        # Get label
        label = 1 if sample['metadata']['label'] == 'ASD' else 0
        label_tensor = torch.LongTensor([label]).squeeze()

        # Apply transform
        if self.transform:
            exp_tensor = self.transform(exp_tensor)
            sub_tensor = self.transform(sub_tensor)

        return {
            'exp': exp_tensor,
            'sub': sub_tensor,
            'label': label_tensor,
            'sub_id': sample['metadata']['sub_id']
        }


# Training and evaluation functions
def train_epoch(model, train_loader, optimizer, criterion, device, scheduler=None, 
                use_contrastive=False, contrastive_criterion=None, contrastive_weight=0.1):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    total_contrastive_loss = 0

    progress_bar = tqdm(train_loader, desc='Training')
    for batch in progress_bar:
        exp_seq = batch['exp'].to(device)
        sub_seq = batch['sub'].to(device)
        exp_mask = batch['exp_mask'].to(device)
        sub_mask = batch['sub_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward pass
        if use_contrastive:
            logits, contrastive_outputs = model(exp_seq, sub_seq, exp_mask, sub_mask, return_contrastive=True)
        else:
            logits = model(exp_seq, sub_seq, exp_mask, sub_mask)

        # Compute classification loss
        classification_loss = criterion(logits, labels)
        loss = classification_loss

        # Compute contrastive loss if enabled
        contrastive_loss = 0
        if use_contrastive and contrastive_criterion is not None and contrastive_outputs is not None:
            exp_embeddings = contrastive_outputs['exp_embeddings']
            sub_embeddings = contrastive_outputs['sub_embeddings']
            contrastive_loss = contrastive_criterion(exp_embeddings, sub_embeddings, labels)
            loss = loss + contrastive_weight * contrastive_loss

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Statistics
        total_loss += classification_loss.item() * exp_seq.size(0)
        if use_contrastive:
            total_contrastive_loss += contrastive_loss.item() * exp_seq.size(0)
        pred = torch.argmax(logits, dim=1)
        total_correct += (pred == labels).sum().item()
        total_samples += exp_seq.size(0)

        # Update progress bar
        postfix = {
            'cls_loss': classification_loss.item(),
            'acc': total_correct / total_samples
        }
        if use_contrastive:
            postfix['contr_loss'] = contrastive_loss.item()
            postfix['total_loss'] = loss.item()
        progress_bar.set_postfix(postfix)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    avg_contrastive_loss = total_contrastive_loss / total_samples if use_contrastive else 0

    return avg_loss, avg_acc, avg_contrastive_loss


def evaluate(model, val_loader, criterion, device):
    """Evaluate model"""
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

            # Forward pass
            logits = model(exp_seq, sub_seq, exp_mask, sub_mask)

            # Compute loss
            loss = criterion(logits, labels)

            # Collect predictions
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)

            total_loss += loss.item() * exp_seq.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        'loss': total_loss / len(val_loader.dataset),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'asd_recall': recall_per_class[1] if len(recall_per_class) > 1 else 0,
        'td_recall': recall_per_class[0] if len(recall_per_class) > 0 else 0,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': np.array(all_probs)
    }

    return metrics


# Subject-independent K-fold cross-validation
class SubjectIndependentKFold:
    """Subject-independent K-fold cross-validation"""

    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, dataset):
        """Generate train/val/test indices"""
        # Collect subject indices
        subject_indices = {}

        for idx in range(len(dataset)):
            sub_id = dataset[idx]['sub_id']
            if sub_id not in subject_indices:
                subject_indices[sub_id] = []
            subject_indices[sub_id].append(idx)

        # Get subject IDs
        subject_ids = list(subject_indices.keys())

        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(subject_ids)

        # Split subjects into folds
        n_subjects = len(subject_ids)
        subjects_per_fold = n_subjects // self.n_splits
        remainder = n_subjects % self.n_splits

        fold_subjects = []
        start_idx = 0
        for i in range(self.n_splits):
            fold_size = subjects_per_fold + (1 if i < remainder else 0)
            end_idx = start_idx + fold_size
            fold_subjects.append(subject_ids[start_idx:end_idx])
            start_idx = end_idx

        # Generate train/val/test splits
        for fold_idx in range(self.n_splits):
            # Test subjects
            test_subjects = fold_subjects[fold_idx]
            test_indices = []
            for sub_id in test_subjects:
                test_indices.extend(subject_indices[sub_id])

            # Train+val subjects
            trainval_subjects = []
            for i in range(self.n_splits):
                if i != fold_idx:
                    trainval_subjects.extend(fold_subjects[i])

            # Split train+val
            np.random.seed(self.random_state + fold_idx)
            np.random.shuffle(trainval_subjects)

            n_trainval = len(trainval_subjects)
            n_train = int(0.8 * n_trainval)

            train_subjects = trainval_subjects[:n_train]
            val_subjects = trainval_subjects[n_train:]

            # Get indices
            train_indices = []
            for sub_id in train_subjects:
                train_indices.extend(subject_indices[sub_id])

            val_indices = []
            for sub_id in val_subjects:
                val_indices.extend(subject_indices[sub_id])

            yield train_indices, val_indices, test_indices


# Visualization functions
def visualize_dtw_patterns(model, dataset, device, save_path, num_samples=5):
    """Visualize DTW patterns for ASD vs TD"""
    model.eval()

    # Collect samples
    asd_samples = []
    td_samples = []

    for i in range(len(dataset)):
        sample = dataset[i]
        if sample['label'].item() == 1 and len(asd_samples) < num_samples:
            asd_samples.append(sample)
        elif sample['label'].item() == 0 and len(td_samples) < num_samples:
            td_samples.append(sample)
        if len(asd_samples) >= num_samples and len(td_samples) >= num_samples:
            break

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    with torch.no_grad():
        # Collect DTW statistics
        asd_dtw_stats = []
        td_dtw_stats = []
        
        # Process ASD samples
        for sample in asd_samples:
            exp_seq = sample['exp'].unsqueeze(0).to(device)
            sub_seq = sample['sub'].unsqueeze(0).to(device)
            exp_mask = torch.ones(1, exp_seq.size(1), dtype=torch.bool).to(device)
            sub_mask = torch.ones(1, sub_seq.size(1), dtype=torch.bool).to(device)

            _, intermediates = model(exp_seq, sub_seq, exp_mask, sub_mask, return_intermediates=True)
            dtw_matrix = intermediates['dtw_matrix'][0].cpu().numpy()
            
            # Extract DTW statistics
            mean_distance = np.mean(dtw_matrix[dtw_matrix > 0])
            min_distance = np.min(dtw_matrix[dtw_matrix > 0])
            max_distance = np.max(dtw_matrix)
            diagonal_mean = np.mean(np.diag(dtw_matrix))
            
            asd_dtw_stats.append({
                'mean': mean_distance,
                'min': min_distance,
                'max': max_distance,
                'diagonal': diagonal_mean
            })

        # Process TD samples
        for sample in td_samples:
            exp_seq = sample['exp'].unsqueeze(0).to(device)
            sub_seq = sample['sub'].unsqueeze(0).to(device)
            exp_mask = torch.ones(1, exp_seq.size(1), dtype=torch.bool).to(device)
            sub_mask = torch.ones(1, sub_seq.size(1), dtype=torch.bool).to(device)

            _, intermediates = model(exp_seq, sub_seq, exp_mask, sub_mask, return_intermediates=True)
            dtw_matrix = intermediates['dtw_matrix'][0].cpu().numpy()
            
            # Extract DTW statistics
            mean_distance = np.mean(dtw_matrix[dtw_matrix > 0])
            min_distance = np.min(dtw_matrix[dtw_matrix > 0])
            max_distance = np.max(dtw_matrix)
            diagonal_mean = np.mean(np.diag(dtw_matrix))
            
            td_dtw_stats.append({
                'mean': mean_distance,
                'min': min_distance,
                'max': max_distance,
                'diagonal': diagonal_mean
            })

        # Plot comparisons
        stats_keys = ['mean', 'min', 'max', 'diagonal']
        stat_names = ['Mean Distance', 'Min Distance', 'Max Distance', 'Diagonal Mean']
        
        for i, (key, name) in enumerate(zip(stats_keys, stat_names)):
            ax = axes[i//2, i%2]
            
            asd_values = [s[key] for s in asd_dtw_stats]
            td_values = [s[key] for s in td_dtw_stats]
            
            # Box plot
            bp = ax.boxplot([asd_values, td_values], labels=['ASD', 'TD'], patch_artist=True)
            bp['boxes'][0].set_facecolor('red')
            bp['boxes'][0].set_alpha(0.6)
            bp['boxes'][1].set_facecolor('blue')
            bp['boxes'][1].set_alpha(0.6)
            
            ax.set_title(f'DTW {name}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Distance Value', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add statistical annotation
            from scipy import stats
            if len(asd_values) > 1 and len(td_values) > 1:
                t_stat, p_val = stats.ttest_ind(asd_values, td_values)
                ax.text(0.5, 0.95, f'p = {p_val:.3f}', transform=ax.transAxes, 
                       ha='center', va='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{save_path}/dtw_patterns_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()





def visualize_ideal_vs_actual_skeleton(model, dataset, device, save_path, feature_type='skeleton'):
    """Visualize ideal (diffusion-generated) vs actual skeleton sequences"""
    model.eval()

    # Find one ASD and one TD sample
    asd_sample = None
    td_sample = None

    for i in range(len(dataset)):
        sample = dataset[i]
        if asd_sample is None and sample['label'].item() == 1:
            asd_sample = (i, sample)
        elif td_sample is None and sample['label'].item() == 0:
            td_sample = (i, sample)
        if asd_sample and td_sample:
            break

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    samples = [('ASD', asd_sample), ('TD', td_sample)]

    with torch.no_grad():
        for row_idx, (label_type, sample_data) in enumerate(samples):
            if sample_data is None:
                continue

            idx, sample = sample_data
            exp_seq = sample['exp'].unsqueeze(0).to(device)
            sub_seq = sample['sub'].unsqueeze(0).to(device)
            exp_mask = torch.ones(1, exp_seq.size(1), dtype=torch.bool).to(device)
            sub_mask = torch.ones(1, sub_seq.size(1), dtype=torch.bool).to(device)

            # Get ideal SUB
            _, intermediates = model(exp_seq, sub_seq, exp_mask, sub_mask, return_intermediates=True)
            ideal_sub = intermediates['ideal_sub'][0].cpu().numpy()
            actual_sub = sub_seq[0].cpu().numpy()
            exp_np = exp_seq[0].cpu().numpy()

            # If using skeleton features, reshape to (T, 33, 4) or appropriate shape
            if feature_type == 'skeleton' and actual_sub.shape[1] == 132:  # 33 * 4
                actual_sub = actual_sub.reshape(-1, 33, 4)
                ideal_sub = ideal_sub.reshape(-1, 33, 4)
                exp_np = exp_np.reshape(-1, 33, 4)

            # Select time frames to visualize
            time_frames = [0, len(actual_sub) // 3, 2 * len(actual_sub) // 3, len(actual_sub) - 1]

            for frame_idx, t in enumerate(time_frames):
                if t >= len(actual_sub):
                    t = len(actual_sub) - 1

                ax = axes[row_idx, frame_idx]

                if feature_type == 'skeleton' and actual_sub.shape[1] == 33:
                    # Plot skeleton points
                    ax.scatter(exp_np[t, :, 0], exp_np[t, :, 1],
                               c='blue', s=20, alpha=0.6, label='EXP')
                    ax.scatter(actual_sub[t, :, 0], actual_sub[t, :, 1],
                               c='green', s=20, alpha=0.6, label='Actual SUB')
                    ax.scatter(ideal_sub[t, :, 0], ideal_sub[t, :, 1],
                               c='red', s=20, alpha=0.6, label='Ideal SUB')
                else:
                    # For non-skeleton features, show feature distribution
                    ax.hist(exp_np[t], bins=20, alpha=0.4, label='EXP', color='blue')
                    ax.hist(actual_sub[t], bins=20, alpha=0.4, label='Actual', color='green')
                    ax.hist(ideal_sub[t], bins=20, alpha=0.4, label='Ideal', color='red')

                ax.set_title(f'{label_type} - Frame {t}')
                if frame_idx == 0:
                    ax.legend()
                ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}/ideal_vs_actual_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def visualize_difference_encoding(model, dataset, device, save_path, num_samples=50):
    """Visualize difference encodings for ASD vs TD with 2D reduction comparison"""
    model.eval()

    # Collect difference encodings and input features
    asd_diff_encodings = []
    td_diff_encodings = []
    asd_input_features = []
    td_input_features = []

    with torch.no_grad():
        for i in range(min(len(dataset), 200)):  # Use more samples for better visualization
            sample = dataset[i]
            exp_seq = sample['exp'].unsqueeze(0).to(device)
            sub_seq = sample['sub'].unsqueeze(0).to(device)
            exp_mask = torch.ones(1, exp_seq.size(1), dtype=torch.bool).to(device)
            sub_mask = torch.ones(1, sub_seq.size(1), dtype=torch.bool).to(device)

            _, intermediates = model(exp_seq, sub_seq, exp_mask, sub_mask, return_intermediates=True)
            diff_encoding = intermediates['diff_encoding'][0].cpu().numpy()
            
            # Get mean input features (EXP + SUB combined)
            exp_features = exp_seq[0].cpu().numpy().mean(axis=0)  # Average over time
            sub_features = sub_seq[0].cpu().numpy().mean(axis=0)  # Average over time
            combined_features = np.concatenate([exp_features, sub_features])

            if sample['label'].item() == 1 and len(asd_diff_encodings) < num_samples:
                asd_diff_encodings.append(diff_encoding)
                asd_input_features.append(combined_features)
            elif sample['label'].item() == 0 and len(td_diff_encodings) < num_samples:
                td_diff_encodings.append(diff_encoding)
                td_input_features.append(combined_features)

    # Convert to arrays
    asd_diff_encodings = np.array(asd_diff_encodings)
    td_diff_encodings = np.array(td_diff_encodings)
    asd_input_features = np.array(asd_input_features)
    td_input_features = np.array(td_input_features)

    if len(asd_diff_encodings) == 0 or len(td_diff_encodings) == 0:
        print("Warning: Not enough samples for visualization")
        return

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 2D PCA of Difference Encodings
    from sklearn.decomposition import PCA
    all_diff_encodings = np.vstack([asd_diff_encodings, td_diff_encodings])
    pca_diff = PCA(n_components=2)
    pca_diff_features = pca_diff.fit_transform(all_diff_encodings)

    axes[0, 0].scatter(pca_diff_features[:len(asd_diff_encodings), 0],
                       pca_diff_features[:len(asd_diff_encodings), 1],
                       c='red', label='ASD', alpha=0.7, s=30, edgecolors='darkred')
    axes[0, 0].scatter(pca_diff_features[len(asd_diff_encodings):, 0],
                       pca_diff_features[len(asd_diff_encodings):, 1],
                       c='blue', label='TD', alpha=0.7, s=30, edgecolors='darkblue')
    axes[0, 0].set_title('2D PCA of Difference Encodings\n(Shows Classification Effectiveness)', 
                         fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel(f'PC1 ({pca_diff.explained_variance_ratio_[0]:.1%} variance)')
    axes[0, 0].set_ylabel(f'PC2 ({pca_diff.explained_variance_ratio_[1]:.1%} variance)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 2D PCA of Input Features (for comparison)
    all_input_features = np.vstack([asd_input_features, td_input_features])
    pca_input = PCA(n_components=2)
    pca_input_features = pca_input.fit_transform(all_input_features)

    axes[0, 1].scatter(pca_input_features[:len(asd_input_features), 0],
                       pca_input_features[:len(asd_input_features), 1],
                       c='red', label='ASD', alpha=0.7, s=30, edgecolors='darkred')
    axes[0, 1].scatter(pca_input_features[len(asd_input_features):, 0],
                       pca_input_features[len(asd_input_features):, 1],
                       c='blue', label='TD', alpha=0.7, s=30, edgecolors='darkblue')
    axes[0, 1].set_title('2D PCA of Raw Input Features\n(For Comparison)', 
                         fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel(f'PC1 ({pca_input.explained_variance_ratio_[0]:.1%} variance)')
    axes[0, 1].set_ylabel(f'PC2 ({pca_input.explained_variance_ratio_[1]:.1%} variance)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. t-SNE of Difference Encodings
    from sklearn.manifold import TSNE
    tsne_diff = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_diff_encodings)//4))
    tsne_diff_features = tsne_diff.fit_transform(all_diff_encodings)

    axes[1, 0].scatter(tsne_diff_features[:len(asd_diff_encodings), 0],
                       tsne_diff_features[:len(asd_diff_encodings), 1],
                       c='red', label='ASD', alpha=0.7, s=30, edgecolors='darkred')
    axes[1, 0].scatter(tsne_diff_features[len(asd_diff_encodings):, 0],
                       tsne_diff_features[len(asd_diff_encodings):, 1],
                       c='blue', label='TD', alpha=0.7, s=30, edgecolors='darkblue')
    axes[1, 0].set_title('t-SNE of Difference Encodings\n(Non-linear Clustering)', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('t-SNE Component 1')
    axes[1, 0].set_ylabel('t-SNE Component 2')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Separation Quality Metrics
    axes[1, 1].axis('off')
    
    # Calculate separation metrics
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import cdist
    
    # Labels for difference encodings
    diff_labels = np.array([1] * len(asd_diff_encodings) + [0] * len(td_diff_encodings))
    input_labels = np.array([1] * len(asd_input_features) + [0] * len(td_input_features))
    
    # Silhouette scores
    sil_diff = silhouette_score(all_diff_encodings, diff_labels)
    sil_input = silhouette_score(all_input_features, input_labels)
    
    # Inter vs intra-class distances
    asd_diff_center = asd_diff_encodings.mean(axis=0)
    td_diff_center = td_diff_encodings.mean(axis=0)
    inter_class_dist_diff = np.linalg.norm(asd_diff_center - td_diff_center)
    
    asd_input_center = asd_input_features.mean(axis=0)
    td_input_center = td_input_features.mean(axis=0)
    inter_class_dist_input = np.linalg.norm(asd_input_center - td_input_center)
    
    # Display metrics
    metrics_text = f"""
    SEPARATION QUALITY COMPARISON
    
    Difference Encodings:
    • Silhouette Score: {sil_diff:.3f}
    • Inter-class Distance: {inter_class_dist_diff:.3f}
    • PC1+PC2 Variance: {(pca_diff.explained_variance_ratio_[:2].sum()):.1%}
    
    Raw Input Features:
    • Silhouette Score: {sil_input:.3f}
    • Inter-class Distance: {inter_class_dist_input:.3f}
    • PC1+PC2 Variance: {(pca_input.explained_variance_ratio_[:2].sum()):.1%}
    
    Improvement Factor:
    • Silhouette: {sil_diff/sil_input:.2f}x better
    • Separation: {inter_class_dist_diff/inter_class_dist_input:.2f}x better
    """
    
    axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{save_path}/difference_encoding_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main training pipeline"""
    # Configuration
    args = {
        'data_path': 'autism_multimodal_dataset_20250726.pkl',
        'feature_type': 'skeleton',  # skeleton, sparse_flow, dense_flow, heatmap
        'batch_size': 8,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'num_folds': 5,
        'device': 'cuda:7' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'diffusion_results_skeleton_0813',
        'seed': 42,
        'early_stopping': True,
        'patience': 20,
        'min_delta': 0.001,
        # Contrastive learning settings
        'use_contrastive': True,  # Set to True to enable contrastive learning
        'contrastive_mode': 'three_way',  # 'three_way' or 'binary'
        'contrastive_temperature': 0.1,
        'contrastive_weight': 0.1,  # Weight for contrastive loss
        'contrastive_projection_dim': 64
    }

    # Set random seeds
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])

    # Create save directory
    Path(args['save_dir']).mkdir(exist_ok=True)

    # Create results file
    results_file = f"{args['save_dir']}/experiment_results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Autism Imitation Behavior Diffusion Classification Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {args['data_path']}\n")
        f.write(f"Feature type: {args['feature_type']}\n")
        f.write(f"Batch size: {args['batch_size']}\n")
        f.write(f"Learning rate: {args['learning_rate']}\n")
        f.write(f"Epochs: {args['num_epochs']}\n")
        f.write(f"K-folds: {args['num_folds']}\n")
        f.write(f"Contrastive Learning: {args['use_contrastive']}\n")
        if args['use_contrastive']:
            f.write(f"Contrastive Mode: {args['contrastive_mode']}\n")
            f.write(f"Contrastive Temperature: {args['contrastive_temperature']}\n")
            f.write(f"Contrastive Weight: {args['contrastive_weight']}\n")
        f.write("=" * 80 + "\n\n")

    # Load dataset (without data augmentation as requested)
    print(f"Loading dataset: {args['data_path']}")
    dataset = AutismDataset(args['data_path'], feature_type=args['feature_type'], transform=None)
    print(f"Dataset size: {len(dataset)}")

    # Get feature dimension
    sample = dataset[0]
    if len(sample['exp'].shape) == 3:
        feature_dim = sample['exp'].shape[1] * sample['exp'].shape[2]
    elif len(sample['exp'].shape) == 4:
        feature_dim = sample['exp'].shape[1] * sample['exp'].shape[2] * sample['exp'].shape[3]
    elif len(sample['exp'].shape) == 2:
        feature_dim = sample['exp'].shape[1]
    else:
        raise ValueError(f"Unsupported feature shape: {sample['exp'].shape}")

    print(f"Feature dimension: {feature_dim}")

    # K-fold cross-validation
    kfold = SubjectIndependentKFold(n_splits=args['num_folds'])
    all_fold_results = []

    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(kfold.split(dataset)):
        print(f"\n{'=' * 50}")
        print(f"Fold {fold_idx + 1}/{args['num_folds']}")
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        # Get subject IDs for each set
        train_sub_ids = [dataset[i]['sub_id'] for i in train_idx]
        val_sub_ids = [dataset[i]['sub_id'] for i in val_idx]
        test_sub_ids = [dataset[i]['sub_id'] for i in test_idx]
        
        print(f"Train subjects: {sorted(set(train_sub_ids))}")
        print(f"Val subjects: {sorted(set(val_sub_ids))}")
        print(f"Test subjects: {sorted(set(test_sub_ids))}")

        # Create data loaders
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

        # Create model
        contrastive_config = None
        if args['use_contrastive']:
            contrastive_config = {
                'projection_dim': args['contrastive_projection_dim'],
                'temperature': args['contrastive_temperature']
            }
        
        model = ImitationDiffusionClassifier(
            feature_dim=feature_dim,
            hidden_dim=64,
            dtw_output_dim=16,
            diff_output_dim=32,
            num_classes=2,
            num_diffusion_steps=20,
            dropout=0.3,
            use_contrastive=args['use_contrastive'],
            contrastive_config=contrastive_config
        ).to(args['device'])

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args['learning_rate'],
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args['num_epochs']
        )

        criterion = nn.CrossEntropyLoss()
        
        # Contrastive learning criterion
        contrastive_criterion = None
        if args['use_contrastive']:
            contrastive_criterion = ContrastiveLoss(
                temperature=args['contrastive_temperature'],
                mode=args['contrastive_mode']
            )

        # Training loop
        best_val_f1 = 0
        patience_counter = 0
        best_epoch = 0

        for epoch in range(args['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{args['num_epochs']}")

            # Train
            train_loss, train_acc, train_contrastive_loss = train_epoch(
                model, train_loader, optimizer, criterion, args['device'], scheduler,
                use_contrastive=args['use_contrastive'],
                contrastive_criterion=contrastive_criterion,
                contrastive_weight=args['contrastive_weight']
            )

            # Validate
            val_metrics = evaluate(model, val_loader, criterion, args['device'])

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            if args['use_contrastive']:
                print(f"Train Contrastive Loss: {train_contrastive_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}, ASD Recall: {val_metrics['asd_recall']:.4f}")

            # Save best model
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_epoch = epoch + 1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'args': args
                }, f"{args['save_dir']}/best_model_fold{fold_idx}.pt")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if args['early_stopping'] and patience_counter >= args['patience']:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model and evaluate on test set
        checkpoint = torch.load(f"{args['save_dir']}/best_model_fold{fold_idx}.pt")
        model.load_state_dict(checkpoint['model_state_dict'])

        test_metrics = evaluate(model, test_loader, criterion, args['device'])

        print(f"\nBest model (epoch {best_epoch}) test results:")
        print(f"Test Acc: {test_metrics['accuracy']:.4f}, Test F1: {test_metrics['f1']:.4f}")
        print(f"ASD Recall: {test_metrics['asd_recall']:.4f}, TD Recall: {test_metrics['td_recall']:.4f}")

        # Generate visualizations
        print("\nGenerating visualizations...")
        visualize_dtw_patterns(model, test_subset, args['device'], args['save_dir'])
        visualize_ideal_vs_actual_skeleton(model, test_subset, args['device'], args['save_dir'], args['feature_type'])
        visualize_difference_encoding(model, test_subset, args['device'], args['save_dir'])

        # Save fold results
        fold_result = {
            'fold': fold_idx + 1,
            'best_epoch': best_epoch,
            'best_val_f1': best_val_f1,
            'test_accuracy': test_metrics['accuracy'],
            'test_f1': test_metrics['f1'],
            'test_asd_recall': test_metrics['asd_recall'],
            'test_td_recall': test_metrics['td_recall'],
            'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
            'train_subjects': sorted(set(train_sub_ids)),
            'val_subjects': sorted(set(val_sub_ids)),
            'test_subjects': sorted(set(test_sub_ids))
        }
        all_fold_results.append(fold_result)

        # Write results
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(f"\nFold {fold_idx + 1} Results:\n")
            f.write(f"Best epoch: {best_epoch}\n")
            f.write(f"Test Accuracy: {test_metrics['accuracy']:.4f}\n")
            f.write(f"Test F1: {test_metrics['f1']:.4f}\n")
            f.write(f"ASD Recall: {test_metrics['asd_recall']:.4f}\n")
            f.write(f"TD Recall: {test_metrics['td_recall']:.4f}\n")
            if args['use_contrastive']:
                f.write(f"Contrastive Learning: Enabled\n")
            f.write(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}\n")
            f.write(f"Train subjects ({len(set(train_sub_ids))}): {sorted(set(train_sub_ids))}\n")
            f.write(f"Val subjects ({len(set(val_sub_ids))}): {sorted(set(val_sub_ids))}\n")
            f.write(f"Test subjects ({len(set(test_sub_ids))}): {sorted(set(test_sub_ids))}\n")
            f.write("-" * 50 + "\n")

        # Clean up
        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()

    # Summary statistics
    avg_acc = np.mean([r['test_accuracy'] for r in all_fold_results])
    avg_f1 = np.mean([r['test_f1'] for r in all_fold_results])
    avg_asd_recall = np.mean([r['test_asd_recall'] for r in all_fold_results])
    avg_td_recall = np.mean([r['test_td_recall'] for r in all_fold_results])

    std_acc = np.std([r['test_accuracy'] for r in all_fold_results])
    std_f1 = np.std([r['test_f1'] for r in all_fold_results])

    # Write summary
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("Summary Results:\n")
        f.write(f"Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f}\n")
        f.write(f"Average F1: {avg_f1:.4f} ± {std_f1:.4f}\n")
        f.write(f"Average ASD Recall: {avg_asd_recall:.4f}\n")
        f.write(f"Average TD Recall: {avg_td_recall:.4f}\n")
        f.write("=" * 80 + "\n")

    # Save JSON results
    json_results = {
        'experiment_info': args,
        'fold_results': all_fold_results,
        'summary': {
            'avg_accuracy': float(avg_acc),
            'avg_f1': float(avg_f1),
            'avg_asd_recall': float(avg_asd_recall),
            'avg_td_recall': float(avg_td_recall),
            'std_accuracy': float(std_acc),
            'std_f1': float(std_f1)
        },
        'contrastive_info': {
            'enabled': args['use_contrastive'],
            'mode': args['contrastive_mode'] if args['use_contrastive'] else None,
            'temperature': args['contrastive_temperature'] if args['use_contrastive'] else None,
            'weight': args['contrastive_weight'] if args['use_contrastive'] else None
        }
    }

    with open(f"{args['save_dir']}/results.json", 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nExperiment completed!")
    print(f"Results saved to: {args['save_dir']}")


if __name__ == '__main__':
    main()