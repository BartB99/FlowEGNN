import math

import torch
import torch.nn as nn

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, t_embd_dim):
        super().__init__()
        self.t_embd_dim = t_embd_dim

    def forward(self, t):
        # t: (batch_size, 1) — timestep values
        t = t * 850
        half = self.t_embd_dim // 2
        freqs = torch.exp(
            -math.log(10) * torch.arange(half, device=t.device) / (half - 1) # originally 10000
        )                                          # (half,)
        args = t * freqs                           # (batch_size, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (batch_size, t_embd_dim)
        return emb
    
class SinusoidalThetaEmbedding(nn.Module):
    def __init__(self, theta_param_embd_dim):
        super().__init__()
        self.theta_param_embd_dim = theta_param_embd_dim

    def forward(self, theta):
        # t: (batch_size, 1) — timestep values
        N, D = theta.shape
        half = self.theta_param_embd_dim // 2
        freqs = torch.exp(
            -math.log(2) * torch.arange(half, device=theta.device) / (half - 1)
        )                                          # (half,)
        embeddings = []
        for param in range(D):
            args = theta[:, param].unsqueeze(1) * freqs        # (batch_size, half)
            emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (batch_size, theta_param_embd_dim)
            embeddings.append(emb)
        return torch.cat(embeddings, dim=1)