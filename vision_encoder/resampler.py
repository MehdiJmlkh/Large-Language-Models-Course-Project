import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from functools import partial
import torch
import torch.nn as nn
from functools import partial
from .pos_embed import get_2d_sincos_pos_embed, get_abs_pos, get_matry_n


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
            self,
            grid_size,
            embed_dim,
            num_heads,
            kv_dim=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        self.num_queries = grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(
                kv_dim, grid_size)).to(torch.bfloat16)
        ).requires_grad_(False)

        self.query = nn.Parameter(torch.zeros(
            self.num_queries, kv_dim)).to(torch.bfloat16)
        trunc_normal_(self.query, std=.02)

        self.attn = nn.MultiheadAttention(kv_dim, num_heads).to(
            device="cuda:0", dtype=torch.bfloat16)

        self.ln_q = norm_layer(kv_dim).to(
            device="cuda:0", dtype=torch.bfloat16)
        self.ln_k = norm_layer(kv_dim).to(
            device="cuda:0", dtype=torch.bfloat16)
        self.ln_v = norm_layer(kv_dim).to(
            device="cuda:0", dtype=torch.bfloat16)

        self.proj = nn.Parameter((embed_dim ** -0.5) * torch.randn(
            kv_dim, embed_dim)).to(device="cuda:0", dtype=torch.bfloat16)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, num_visual_tokens=256, tgt_size=(24, 24), attn_mask=None):
        pos_embed = get_abs_pos(self.pos_embed, tgt_size)

        x = x.permute(1, 0, 2)  # x: (seq_len, batch_size, dim)
        B = x.shape[1]  # true batch size

        matry_n = get_matry_n(num_visual_tokens)
        q = self.query[:matry_n]  # (matry_n, dim)
        q = self._repeat(q, B)    # (matry_n, B, dim)

        k = self._repeat(pos_embed, B).to(
            device="cuda:0", dtype=torch.bfloat16)
        v = x
        q = q.to(device="cuda:0")

        q = self.ln_q(q + self.pos_embed[:matry_n].unsqueeze(1).to(
            device="cuda:0")).to(device=x.device, dtype=torch.bfloat16)
        k = self.ln_k(k).to(device=x.device, dtype=torch.bfloat16)
        v = self.ln_v(v).to(device=x.device, dtype=torch.bfloat16)

        out = self.attn(q, k, v, attn_mask=attn_mask)[0]  # (matry_n, B, dim)

        x = out.permute(1, 0, 2)

        x = x @ self.proj

        return x

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)
