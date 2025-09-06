import torch.nn as nn
from torch.nn import functional as F
from functools import partial
import torch.nn as nn
from functools import partial
from .resampler import Resampler


class MultiResampler(nn.Module):
    """
    experimental multi-resampler

    Args:
        num_resamplers:int = number of resamplers(qformers)

    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
            self,
            grid_size,
            embed_dim,
            num_heads,
            num_resamplers,
            kv_dim=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.resamplers = nn.ModuleList([Resampler(
            grid_size, embed_dim, num_heads, kv_dim, norm_layer) for _ in range(num_resamplers)])

    def _init_weights(self, m):
        for i in range(len(self.resamplers)):
            self.resamplers[i]._init_weights(m)

    def forward(self, x: list, num_visual_tokens=256, tgt_size=(27, 27), attn_mask=None):
        assert len(x) == len(self.resamplers)

        num_vt_each = num_visual_tokens//len(self.resamplers)
        outs = []
        for i in range(len(self.resamplers)):
            outs.append(self.resamplers[i].forward(
                x[i], num_vt_each, tgt_size))
        # concat = torch.cat(outs, dim=1)
        # print("multi resampler:", outs[0].shape)
        return outs
