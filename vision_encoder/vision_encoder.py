import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from functools import partial
from torchvision.transforms import functional as F_transforms
from huggingface_hub import PyTorchModelHubMixin
from .multi_resampler import MultiResampler


class MultiCropVisionEncoder(nn.Module, PyTorchModelHubMixin):
    def __init__(self, qformer_config, vision_tower):
        super().__init__()
        self.num_resamplers = qformer_config["num_resamplers"]
        self.embed_dim = qformer_config["embed_dim"]
        self.grid_size = qformer_config["grid_size"]

        self.vision_tower = vision_tower
        vision_hidden_size = self.vision_tower.config.hidden_size
        self.vision_tower_image_size = self.vision_tower.config.image_size

        self.vision_proj = nn.Linear(vision_hidden_size, self.embed_dim)

        self.multi_resampler = MultiResampler(
            grid_size=qformer_config["grid_size"],
            embed_dim=qformer_config["embed_dim"],
            num_heads=qformer_config["num_heads"],
            num_resamplers=qformer_config["num_resamplers"],
            kv_dim=qformer_config["kv_dim"],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

    def forward(self, pixel_values, patch_attention_mask=None):
        pixel_values = pixel_values.to(torch.bfloat16)
        b4, c, h, w = pixel_values.shape
        assert b4 % self.num_resamplers == 0, "Batch size must be divisible by number of resamplers"
        batch_size = b4 // self.num_resamplers

        pixel_values = pixel_values.view(
            batch_size, self.num_resamplers, c, h, w)
        patch_embeddings_list = []
        for i in range(self.num_resamplers):
            sub_batch = pixel_values[:, i, :, :, :]

            resized_sub_batch = F_transforms.resize(sub_batch, size=(432, 432))

            vision_outputs = self.vision_tower(
                pixel_values=resized_sub_batch, interpolate_pos_encoding=True)
            embeddings = vision_outputs.last_hidden_state

            embeddings = embeddings.to(torch.bfloat16)

            embeddings = embeddings[:, 1:, :]
            embeddings = self.vision_proj(embeddings)

            patch_embeddings_list.append(embeddings)

        out = self.multi_resampler(
            patch_embeddings_list, num_visual_tokens=self.grid_size**2 * self.num_resamplers)

        stacked = torch.stack(out, dim=1)
        out = stacked.view(-1, stacked.shape[-2], stacked.shape[-1])
        out = out.to(device=pixel_values.device, dtype=torch.bfloat16)

        return BaseModelOutput(last_hidden_state=out)
