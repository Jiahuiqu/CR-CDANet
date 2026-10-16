"""DADiff backbone: a time-conditioned AT-Mamba U-Net for cross-modal translation.

The network follows the dual-branch design of :mod:`src.models.at_mamba`, with
two differences required by domain-adaptive conditional diffusion:

* the conditioning modality and the modality to be generated are both
  three-channel images (RGB and SAR), so the two branches share one patch
  embedding;
* every stage of the conditional encoder is fused with the corresponding stage
  of the ``x_t`` encoder, and the conditional-encoder features of the three
  scales are returned so that the multi-scale joint domain adaptation loss
  (MMD / CORAL) can be applied to them.

The noise level enters through a sinusoidal embedding that modulates the ``x_t``
branch only.
"""

import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_

from .blocks import (Final_PatchExpand2D, PatchEmbed2D, PatchMerging2D, PatchExpand2D,
                     SS2D, UNET_AttentionBlock, UNET_ResidualBlock)


class TimeEmbedding(nn.Module):
    """MLP that lifts the noise-level embedding to ``4 * n_embd`` channels."""

    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal encoding of the continuous noise level."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.count = self.dim // 2
        self.flag = False
        if not self.count:
            self.count = 1
            self.linear = nn.Linear(2, 1)
            self.flag = True

    def forward(self, noise_level):
        step = torch.arange(self.count, dtype=noise_level.dtype,
                            device=noise_level.device) / self.count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        if self.flag:
            encoding = self.linear(encoding)
        return encoding


class VSSBlockContext(nn.Module):
    """Conditional-branch block: attention refinement + SS2D residual."""

    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        n_head: int = 8,
        n_embd: int = 12,
        first_block: bool = False,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state)
        self.drop_path = DropPath(drop_path)

        self.unet_attention_block = UNET_AttentionBlock(n_head=n_head, n_embd=n_embd)
        self.first_block = first_block

    def forward(self, context: torch.Tensor):
        if self.first_block:
            x1 = self.unet_attention_block(context.permute(0, 3, 1, 2))
            x1 = x1.permute(0, 2, 3, 1)
        else:
            x1 = context

        return x1 + self.drop_path(self.self_attention(self.ln_1(x1)))


class VSSBlockXT(nn.Module):
    """Noisy-branch block: noise-level residual refinement + SS2D residual."""

    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        n_time: int = 1280,
        first_block: bool = False,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state)
        self.drop_path = DropPath(drop_path)

        self.unet_residual_block = UNET_ResidualBlock(in_channels=hidden_dim, out_channels=hidden_dim, n_time=n_time)
        self.first_block = first_block

    def forward(self, input: torch.Tensor, time: torch.Tensor):
        if self.first_block:
            x1 = self.unet_residual_block(input.permute(0, 3, 1, 2), time)
            x1 = x1.permute(0, 2, 3, 1)
        else:
            x1 = input

        return x1 + self.drop_path(self.self_attention(self.ln_1(x1)))


class VSSBlockDecoder(nn.Module):
    """Decoder block: attention refinement + SS2D residual."""

    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        n_head: int = 8,
        n_embd: int = 12,
        first_block: bool = False,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state)
        self.drop_path = DropPath(drop_path)

        self.unet_attention_block = UNET_AttentionBlock(n_head=n_head, n_embd=n_embd)
        self.first_block = first_block

    def forward(self, input: torch.Tensor):
        if self.first_block:
            x1 = self.unet_attention_block(input.permute(0, 3, 1, 2))
            x1 = x1.permute(0, 2, 3, 1)
        else:
            x1 = input

        return x1 + self.drop_path(self.self_attention(self.ln_1(x1)))


class VSSLayerContext(nn.Module):
    """One conditional-encoder stage followed by an optional merge."""

    def __init__(
        self,
        dim,
        depth,
        d_state=16,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlockContext(
                hidden_dim=dim,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                n_head=8,
                n_embd=dim // 8,
                first_block=(i == 0),
            )
            for i in range(depth)])

        def _init_weights(module: nn.Module):
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    p = p.clone().detach_()
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))

        self.apply(_init_weights)

        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, context):
        for blk in self.blocks:
            context = blk(context)
        if self.downsample is not None:
            context = self.downsample(context)
        return context


class VSSLayerXT(nn.Module):
    """One noisy-branch encoder stage followed by an optional merge."""

    def __init__(
        self,
        dim,
        depth,
        d_state=16,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlockXT(
                hidden_dim=dim,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                n_time=1280,
                first_block=(i == 0),
            )
            for i in range(depth)])

        def _init_weights(module: nn.Module):
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    p = p.clone().detach_()
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))

        self.apply(_init_weights)

        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x, time):
        for blk in self.blocks:
            x = blk(x, time)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class VSSLayerDecoder(nn.Module):
    """One decoder stage: optional expansion followed by ``depth`` blocks."""

    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        upsample=None,
        use_checkpoint=False,
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlockDecoder(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                n_head=8,
                n_embd=dim // 8,
                first_block=(i == 0),
            )
            for i in range(depth)])

        def _init_weights(module: nn.Module):
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    p = p.clone().detach_()
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))

        self.apply(_init_weights)

        self.upsample = upsample(dim=dim, norm_layer=norm_layer) if upsample is not None else None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class VSSM(nn.Module):
    """Time-conditioned cross-modal encoder / decoder.

    Args:
        in_chans: Channels of the images of both modalities.
        num_classes: Output channels (equal to ``in_chans``).
        depths: Depth of each encoder / decoder stage.
        dims: Channel width of each encoder stage.
        dims_decoder: Channel width of each decoder stage.
    """

    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 depths=(2, 2, 2), depths_decoder=(2, 2, 2),
                 dims=(64, 128, 256), dims_decoder=(256, 128, 64), d_state=16, drop_rate=0.1,
                 attn_drop_rate=0.0, drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        # The two modalities have the same number of channels (RGB and SAR are
        # both three-channel), so a single embedding is shared by both branches.
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans,
                                        embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers_context = nn.ModuleList()
        self.layers_x_t = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers_context.append(VSSLayerContext(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            ))
            self.layers_x_t.append(VSSLayerXT(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            ))

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers_up.append(VSSLayerDecoder(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            ))

        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4, num_classes, 1)
        self.apply(self._init_weights)

        self.shallow_fusion1_64 = nn.Conv2d(128, 64, 1, 1, 0)
        self.shallow_fusion2_64 = nn.Conv2d(128, 64, 1, 1, 0)
        self.shallow_fusion1_128 = nn.Conv2d(256, 128, 1, 1, 0)
        self.shallow_fusion2_128 = nn.Conv2d(256, 128, 1, 1, 0)
        self.shallow_fusion1_256 = nn.Conv2d(512, 256, 1, 1, 0)
        self.shallow_fusion2_256 = nn.Conv2d(512, 256, 1, 1, 0)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, time, context):
        """Encode both branches and return the fused features of the last scale.

        Also returns the multi-scale features of the conditional encoder, which
        are the ones aligned by the domain adaptation loss.
        """
        skip_list = []
        context_features = []

        x = self.pos_drop(self.patch_embed(x))
        context = self.pos_drop(self.patch_embed(context))

        for i, (layer_context, layer_x_t) in enumerate(zip(self.layers_context, self.layers_x_t)):
            if i == 0:
                context, x = context.permute(0, 3, 1, 2), x.permute(0, 3, 1, 2)
                combined = self.shallow_fusion1_64(torch.concat([context, x], dim=1)) + context
                combined = self.shallow_fusion2_64(torch.concat([combined, x], dim=1)) + combined
                combined = combined.permute(0, 2, 3, 1)
                context, x = context.permute(0, 2, 3, 1), x.permute(0, 2, 3, 1)
                skip_list.append(combined)
                context_features.append(context)

            context = layer_context(context)
            x = layer_x_t(x, time)
            base_filter = x.shape[-1]

            context, x = context.permute(0, 3, 1, 2), x.permute(0, 3, 1, 2)

            if base_filter == 128:
                combined = self.shallow_fusion1_128(torch.concat([context, x], dim=1)) + context
                combined = self.shallow_fusion2_128(torch.concat([combined, x], dim=1)) + combined
            elif base_filter == 256:
                combined = self.shallow_fusion1_256(torch.concat([context, x], dim=1)) + context
                combined = self.shallow_fusion2_256(torch.concat([combined, x], dim=1)) + combined
            else:
                raise ValueError(f"Unexpected encoder width {base_filter} at stage {i}")

            combined = combined.permute(0, 2, 3, 1)
            context, x = context.permute(0, 2, 3, 1), x.permute(0, 2, 3, 1)
            if i != self.num_layers - 1:
                skip_list.append(combined)
                context_features.append(context)

        return combined, skip_list, context_features

    def forward_features_up(self, x, skip_list):
        layer_outputs = []
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skip_list[-inx])
            layer_outputs.append(x)
        return x, layer_outputs

    def forward_final(self, x):
        x = self.final_up(x)
        x = x.permute(0, 3, 1, 2)
        return self.final_conv(x)

    def forward(self, x, context, time):
        x, skip_list, context_features = self.forward_features(x, time, context)
        x, layer_outputs = self.forward_features_up(x, skip_list)
        x = self.forward_final(x)
        return x, layer_outputs, context_features


class DADiffBackbone(nn.Module):
    """Top-level module of the conditional diffusion model.

    Forward inputs:
        ``x_xnoisy``: ``(context, x_t)`` -- the conditioning image and the noisy
        image that is being generated.
        ``timesteps``: continuous ``sqrt(alpha_bar)`` values of shape ``(B, 1)``.

    Forward outputs:
        ``(x_recon, context_features)``, where ``x_recon`` is the direct estimate
        of the clean target image and ``context_features`` holds the three
        scales of conditional-encoder features used by the alignment loss.
    """

    def __init__(self,
                 in_chans=3,
                 depths=(2, 2, 2),
                 depths_decoder=(2, 2, 2),
                 attn_drop_rate=0.2,
                 drop_path_rate=0.2,
                 ):
        super().__init__()

        self.num_classes = in_chans

        out_channels = 320
        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(out_channels),
            nn.Linear(out_channels, out_channels * 4),
            nn.GELU(),
            nn.Linear(out_channels * 4, out_channels)
        )

        self.vmunet = VSSM(in_chans=in_chans,
                           num_classes=self.num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                           attn_drop_rate=attn_drop_rate,
                           )

        self.time_embedding = TimeEmbedding(n_embd=out_channels)

    def forward(self, x_xnoisy, timesteps):
        context, x = x_xnoisy

        t = self.noise_level_mlp(timesteps)
        time_emb = self.time_embedding(t)

        logits, _layer_outputs, context_features = self.vmunet(x, context, time_emb)
        return logits, context_features
