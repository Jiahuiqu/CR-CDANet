"""AT-Mamba encoder / decoder for multimodal change detection.

The network follows a dual-encoder single-decoder design.  Each modality is
encoded by its own :class:`VSSM`, which processes the bi-temporal pair of that
modality through a *conditional* branch (the reference image) and an *x_t*
branch (the image to be compared), fusing the two at every scale.  The decoder
(:class:`VSSM_fusion`) first merges the multi-scale features of the two
modalities with 1x1 convolutions and then reconstructs a dense logit map.

Feature maps of both encoders are returned as well so that the cross-domain
alignment losses (MMD / CORAL) can be computed at every scale.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_

from .blocks import (Final_PatchExpand2D, PatchEmbed2D, PatchMerging2D, PatchExpand2D,
                     SS2D, UNET_AttentionBlock)


class VSSBlock(nn.Module):
    """Visual state-space block: optional attention refinement + SS2D residual.

    The attention refinement is applied only by the first block of a stage
    (``first_block=True``).
    """

    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
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

    def forward(self, x: torch.Tensor):
        if self.first_block:
            x1 = self.unet_attention_block(x.permute(0, 3, 1, 2))
            x1 = x1.permute(0, 2, 3, 1)
        else:
            x1 = x

        return x1 + self.drop_path(self.self_attention(self.ln_1(x1)))


class VSSLayer(nn.Module):
    """One encoder stage: ``depth`` VSS blocks followed by an optional merge."""

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
            VSSBlock(
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

        # Kept for fidelity with the reference implementation; the effective
        # initialisation of these projections is the one performed by VSSM.
        def _init_weights(module: nn.Module):
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    p = p.clone().detach_()
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))

        self.apply(_init_weights)

        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class VSSLayerDecoder(nn.Module):
    """One decoder stage: optional expansion followed by ``depth`` VSS blocks."""

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
            VSSBlock(
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
    """Single-modality dual-branch encoder.

    ``x`` and ``context`` are the two time steps of one modality.  They are
    encoded in parallel and fused at every scale by the ``shallow_fusion*``
    projections.  The final scale is returned together with the multi-scale
    encoder features of both branches.

    Args:
        patch_size: Patch embedding kernel size (4 by default).
        in_chans: Number of channels of the modality.
        num_classes: Output channels of the (unused) final head.
        depths: Depth of each encoder stage.
        depths_decoder: Depth of each decoder stage.
        dims: Channel width of each encoder stage.
        dims_decoder: Channel width of each decoder stage.
    """

    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=(2, 2, 2), depths_decoder=(2, 2, 2),
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

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        # Encoder: a conditional branch and an x_t branch per stage.
        self.layers_context = nn.ModuleList()
        self.layers_x_t = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers_context.append(VSSLayer(
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

            self.layers_x_t.append(VSSLayer(
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

        # Decoder.
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

        # 1x1 projections used to fuse the two branches at each scale.
        self.shallow_fusion1_32 = nn.Conv2d(64, 32, 1, 1, 0)
        self.shallow_fusion2_32 = nn.Conv2d(64, 32, 1, 1, 0)
        self.shallow_fusion1_64 = nn.Conv2d(128, 64, 1, 1, 0)
        self.shallow_fusion2_64 = nn.Conv2d(128, 64, 1, 1, 0)
        self.shallow_fusion1_128 = nn.Conv2d(256, 128, 1, 1, 0)
        self.shallow_fusion2_128 = nn.Conv2d(256, 128, 1, 1, 0)
        self.shallow_fusion1_256 = nn.Conv2d(512, 256, 1, 1, 0)
        self.shallow_fusion2_256 = nn.Conv2d(512, 256, 1, 1, 0)
        self.shallow_fusion1_512 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.shallow_fusion2_512 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.shallow_fusion1_1024 = nn.Conv2d(2048, 1024, 1, 1, 0)
        self.shallow_fusion2_1024 = nn.Conv2d(2048, 1024, 1, 1, 0)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, context):
        """Encode the two time steps and return the fused features per scale."""
        skip_list = []
        x = self.pos_drop(self.patch_embed(x))
        context = self.pos_drop(self.patch_embed(context))

        encoder_outputs_t1 = []
        encoder_outputs_t2 = []

        for i, (layer_context, layer_x_t) in enumerate(zip(self.layers_context, self.layers_x_t)):
            if i == 0:
                context, x = context.permute(0, 3, 1, 2), x.permute(0, 3, 1, 2)
                encoder_outputs_t1.append(context)
                encoder_outputs_t2.append(x)
                combined = self.shallow_fusion1_64(torch.concat([context, x], dim=1))
                combined = self.shallow_fusion2_64(torch.concat([combined, x], dim=1)) + combined
                combined = combined.permute(0, 2, 3, 1)
                context, x = context.permute(0, 2, 3, 1), x.permute(0, 2, 3, 1)
                skip_list.append(combined)

            context = layer_context(context)
            x = layer_x_t(x)
            base_filter = x.shape[-1]

            context, x = context.permute(0, 3, 1, 2), x.permute(0, 3, 1, 2)

            if base_filter == 128:
                if i != 2:
                    encoder_outputs_t1.append(context)
                    encoder_outputs_t2.append(x)
                combined = self.shallow_fusion1_128(torch.concat([context, x], dim=1))
                combined = self.shallow_fusion2_128(torch.concat([combined, x], dim=1)) + combined
            elif base_filter == 256:
                if i != 2:
                    encoder_outputs_t1.append(context)
                    encoder_outputs_t2.append(x)
                combined = self.shallow_fusion1_256(torch.concat([context, x], dim=1))
                combined = self.shallow_fusion2_256(torch.concat([combined, x], dim=1)) + combined
            else:
                raise ValueError(f"Unexpected encoder width {base_filter} at stage {i}")

            combined = combined.permute(0, 2, 3, 1)
            context, x = context.permute(0, 2, 3, 1), x.permute(0, 2, 3, 1)
            if i != 2:
                skip_list.append(combined)

        return combined, skip_list, encoder_outputs_t1, encoder_outputs_t2

    def forward_features_up(self, x, skip_list):
        layer_outputs = []
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skip_list[-inx])
            layer_outputs.append(x)
        return layer_outputs

    def forward_final(self, x):
        x = self.final_up(x)
        x = x.permute(0, 3, 1, 2)
        return self.final_conv(x)

    def forward(self, x, context):
        x, skip_list, encoder_outputs_t1, encoder_outputs_t2 = self.forward_features(x, context)
        layer_outputs = self.forward_features_up(x, skip_list)
        return layer_outputs, encoder_outputs_t1, encoder_outputs_t2


class VSSM_fusion(nn.Module):
    """Cross-modal decoder.

    The multi-scale decoder features of the two modalities are combined with
    ``1x1`` convolutions (a two-layer residual projection per scale) before
    being passed through the corresponding decoder stage.
    """

    def __init__(self, patch_size=4, in_chans=3, num_classes=2, depths=(2, 2, 2), depths_decoder=(2, 2, 2),
                 dims=(64, 128, 256), dims_decoder=(128, 64, 32), d_state=16, drop_rate=0.1,
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

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

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
                upsample=PatchExpand2D,
                use_checkpoint=use_checkpoint,
            ))

        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=2, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 2, num_classes, 1)
        self.apply(self._init_weights)

        self.shallow_fusion1_32 = nn.Conv2d(64, 32, 1, 1, 0)
        self.shallow_fusion2_32 = nn.Conv2d(64, 32, 1, 1, 0)
        self.shallow_fusion1_64 = nn.Conv2d(128, 64, 1, 1, 0)
        self.shallow_fusion2_64 = nn.Conv2d(128, 64, 1, 1, 0)
        self.shallow_fusion1_128 = nn.Conv2d(256, 128, 1, 1, 0)
        self.shallow_fusion2_128 = nn.Conv2d(256, 128, 1, 1, 0)
        self.shallow_fusion1_256 = nn.Conv2d(512, 256, 1, 1, 0)
        self.shallow_fusion2_256 = nn.Conv2d(512, 256, 1, 1, 0)
        self.shallow_fusion1_512 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.shallow_fusion2_512 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.shallow_fusion1_1024 = nn.Conv2d(2048, 1024, 1, 1, 0)
        self.shallow_fusion2_1024 = nn.Conv2d(2048, 1024, 1, 1, 0)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features_up(self, layer_outputs_1, layer_outputs_2):
        x = None
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                fused = torch.concat([layer_outputs_1[0].permute(0, 3, 1, 2),
                                      layer_outputs_2[0].permute(0, 3, 1, 2)], dim=1)
                combined = self.shallow_fusion1_256(fused)
                combined = self.shallow_fusion2_256(fused) + combined
                combined = combined.permute(0, 2, 3, 1)
                x = layer_up(combined)
            elif inx == 1:
                fused = torch.concat([layer_outputs_1[1].permute(0, 3, 1, 2),
                                      layer_outputs_2[1].permute(0, 3, 1, 2)], dim=1)
                combined = self.shallow_fusion1_128(fused)
                combined = self.shallow_fusion2_128(fused) + combined
                combined = combined.permute(0, 2, 3, 1)
                x = layer_up(x + combined)
            elif inx == 2:
                fused = torch.concat([layer_outputs_1[2].permute(0, 3, 1, 2),
                                      layer_outputs_2[2].permute(0, 3, 1, 2)], dim=1)
                combined = self.shallow_fusion1_64(fused)
                combined = self.shallow_fusion2_64(fused) + combined
                combined = combined.permute(0, 2, 3, 1)
                x = layer_up(x + combined)
        return x

    def forward_final(self, x):
        x = self.final_up(x)
        x = x.permute(0, 3, 1, 2)
        return self.final_conv(x)

    def forward(self, layer_outputs_1, layer_outputs_2):
        layer_outputs = self.forward_features_up(layer_outputs_1, layer_outputs_2)
        return self.forward_final(layer_outputs)


class VMUNet(nn.Module):
    """AT-Mamba Encoder & Decoder for multimodal change detection.

    Args:
        in_chans_mod1: Input channels of modality 1 (e.g. RGB).
        in_chans_mod2: Input channels of modality 2 (e.g. SAR).
        depths: Encoder depth per stage.
        depths_decoder: Decoder depth per stage.
        fusion_decoder: Decoder width per stage of the cross-modal decoder.
        attn_drop_rate: Dropout rate inside the attention blocks.
        drop_path_rate: Stochastic depth rate.
        num_classes: Number of change classes in the output logit map.

    Forward input:
        ``modal1`` and ``modal2`` are ``(t1, t2)`` pairs, each of shape
        ``(B, C, H, W)``.  For a ``P x P`` input patch the logit map has shape
        ``(B, num_classes, P - 1, P - 1)``.

    Forward output:
        ``(logits, enc_m1_ctx, enc_m1_x, enc_m2_ctx, enc_m2_x)`` where the two
        encoder lists of a modality hold the multi-scale features of its
        conditional branch and of its ``x`` branch respectively.
    """

    def __init__(self,
                 in_chans_mod1=3,
                 in_chans_mod2=224,
                 depths=(2, 2, 2),
                 depths_decoder=(2, 2, 2),
                 fusion_decoder=(128, 64, 32),
                 attn_drop_rate=0.2,
                 drop_path_rate=0.2,
                 num_classes=2,
                 ):
        super().__init__()

        self.num_classes = num_classes

        self.vmunet_m1 = VSSM(in_chans=in_chans_mod1,
                              num_classes=self.num_classes,
                              depths=depths,
                              depths_decoder=depths_decoder,
                              drop_path_rate=drop_path_rate,
                              attn_drop_rate=attn_drop_rate,
                              )
        self.vmunet_m2 = VSSM(in_chans=in_chans_mod2,
                              num_classes=self.num_classes,
                              depths=depths,
                              depths_decoder=depths_decoder,
                              drop_path_rate=drop_path_rate,
                              attn_drop_rate=attn_drop_rate,
                              )

        self.fusion = VSSM_fusion(num_classes=num_classes,
                                  depths_decoder=[2, 2, 2],
                                  dims_decoder=fusion_decoder
                                  )

    def forward(self, modal1, modal2):
        modal1_t1, modal1_t2 = modal1
        modal2_t1, modal2_t2 = modal2

        layer_outputs_1, enc_m1_ctx, enc_m1_x = self.vmunet_m1(modal1_t1, modal1_t2)
        layer_outputs_2, enc_m2_ctx, enc_m2_x = self.vmunet_m2(modal2_t1, modal2_t2)
        logits = self.fusion(layer_outputs_1, layer_outputs_2)

        if self.num_classes == 1:
            return torch.sigmoid(logits), enc_m1_ctx, enc_m1_x, enc_m2_ctx, enc_m2_x
        return logits, enc_m1_ctx, enc_m1_x, enc_m2_ctx, enc_m2_x
