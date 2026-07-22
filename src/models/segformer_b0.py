"""Pinned, dependency-free SegFormer-B0 reference implementation.

This module implements the MiT-B0 encoder and the lightweight all-MLP
decoder described by the SegFormer paper. It is intentionally kept local so
the graph contract does not depend on a mutable image-model library default.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


SEGFORMER_IMPLEMENTATION_ID = "mit-b0-all-mlp-v1"
SEGFORMER_PRETRAINED_ARTIFACT_ID = "mit-b0-imagenet-1k-v1"


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must be in [0, 1)")
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * random_tensor.floor() / keep_prob


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, stride: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        x = self.proj(x)
        height, width = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), height, width


class MixFFN(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_dim,
        )
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        x = self.fc1(x)
        batch, tokens, channels = x.shape
        spatial = x.transpose(1, 2).reshape(batch, channels, height, width)
        spatial = self.dwconv(spatial)
        x = spatial.flatten(2).transpose(1, 2)
        x = self.activation(x)
        x = self.dropout(x)
        return self.dropout(self.fc2(x))


class EfficientSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, sr_ratio: int, drop: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key_value = nn.Linear(embed_dim, embed_dim * 2)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        batch, tokens, channels = x.shape
        query = self.query(x).reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        source = x
        if self.sr_ratio > 1:
            source = x.transpose(1, 2).reshape(batch, channels, height, width)
            source = self.sr(source).flatten(2).transpose(1, 2)
            source = self.sr_norm(source)
        key_value = self.key_value(source)
        key_value = key_value.reshape(batch, -1, 2, self.num_heads, self.head_dim)
        key_value = key_value.permute(2, 0, 3, 1, 4)
        key, value = key_value[0], key_value[1]
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.dropout(attention)
        output = attention @ value
        output = output.transpose(1, 2).reshape(batch, tokens, channels)
        return self.dropout(self.projection(output))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        sr_ratio: int,
        mlp_ratio: int,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = EfficientSelfAttention(embed_dim, num_heads, sr_ratio)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MixFFN(embed_dim, embed_dim * mlp_ratio)

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        x = x + self.drop_path(self.attention(self.norm1(x), height, width))
        x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
        return x


class SegFormerB0(nn.Module):
    """MiT-B0 encoder with a two-class cloud head and flexible training spatial size."""

    def __init__(self, num_classes: int = 2, in_channels: int = 3, decoder_dim: int = 256) -> None:
        super().__init__()
        if num_classes != 2 or in_channels != 3:
            raise ValueError("the released MVP is RGB with exactly two classes")
        embed_dims = (32, 64, 160, 256)
        depths = (2, 2, 2, 2)
        heads = (1, 2, 5, 8)
        sr_ratios = (8, 4, 2, 1)
        self.patch_embeds = nn.ModuleList()
        self.blocks = nn.ModuleList()
        previous_dim = in_channels
        block_count = sum(depths)
        block_index = 0
        for index, (embed_dim, depth, num_heads, sr_ratio) in enumerate(
            zip(embed_dims, depths, heads, sr_ratios)
        ):
            patch_size = 7 if index == 0 else 3
            stride = 4 if index == 0 else 2
            self.patch_embeds.append(OverlapPatchEmbed(previous_dim, embed_dim, patch_size, stride))
            stage_blocks = []
            for _ in range(depth):
                stage_blocks.append(
                    TransformerBlock(
                        embed_dim,
                        num_heads,
                        sr_ratio,
                        mlp_ratio=4,
                        drop_path=0.1 * block_index / max(block_count - 1, 1),
                    )
                )
                block_index += 1
            self.blocks.append(nn.ModuleList(stage_blocks))
            previous_dim = embed_dim

        self.linear_projections = nn.ModuleList(
            [nn.Linear(embed_dim, decoder_dim) for embed_dim in embed_dims]
        )
        self.linear_fuse = nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1)
        self.fuse_norm = nn.BatchNorm2d(decoder_dim)
        self.fuse_activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)
        self.classifier = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: Tensor) -> list[Tensor]:
        features: list[Tensor] = []
        for patch_embed, blocks in zip(self.patch_embeds, self.blocks):
            x, height, width = patch_embed(x)
            for block in blocks:
                x = block(x, height, width)
            feature = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], height, width)
            features.append(feature)
            x = feature
        return features

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"SegFormer-B0 expects NCHW RGB input, got {tuple(x.shape)}")
        features = self.forward_features(x)
        target_height, target_width = features[0].shape[-2:]
        projected: list[Tensor] = []
        for feature, projection in zip(features, self.linear_projections):
            batch, channels, height, width = feature.shape
            tokens = feature.flatten(2).transpose(1, 2)
            tokens = projection(tokens)
            feature_projected = tokens.transpose(1, 2).reshape(batch, -1, height, width)
            if (height, width) != (target_height, target_width):
                feature_projected = F.interpolate(
                    feature_projected,
                    size=(target_height, target_width),
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(feature_projected)
        fused = torch.cat(projected, dim=1)
        fused = self.fuse_activation(self.fuse_norm(self.linear_fuse(fused)))
        return self.classifier(self.dropout(fused))


class SegFormerB0ForCloudSegmentation(SegFormerB0):
    """Explicit application name used by training and runtime factories."""


def get_segformer_b0(*, pretrained: bool = False, num_classes: int = 2, num_channels: int = 3) -> SegFormerB0ForCloudSegmentation:
    if pretrained:
        raise FileNotFoundError(
            "The pinned pretrained artifact is external to this repository; "
            "load it through the release checkpoint manifest."
        )
    return SegFormerB0ForCloudSegmentation(
        num_classes=num_classes,
        in_channels=num_channels,
    )


if __name__ == "__main__":
    model = get_segformer_b0()
    output = model(torch.zeros(1, 3, 256, 256))
    print(tuple(output.shape))
