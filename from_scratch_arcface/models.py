import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNPReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, groups: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.depthwise = ConvBNPReLU(in_channels, in_channels, 3, stride, 1, groups=in_channels)
        self.pointwise = ConvBNPReLU(in_channels, out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class MobileBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, expansion: int):
        super().__init__()
        hidden_channels = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels
        self.layers = nn.Sequential(
            ConvBNPReLU(in_channels, hidden_channels, 1, 1, 0),
            ConvBNPReLU(hidden_channels, hidden_channels, 3, stride, 1, groups=hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        if self.use_residual:
            out = out + x
        return out


class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size: int = 512):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNPReLU(3, 64, 3, 2, 1),
            DepthwiseSeparable(64, 64, 1),
            MobileBottleneck(64, 64, 2, 2),
            MobileBottleneck(64, 64, 1, 2),
            MobileBottleneck(64, 64, 1, 2),
            MobileBottleneck(64, 128, 2, 4),
            MobileBottleneck(128, 128, 1, 2),
            MobileBottleneck(128, 128, 1, 2),
            MobileBottleneck(128, 128, 1, 2),
            MobileBottleneck(128, 128, 1, 2),
            MobileBottleneck(128, 128, 1, 2),
            MobileBottleneck(128, 128, 2, 4),
            MobileBottleneck(128, 128, 1, 2),
            MobileBottleneck(128, 128, 1, 2),
            ConvBNPReLU(128, 512, 1, 1, 0),
        )
        self.output = nn.Sequential(
            nn.Conv2d(512, 512, 7, 1, 0, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.Flatten(),
            nn.Linear(512, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.output(x)
        return F.normalize(x)


class IRBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(x) + self.shortcut(x)


class IRResNet(nn.Module):
    def __init__(self, layers: list[int], embedding_size: int = 512, dropout: float = 0.4):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        channels = [64, 128, 256, 512]
        blocks = []
        in_channels = 64
        for stage_idx, (out_channels, num_blocks) in enumerate(zip(channels, layers)):
            for block_idx in range(num_blocks):
                stride = 2 if block_idx == 0 and stage_idx > 0 else 1
                blocks.append(IRBlock(in_channels, out_channels, stride))
                in_channels = out_channels
        self.body = nn.Sequential(*blocks)
        self.output = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(512 * 14 * 14, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output(x)
        return F.normalize(x)


def build_backbone(name: str, embedding_size: int = 512) -> nn.Module:
    name = name.lower()
    if name == "mobilefacenet":
        return MobileFaceNet(embedding_size=embedding_size)
    if name == "iresnet18":
        return IRResNet([2, 2, 2, 2], embedding_size=embedding_size)
    if name == "iresnet34":
        return IRResNet([3, 4, 6, 3], embedding_size=embedding_size)
    raise ValueError(f"Unknown backbone: {name}")
