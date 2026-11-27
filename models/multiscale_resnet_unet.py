"""Lightweight multi-scale residual adapters for shallow ResNet18 U-Net skips.

The existing encoder and decoder keep their original state-dict names. New
parameters live exclusively below ``skip_adapters`` so that a caller can load a
baseline checkpoint while explicitly checking the allowed missing keys.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn

from .resnet_unet import ResNetUNet


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _dilation_values(dilations: Sequence[int]) -> tuple[int, ...]:
    if isinstance(dilations, (str, bytes)) or not isinstance(dilations, Sequence):
        raise TypeError("dilations must be a nonempty sequence of positive integers")
    values = tuple(dilations)
    if not values:
        raise ValueError("dilations must contain at least one positive integer")
    return tuple(_positive_integer(value, "dilation") for value in values)


class LightweightMultiScaleResidual(nn.Module):
    """Add local multi-scale context without changing feature shape.

    By default the residual path compresses ``C`` channels to ``max(C // 4, 1)``,
    applies parallel depthwise 3x3 convolutions with dilation 1 and 2, concatenates
    the branch outputs, and restores ``C`` channels with a 1x1 convolution.
    Padding preserves both spatial dimensions. No BatchNorm or spatial
    resampling is introduced.

    The final fusion weight and bias are initialized to zero, making the module
    an exact identity for finite inputs at initialization. On the first backward
    pass only the fusion convolution can receive nonzero *data-loss* gradients;
    the earlier convolutions receive zero gradients through its zero weights.
    Once an optimizer step changes the fusion weights, gradients can reach those
    earlier convolutions. Optimizer weight decay can still update parameters
    with zero data-loss gradients, so this is not a parameter-freezing mechanism.

    Args:
        channels: Positive integer input/output channel count (booleans rejected).
        reduction: Positive integer channel reduction factor, default 4.
        dilations: Nonempty sequence of positive integer branch dilation rates.
            Defaults to ``(1, 2)``; booleans are rejected.

    Shape:
        Input and output are both ``[N, channels, H, W]``.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        dilations: Sequence[int] = (1, 2),
    ):
        super().__init__()
        self.channels = _positive_integer(channels, "channels")
        self.reduction = _positive_integer(reduction, "reduction")
        self.dilations = _dilation_values(dilations)
        self.hidden_channels = max(channels // self.reduction, 1)

        self.reduce = nn.Sequential(
            nn.Conv2d(channels, self.hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.hidden_channels,
                        self.hidden_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=self.hidden_channels,
                    ),
                    nn.ReLU(inplace=True),
                )
                for dilation in self.dilations
            ]
        )
        self.fuse = nn.Conv2d(
            len(self.dilations) * self.hidden_channels, channels, kernel_size=1
        )
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        branches = torch.cat([branch(reduced) for branch in self.branches], dim=1)
        return x + self.fuse(branches)


class MultiScaleResNetUNet(ResNetUNet):
    """ResNet18 U-Net with optional adapters on its shallow decoder skips.

    ``e0`` is the stem feature at approximately half input resolution; ``e1`` is
    the layer1 feature at approximately quarter resolution. Only the selected
    *decoder skip copies* are adapted. All subsequent encoder stages consume
    the unmodified encoder features, retaining the pretrained encoder path.

    All base parameters and buffers retain their original names. Adapters are
    registered outside ``encoder`` and remain trainable when the encoder is
    frozen. The inherited ``train`` method continues to keep a frozen encoder's
    BatchNorm running statistics fixed. No new BatchNorm layers are added.

    Args:
        pretrained: Forwarded to :class:`ResNetUNet`.
        freeze_encoder: Forwarded to :class:`ResNetUNet`.
        stages: Nonempty sequence containing ``"e0"``, ``"e1"``, or both,
            without duplicates. A bare string is rejected. Defaults to ``e0``.
        reduction: Channel reduction factor shared by the skip adapters.
        dilations: Branch dilation rates shared by the skip adapters.

    Output shape follows the base model, including odd spatial dimensions:
    ``[N, 3, H, W] -> [N, 1, H, W]`` for inputs supported by the base encoder.
    A baseline checkpoint must be loaded by the caller with an explicit check
    that only ``skip_adapters.*`` keys are missing; do not silently ignore
    missing or unexpected base-model keys.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_encoder: bool = True,
        stages: Sequence[str] = ("e0",),
        reduction: int = 4,
        dilations: Sequence[int] = (1, 2),
    ):
        if isinstance(stages, (str, bytes)) or not isinstance(stages, Sequence):
            raise TypeError("stages must be a nonempty sequence of 'e0'/'e1' names")
        stages = tuple(stages)
        if not stages:
            raise ValueError("stages must contain at least one of 'e0' or 'e1'")
        if any(not isinstance(stage, str) for stage in stages):
            raise TypeError("every stage name must be a string")
        if any(stage not in ("e0", "e1") for stage in stages):
            raise ValueError("only shallow skip stages 'e0' and 'e1' are supported")
        if len(set(stages)) != len(stages):
            raise ValueError("stages must not contain duplicate names")
        # Validate before creating a possibly pretrained base model.
        reduction = _positive_integer(reduction, "reduction")
        dilations = _dilation_values(dilations)

        super().__init__(pretrained=pretrained, freeze_encoder=freeze_encoder)
        self.stages = stages
        self.skip_adapters = nn.ModuleDict(
            {
                stage: LightweightMultiScaleResidual(
                    64, reduction=reduction, dilations=dilations
                )
                for stage in stages
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = x
        normalized = (x - self.mean) / self.std

        # Keep the complete pretrained encoder path identical to the base model.
        e0 = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(normalized)))
        e1 = self.encoder.layer1(self.encoder.maxpool(e0))
        e2 = self.encoder.layer2(e1)
        e3 = self.encoder.layer3(e2)
        e4 = self.encoder.layer4(e3)

        # Adapt only decoder skips: e1 above must never consume an adapted e0.
        skip0 = self.skip_adapters["e0"](e0) if "e0" in self.skip_adapters else e0
        skip1 = self.skip_adapters["e1"](e1) if "e1" in self.skip_adapters else e1

        d4 = self.dec4(self.up_and_concat(e4, e3))
        d3 = self.dec3(self.up_and_concat(d4, e2))
        d2 = self.dec2(self.up_and_concat(d3, skip1))
        d1 = self.dec1(self.up_and_concat(d2, skip0))
        d0 = self.dec0(self.up_and_concat(d1, original))
        return self.output(d0)
