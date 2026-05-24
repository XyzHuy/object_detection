from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterMaskedDepthwiseConv(nn.Module):
    def __init__(self, channels: int, radius: int):
        super().__init__()
        if radius < 1:
            raise ValueError("radius must be >= 1")

        kernel_size = radius * 2 + 1
        self.radius = radius
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels))

        mask = torch.ones(1, 1, kernel_size, kernel_size)
        mask[:, :, radius, radius] = 0.0
        self.register_buffer("mask", mask)
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.mask
        return F.conv2d(
            x,
            weight,
            self.bias,
            stride=1,
            padding=self.radius,
            groups=x.shape[1],
        )


class LocalContextPredictor(nn.Module):
    def __init__(self, channels: int, radius: int, hidden_channels: int | None = None, use_coords: bool = True):
        super().__init__()
        self.use_coords = use_coords
        in_channels = channels + 2 if use_coords else channels
        hidden_channels = hidden_channels or max(channels, 64)

        self.local = nn.Sequential(
            CenterMaskedDepthwiseConv(in_channels, radius),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_coords:
            x = torch.cat([x, self._coord_channels(x)], dim=1)
        return self.mix(self.local(x))

    @staticmethod
    def _coord_channels(x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        y = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        x_coord = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(y, x_coord, indexing="ij")
        coords = torch.stack((xx, yy), dim=0).unsqueeze(0)
        return coords.expand(b, -1, -1, -1)


class LocalJEPABranch(nn.Module):
    """
    Train-only local JEPA branch for dense YOLO features.

    Each predictor has a strict local receptive field and a zeroed center kernel,
    so the vector at (y, x) must be reconstructed from neighboring cells, not
    copied from the same cell. Targets are the real feature vectors at the same
    coordinates with stop-gradient.
    """

    def __init__(
        self,
        channels=(128, 256, 512),
        radii=(2, 2, 1),
        scale_weights=(1.0, 0.5, 0.25),
        targets_per_image=(64, 32, 16),
        use_coords: bool = True,
    ):
        super().__init__()
        if not (len(channels) == len(radii) == len(scale_weights) == len(targets_per_image)):
            raise ValueError("channels, radii, scale_weights, and targets_per_image must have the same length")

        self.scale_weights = tuple(float(weight) for weight in scale_weights)
        self.targets_per_image = tuple(int(count) for count in targets_per_image)
        self.predictors = nn.ModuleList(
            LocalContextPredictor(channel, radius, use_coords=use_coords)
            for channel, radius in zip(channels, radii)
        )
        self.radii = tuple(int(radius) for radius in radii)

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        total = features[0].new_zeros(())
        losses: dict[str, torch.Tensor] = {}

        for idx, (feature, predictor, radius, weight, num_targets) in enumerate(
            zip(features, self.predictors, self.radii, self.scale_weights, self.targets_per_image),
            start=3,
        ):
            prediction = predictor(feature)
            scale_loss = self._scale_loss(prediction, feature.detach(), radius, num_targets)
            losses[f"p{idx}"] = scale_loss.detach()
            total = total + weight * scale_loss

        losses["loss"] = total
        return losses

    @staticmethod
    def _scale_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        radius: int,
        targets_per_image: int,
    ) -> torch.Tensor:
        b, c, h, w = prediction.shape
        target_count = min(max(int(targets_per_image), 1), h * w)
        positions = LocalJEPABranch._sample_positions(b, h, w, radius, target_count, prediction.device)

        pred_flat = prediction.flatten(2).transpose(1, 2)
        target_flat = target.flatten(2).transpose(1, 2)
        batch_idx = torch.arange(b, device=prediction.device).unsqueeze(1)

        pred_vectors = pred_flat[batch_idx, positions]
        target_vectors = target_flat[batch_idx, positions]
        pred_vectors = F.normalize(pred_vectors, dim=-1)
        target_vectors = F.normalize(target_vectors, dim=-1)
        return (1.0 - F.cosine_similarity(pred_vectors, target_vectors, dim=-1)).mean()

    @staticmethod
    def _sample_positions(
        batch_size: int,
        height: int,
        width: int,
        radius: int,
        target_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        if height > 2 * radius and width > 2 * radius:
            y = torch.arange(radius, height - radius, device=device)
            x = torch.arange(radius, width - radius, device=device)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            candidates = (yy * width + xx).reshape(-1)
        else:
            candidates = torch.arange(height * width, device=device)

        if candidates.numel() <= target_count:
            return candidates.unsqueeze(0).expand(batch_size, -1)

        noise = torch.rand(batch_size, candidates.numel(), device=device)
        sampled = noise.topk(target_count, dim=1, largest=False).indices
        return candidates[sampled]
