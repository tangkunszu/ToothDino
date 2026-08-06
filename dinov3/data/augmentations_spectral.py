# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with the
# terms of the DINOv3 License Agreement.

"""
Frequency-domain augmentation for dental X-ray images.

Motivation:
  X-ray image formation follows the Beer-Lambert law. Different exposure presets
  (kVp, mAs) shift spectral energy across spatial-frequency bands. Mid-frequency
  components are most susceptible to scatter-induced blur, while the DC component
  (low freq) encodes overall density and high-frequency components encode bone
  edges and fine dental structures.

  By randomly attenuating a mid-frequency band in the 2-D FFT amplitude spectrum
  we simulate across-device and across-exposure variability without touching
  phase (which encodes structural position), making the augmentation
  anatomically safe: the spatial arrangement of teeth is preserved.

Usage (tensor, float32 in [0, 1], after ToImage/ToDtype):
    aug = FrequencySpectrumAugmentation(p=0.5)
    img_aug = aug(img)   # img: Tensor [C, H, W]
"""

import logging
import math
import random

import torch
import torch.nn as nn

logger = logging.getLogger("dinov3")


class FrequencySpectrumAugmentation(nn.Module):
    """
    Random mid-frequency band attenuation in the 2-D FFT amplitude spectrum.

    Args:
        p (float):
            Probability of applying the augmentation per image. Default: 0.5.
        mask_low_range (tuple[float, float]):
            Uniform range from which the normalized low cut-off of the
            attenuated band is sampled (0 = DC, 1 = Nyquist). Default: (0.08, 0.40).
        mask_width_range (tuple[float, float]):
            Uniform range from which the bandwidth of the attenuated band is
            sampled (expressed as a fraction of Nyquist). Default: (0.10, 0.30).
        attenuation_range (tuple[float, float]):
            Multiplicative attenuation applied to amplitude in the selected
            band. 1.0 = no-op, 0.0 = full suppression. Default: (0.30, 0.85).
        per_channel (bool):
            If True, sample a different band per channel (useful for true RGB;
            for grayscale-converted X-rays setting this to False is fine).
            Default: False.
    """

    def __init__(
        self,
        p: float = 0.5,
        mask_low_range: tuple = (0.08, 0.40),
        mask_width_range: tuple = (0.10, 0.30),
        attenuation_range: tuple = (0.30, 0.85),
        per_channel: bool = False,
    ):
        super().__init__()
        self.p = p
        self.mask_low_range = mask_low_range
        self.mask_width_range = mask_width_range
        self.attenuation_range = attenuation_range
        self.per_channel = per_channel

    def _build_radial_map(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Return a [H, W] tensor of normalized radial distances from the FFT centre."""
        cy, cx = H // 2, W // 2
        ys = torch.arange(H, device=device, dtype=torch.float32) - cy
        xs = torch.arange(W, device=device, dtype=torch.float32) - cx
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        r = torch.sqrt(yy ** 2 + xx ** 2)
        # Normalise so that the corner of the spectrum = 1.0
        r_norm = r / (math.sqrt((H / 2) ** 2 + (W / 2) ** 2) + 1e-6)
        return r_norm  # [H, W]

    def _attenuate_channel(
        self,
        channel: torch.Tensor,
        r_norm: torch.Tensor,
        low: float,
        high: float,
        atten: float,
    ) -> torch.Tensor:
        """Apply band attenuation to a single [H, W] channel (float32, [0,1])."""
        # 2-D FFT and centre shift
        fft = torch.fft.fft2(channel)
        fft_shifted = torch.fft.fftshift(fft)

        # Build binary band mask
        band_mask = (r_norm >= low) & (r_norm <= high)  # [H, W]

        # Attenuate amplitude, preserve phase
        amplitude = fft_shifted.abs()
        phase = fft_shifted.angle()
        amplitude = amplitude.masked_fill(band_mask, amplitude.masked_fill(~band_mask, 0.0).max() * atten)
        # Cleaner: replace band amplitude with attenuation factor * original
        amplitude2 = fft_shifted.abs().clone()
        amplitude2[band_mask] = amplitude2[band_mask] * atten
        fft_modified = amplitude2 * torch.exp(1j * phase)

        # Inverse FFT
        fft_unshifted = torch.fft.ifftshift(fft_modified)
        img_out = torch.fft.ifft2(fft_unshifted).real
        return img_out.clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float tensor of shape [C, H, W] in range [0, 1].
        Returns:
            Augmented tensor of the same shape.
        """
        if random.random() > self.p:
            return x

        C, H, W = x.shape
        r_norm = self._build_radial_map(H, W, device=x.device)

        result = x.clone()

        if self.per_channel:
            for c in range(C):
                low = random.uniform(*self.mask_low_range)
                high = float(min(low + random.uniform(*self.mask_width_range), 1.0))
                atten = random.uniform(*self.attenuation_range)
                result[c] = self._attenuate_channel(x[c].float(), r_norm, low, high, atten)
        else:
            # Share band params across channels (appropriate for grayscale X-rays
            # replicated to 3 channels)
            low = random.uniform(*self.mask_low_range)
            high = float(min(low + random.uniform(*self.mask_width_range), 1.0))
            atten = random.uniform(*self.attenuation_range)
            for c in range(C):
                result[c] = self._attenuate_channel(x[c].float(), r_norm, low, high, atten)

        return result.to(x.dtype)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"p={self.p}, "
            f"mask_low_range={self.mask_low_range}, "
            f"mask_width_range={self.mask_width_range}, "
            f"attenuation_range={self.attenuation_range}, "
            f"per_channel={self.per_channel})"
        )
