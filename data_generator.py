"""
Synthetic Particle Image Generator for Self-Supervised Tracking

Generates pairs of 2D particle images with known ground-truth displacement
fields for training and validating self-supervised particle tracking networks.

Supported flow types:
    - vortex: Lamb-Oseen vortex
    - uniform: Constant translation
    - shear: Linear shear flow
    - channel: Parabolic (Poiseuille) profile
    - multi_vortex: Superposition of multiple vortices
    - random: Random mix of all types
"""

import torch
import numpy as np
from torch.utils.data import Dataset


class SyntheticParticleDataset(Dataset):
    """
    On-the-fly generator for synthetic particle image pairs with known flow.

    Each sample returns:
        images: [2, H, W] — concatenated frame pair
        flow:   [2, H, W] — ground-truth displacement field (u, v) in pixels
    """

    FLOW_TYPES = ["vortex", "uniform", "shear", "channel", "multi_vortex"]

    def __init__(
        self,
        num_samples=1000,
        image_size=256,
        num_particles_range=(500, 2000),
        particle_diameter_range=(2.0, 4.0),
        noise_level=0.05,
        flow_type="random",
        max_displacement=8.0,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_particles_range = num_particles_range
        self.particle_diameter_range = particle_diameter_range
        self.noise_level = noise_level
        self.flow_type = flow_type
        self.max_displacement = max_displacement

    def __len__(self):
        return self.num_samples

    # ------------------------------------------------------------------
    # Flow field generation
    # ------------------------------------------------------------------

    def _generate_flow_field(self, flow_type):
        """Return a [2, H, W] displacement field in pixels."""
        S = self.image_size
        y, x = torch.meshgrid(
            torch.linspace(0, 1, S),
            torch.linspace(0, 1, S),
            indexing="ij",
        )
        md = self.max_displacement / S  # normalised max displacement

        if flow_type == "vortex":
            cx = 0.5 + 0.2 * (torch.rand(1).item() - 0.5)
            cy = 0.5 + 0.2 * (torch.rand(1).item() - 0.5)
            r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2) + 1e-6
            gamma = md * (1.0 + torch.rand(1).item())
            r0 = 0.1 + 0.05 * torch.rand(1).item()
            v_theta = gamma / (2 * np.pi * r) * (1 - torch.exp(-(r ** 2) / r0 ** 2))
            u = -v_theta * (y - cy) / r
            v = v_theta * (x - cx) / r

        elif flow_type == "uniform":
            angle = 2 * np.pi * torch.rand(1).item()
            mag = md * (0.3 + 0.7 * torch.rand(1).item())
            u = torch.full_like(x, mag * np.cos(angle))
            v = torch.full_like(x, mag * np.sin(angle))

        elif flow_type == "shear":
            mag = md * (0.5 + 0.5 * torch.rand(1).item())
            if torch.rand(1).item() > 0.5:
                u = mag * (y - 0.5)
                v = torch.zeros_like(x)
            else:
                u = torch.zeros_like(x)
                v = mag * (x - 0.5)

        elif flow_type == "channel":
            mag = md * (0.5 + 0.5 * torch.rand(1).item())
            u = mag * 4 * y * (1 - y)
            v = torch.zeros_like(x)

        elif flow_type == "multi_vortex":
            u = torch.zeros_like(x)
            v = torch.zeros_like(x)
            n_vortices = torch.randint(2, 5, (1,)).item()
            for _ in range(n_vortices):
                cx, cy = torch.rand(1).item(), torch.rand(1).item()
                r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2) + 1e-6
                sign = 1 if torch.rand(1).item() > 0.5 else -1
                gamma = md * (0.5 + 0.5 * torch.rand(1).item()) * sign
                r0 = 0.05 + 0.1 * torch.rand(1).item()
                v_theta = gamma / (2 * np.pi * r) * (1 - torch.exp(-(r ** 2) / r0 ** 2))
                u += -v_theta * (y - cy) / r
                v += v_theta * (x - cx) / r
        else:
            raise ValueError(f"Unknown flow type: {flow_type}")

        # Convert normalised displacement → pixel displacement
        flow = torch.stack([u * S, v * S], dim=0)  # [2, H, W]
        return flow

    # ------------------------------------------------------------------
    # Particle rendering
    # ------------------------------------------------------------------

    def _render_particles(self, positions, diameters):
        """
        Render a particle image as a sum of Gaussian blobs.

        Uses the separable-Gaussian trick: each 2-D Gaussian is the outer
        product of two 1-D Gaussians, so the sum over N particles becomes
        a single [S, N] @ [N, S] matrix multiply — orders of magnitude
        faster than a Python loop over N particles.

        Args:
            positions: [N, 2] (x, y) in pixel coords
            diameters: [N] particle diameters in pixels
        Returns:
            image: [H, W] float tensor in [0, 1]
        """
        S = self.image_size
        coords = torch.arange(S, dtype=torch.float32)   # [S]

        px = positions[:, 0]          # [N]
        py = positions[:, 1]          # [N]
        sigma = diameters / 4.0       # [N]
        inv_2s2 = 1.0 / (2.0 * sigma ** 2)              # [N]

        # 1-D Gaussian components  — [S, N]
        gx = torch.exp(-(coords.unsqueeze(1) - px.unsqueeze(0)) ** 2 * inv_2s2.unsqueeze(0))
        gy = torch.exp(-(coords.unsqueeze(1) - py.unsqueeze(0)) ** 2 * inv_2s2.unsqueeze(0))

        # Sum of outer products via matmul: image[y, x] = Σ_n gy[y,n]·gx[x,n]
        image = gy @ gx.t()           # [S, S]

        mx = image.max()
        if mx > 0:
            image = image / mx
        return image

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        # Pick flow type
        if self.flow_type == "random":
            ft = self.FLOW_TYPES[torch.randint(0, len(self.FLOW_TYPES), (1,)).item()]
        else:
            ft = self.flow_type

        flow = self._generate_flow_field(ft)  # [2, H, W]

        # Sample particles
        n = torch.randint(self.num_particles_range[0], self.num_particles_range[1], (1,)).item()
        positions1 = torch.rand(n, 2) * self.image_size
        diameters = (
            torch.rand(n) * (self.particle_diameter_range[1] - self.particle_diameter_range[0])
            + self.particle_diameter_range[0]
        )

        # Displace particles according to flow field (vectorised)
        S = self.image_size
        ix = positions1[:, 0].clamp(0, S - 1).long()   # [N]
        iy = positions1[:, 1].clamp(0, S - 1).long()   # [N]
        positions2 = positions1.clone()
        positions2[:, 0] += flow[0, iy, ix]
        positions2[:, 1] += flow[1, iy, ix]

        # Render
        img1 = self._render_particles(positions1, diameters)
        img2 = self._render_particles(positions2, diameters)

        # Sensor noise
        if self.noise_level > 0:
            img1 = (img1 + self.noise_level * torch.randn_like(img1)).clamp(0, 1)
            img2 = (img2 + self.noise_level * torch.randn_like(img2)).clamp(0, 1)

        images = torch.stack([img1, img2], dim=0)  # [2, H, W]
        return images, flow
