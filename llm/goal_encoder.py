"""
High-level VLA goal encoder.

Takes an image + language instruction → produces a compact goal vector
that tells the low-level world model WHAT to achieve.

Architecture:
    image (64×64×3) → small CNN → image_emb
    instruction      → pretrained sentence encoder → lang_emb
    [image_emb, lang_emb] → MLP → goal (goal_dim)
"""

import torch
import torch.nn as nn


class GoalEncoder(nn.Module):
    """
    Fuses vision + language into a goal embedding for the world model.

    Forward:  (image [B,3,H,W], lang_emb [B,384]) → goal [B, goal_dim]
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ── Image branch (lightweight CNN) ──
        self.image_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 64→32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32→16
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # 16→8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),  # → 64×4×4 = 1024
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
        )

        # ── Language projection ──
        self.lang_net = nn.Sequential(
            nn.Linear(cfg.language_dim, 128),
            nn.ReLU(),
        )

        # ── Fusion → goal ──
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.goal_dim),
        )

    def forward(self, image, lang_emb):
        """
        Args:
            image:    [B, 3, H, W] float32, already normalized to [0,1]
            lang_emb: [B, language_dim] float32
        Returns:
            goal:     [B, goal_dim] float32
        """
        img_feat = self.image_net(image)
        lang_feat = self.lang_net(lang_emb)
        return self.fusion(torch.cat([img_feat, lang_feat], dim=1))
