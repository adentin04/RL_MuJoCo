"""VLA feature extractor — fuses image, state, and language into one embedding."""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class VLAExtractor(BaseFeaturesExtractor):
    """
    Multi-modal feature extractor for Stable-Baselines3.

    Processes three observation keys:
        image       → CNN  → image_features
        state       → MLP  → state_features
        instruction → MLP  → language_features
    Then concatenates and fuses into a single vector.
    """

    def __init__(self, observation_space: spaces.Dict, config):
        features_dim = config.combined_features
        super().__init__(observation_space, features_dim=features_dim)

        img_c = observation_space["image"].shape[0]      # 3
        state_dim = observation_space["state"].shape[0]   # 15
        lang_dim = observation_space["instruction"].shape[0]  # 384

        # ── Image encoder (small CNN) ──
        self.image_net = nn.Sequential(
            nn.Conv2d(img_c, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, config.image_features),
            nn.ReLU(),
        )

        # ── State encoder ──
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, config.state_features),
            nn.ReLU(),
        )

        # ── Language encoder (projects pretrained embedding) ──
        self.lang_net = nn.Sequential(
            nn.Linear(lang_dim, 128),
            nn.ReLU(),
            nn.Linear(128, config.language_features),
            nn.ReLU(),
        )

        # ── Fusion: merge all modalities ──
        fusion_in = config.image_features + config.state_features + config.language_features
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        img = observations["image"].float() / 255.0
        state = observations["state"].float()
        lang = observations["instruction"].float()

        img_feat = self.image_net(img)
        state_feat = self.state_net(state)
        lang_feat = self.lang_net(lang)

        return self.fusion(torch.cat([img_feat, state_feat, lang_feat], dim=1))
