import torch
import torch.nn as nn

from .pairwise_features import (
    PairwiseFeatureEmbedder
)

from .transformer import TransformerStack


class RiboSphereEncoder(nn.Module):
    """
    RiboSphere geometric encoder.

    Input:
        [B, L, 10, 3]

    Output:
        [B, L, 256]
    """

    def __init__(self):

        super().__init__()

        # RiboSphere a10 configuration

        self.num_atoms = 10
        self.n_channels_encoder = 256
        self.n_layers_encoder = 2
        self.n_heads = 8
        self.mlp_factor = 4
        self.n_channels_pair = 64
        self.window_size = 8

        # Coordinate encoder

        self.coordinate_encoder = nn.Sequential(

            nn.Linear(
                self.num_atoms * 3,
                self.n_channels_encoder
            ),

            nn.SiLU(),

            nn.Linear(
                self.n_channels_encoder,
                self.n_channels_encoder
            ),

            nn.LayerNorm(
                self.n_channels_encoder
            ),
        )

        # Pairwise geometric features

        self.pairwise_feature_embedder = (
            PairwiseFeatureEmbedder(
                num_channels=64,
                num_distance_buckets=100,
            )
        )

        # Transformer

        self.encoder = TransformerStack(

            num_channels=256,
            num_heads=8,
            mlp_factor=4,
            window_size=8,
            num_layers=2,
            attention_backend="sdpa",
            dropout=0.1,
            pairwise_channels=64,
            is_causal=False,
        )

    def forward(self, coordinates):

        if coordinates.ndim != 4:
            raise ValueError(
                f"Expected [B,L,10,3], "
                f"got {coordinates.shape}"
            )

        if coordinates.shape[2] != 10:
            raise ValueError(
                f"Expected 10 atoms, "
                f"got {coordinates.shape[2]}"
            )

        if coordinates.shape[3] != 3:
            raise ValueError(
                f"Expected XYZ coordinates, "
                f"got {coordinates.shape[3]}"
            )

        # Center coordinates

        centered_coordinates = (
            coordinates
            - coordinates.mean(
                dim=(1, 2),
                keepdim=True
            )
        )

        # Pairwise features

        pairwise_features = (
            self.pairwise_feature_embedder(
                centered_coordinates
            )
        )

        # Flatten [10,3] -> [30]

        flattened_coordinates = (
            centered_coordinates.reshape(
                coordinates.shape[0],
                coordinates.shape[1],
                self.num_atoms * 3
            )
        )

        # Coordinate embedding

        encoder_states = (
            self.coordinate_encoder(
                flattened_coordinates
            )
        )

        # Transformer

        encoder_states = self.encoder(
            encoder_states,
            pairwise_features=pairwise_features
        )

        return encoder_states