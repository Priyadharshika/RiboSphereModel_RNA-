import torch
import torch.nn as nn

from .feed_forward import FeedForward


class PairwiseFeatureEmbedder(nn.Module):
    """
    Creates pairwise RNA residue features from:

    1. Residue-center distances
    2. Relative residue positions
    """

    def __init__(
        self,
        num_channels,
        num_distance_buckets,
    ):

        super().__init__()

        self.distance_embedding = nn.Embedding(
            num_distance_buckets,
            num_channels
        )

        self.relative_position_embedding = (
            nn.Embedding(
                128,
                num_channels
            )
        )

        self.register_buffer(
            "bins",
            torch.linspace(
                0,
                4**2,
                num_distance_buckets - 1
            )
        )

        self.projection = FeedForward(
            num_channels,
            4 * num_channels,
            num_channels,
            activation=nn.GELU,
        )

        self.norm = nn.LayerNorm(
            num_channels
        )

        self.num_channels = num_channels

    def forward(self, coordinates):

        sequence_length = coordinates.shape[1]

        # [B,L,10,3]
        # -> [B,L,3]

        residue_centers = coordinates.mean(
            dim=2
        )

        # [B,L,3]
        # -> [B,L,L]

        squared_distances = (
            (
                residue_centers[:, :, None]
                - residue_centers[:, None, :]
            ) ** 2
        ).sum(dim=-1)

        # Relative residue positions

        residue_indices = torch.arange(
            sequence_length,
            device=residue_centers.device
        )

        relative_indices = (
            residue_indices[:, None]
            - residue_indices[None, :]
        ).clip(
            min=-64,
            max=63
        ) + 64

        relative_position_features = (
            self.relative_position_embedding(
                relative_indices
            )
        )

        # Convert distances into buckets

        distance_buckets = torch.bucketize(
            squared_distances,
            self.bins
        )

        pairwise_features = (
            self.distance_embedding(
                distance_buckets
            )
            + relative_position_features
        )

        return self.projection(
            self.norm(
                pairwise_features
            )
        )