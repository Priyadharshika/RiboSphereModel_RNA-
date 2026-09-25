import torch
import torch.nn as nn

from .attention import SelfAttention
from .feed_forward import FeedForward
from .normalization import root_mean_square_norm


class TransformerBlock(nn.Module):

    def __init__(
        self,
        num_channels,
        num_heads,
        mlp_factor,
        attention_backend="sdpa",
        dropout=0.1,
        use_pairwise_bias=False,
        pairwise_channels=0,
    ):

        super().__init__()

        self.attention = SelfAttention(
            model_dimension=num_channels,
            num_heads=num_heads,
            attention_backend=attention_backend,
            dropout=dropout,
        )

        self.feed_forward = FeedForward(
            num_channels,
            num_channels * mlp_factor,
            num_channels,
            activation=nn.GELU,
            dropout=dropout,
        )

        self.use_pairwise_bias = (
            use_pairwise_bias
        )

        if use_pairwise_bias:

            self.pair_bias_projection = nn.Linear(
                pairwise_channels,
                1
            )

            self.pair_bias_norm = nn.LayerNorm(
                pairwise_channels
            )

    def _add_pair_bias(
        self,
        pairwise_features,
        attention_arguments
    ):

        pair_bias = (
            self.pair_bias_projection(
                self.pair_bias_norm(
                    pairwise_features
                )
            )
            .squeeze(-1)
        )

        attention_arguments = dict(
            attention_arguments
        )

        attention_mask = (
            attention_arguments["attn_mask"]
        )

        additive_pair_bias = torch.where(
            attention_mask,
            pair_bias,
            torch.full_like(
                pair_bias,
                -torch.inf
            )
        )

        attention_arguments["attn_mask"] = (
            additive_pair_bias.unsqueeze(1)
        )

        return attention_arguments

    def forward(
        self,
        hidden_states,
        pairwise_features=None,
        **attention_arguments
    ):

        if self.use_pairwise_bias:

            if pairwise_features is None:
                raise ValueError(
                    "pairwise_features required"
                )

            attention_arguments = (
                self._add_pair_bias(
                    pairwise_features,
                    attention_arguments
                )
            )

        hidden_states = (
            hidden_states
            + self.attention(
                root_mean_square_norm(
                    hidden_states
                ),
                **attention_arguments
            )
        )

        hidden_states = (
            hidden_states
            + self.feed_forward(
                root_mean_square_norm(
                    hidden_states
                )
            )
        )

        return hidden_states


class TransformerStack(nn.Module):

    def __init__(
        self,
        num_channels,
        num_heads,
        mlp_factor,
        window_size,
        num_layers,
        attention_backend="sdpa",
        dropout=0.1,
        pairwise_channels=0,
        is_causal=False,
    ):

        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(
                num_channels=num_channels,
                num_heads=num_heads,
                mlp_factor=mlp_factor,
                attention_backend=attention_backend,
                dropout=dropout,
                use_pairwise_bias=(
                    pairwise_channels > 0
                ),
                pairwise_channels=(
                    pairwise_channels
                ),
            )
            for _ in range(num_layers)
        ])

        self.window_size = window_size
        self.is_causal = is_causal

    def forward(
        self,
        hidden_states,
        pairwise_features=None
    ):

        sequence_length = (
            hidden_states.shape[1]
        )

        positions = torch.arange(
            sequence_length,
            device=hidden_states.device
        )

        attention_mask = (
            positions[:, None]
            - positions[None, :]
        ).abs() <= self.window_size

        if self.is_causal:

            attention_mask = (
                attention_mask
                & (
                    positions[:, None]
                    >= positions[None, :]
                )
            )

        attention_arguments = {
            "attn_mask": attention_mask.unsqueeze(0)
        }

        for block in self.blocks:

            hidden_states = block(
                hidden_states,
                pairwise_features=pairwise_features,
                **attention_arguments
            )

        return hidden_states