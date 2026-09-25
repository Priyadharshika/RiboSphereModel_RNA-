import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SelfAttention
from .feed_forward import FeedForward

# ============================================================
# Adaptive modulation
# ============================================================

def apply_adaptive_modulation(inputs, shift, scale):
    return (
        inputs * (1 + scale.unsqueeze(-2))
        + shift.unsqueeze(-2)
    )


# ============================================================
# Sinusoidal timestep embedding
# ============================================================

class SinusoidalTimestepEmbedding(nn.Module):

    def __init__(
        self,
        hidden_size,
        frequency_embedding_size=256,
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(
                frequency_embedding_size,
                hidden_size,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_size,
                hidden_size,
                bias=True,
            ),
        )

        self.frequency_embedding_size = (
            frequency_embedding_size
        )

    @staticmethod
    def create_sinusoidal_embedding(
        times,
        embedding_dimension,
        max_period=10_000,
    ):

        half_dimension = embedding_dimension // 2

        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(
                half_dimension,
                dtype=torch.float32,
                device=times.device,
            )
            / half_dimension
        )

        phase = (
            times[:, None].float()
            * frequencies[None]
        )

        embedding = torch.cat(
            [
                torch.cos(phase),
                torch.sin(phase),
            ],
            dim=-1,
        )

        if embedding_dimension % 2:

            embedding = torch.cat(
                [
                    embedding,
                    torch.zeros_like(
                        embedding[:, :1]
                    ),
                ],
                dim=-1,
            )

        return embedding

    def forward(self, times):

        frequency_embedding = (
            self.create_sinusoidal_embedding(
                times,
                self.frequency_embedding_size,
            )
        )

        return self.projection(
            frequency_embedding
        )


# ============================================================
# Adaptive input projection
# ============================================================

class AdaptiveInputProjection(nn.Module):

    def __init__(
        self,
        input_channels,
        output_channels,
    ):
        super().__init__()

        self.projection = nn.Linear(
            input_channels,
            output_channels,
            bias=True,
        )

        self.norm = nn.LayerNorm(
            output_channels,
            elementwise_affine=False,
            eps=1e-6,
        )

        self.adaptive_norm_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                output_channels,
                2 * output_channels,
                bias=True,
            ),
        )

    def forward(
        self,
        inputs,
        conditioning,
    ):

        shift, scale = (
            self.adaptive_norm_modulation(
                conditioning
            ).chunk(2, dim=-1)
        )

        outputs = self.projection(inputs)

        return apply_adaptive_modulation(
            self.norm(outputs),
            shift,
            scale,
        )


# ============================================================
# Diffusion Transformer Block
# ============================================================

class DiffusionTransformerBlock(nn.Module):

    def __init__(
        self,
        num_channels,
        num_heads,
        mlp_factor,
        normalize_queries_and_keys=False,
        dropout=0.1,
        shared_adaln=None,
        attention_backend="sdpa",
    ):
        super().__init__()

        self.attention_backend = (
            attention_backend
        )

        # Use the SelfAttention class from
        # your encoder implementation
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

        self.norm1 = nn.LayerNorm(
            num_channels,
            elementwise_affine=False,
        )

        self.norm2 = nn.LayerNorm(
            num_channels,
            elementwise_affine=False,
        )

        # Kept exactly as in repository
        self.norm3 = nn.LayerNorm(
            num_channels,
            elementwise_affine=False,
        )

        if shared_adaln is not None:

            self.adaptive_norm_modulation = (
                shared_adaln
            )

        else:

            self.adaptive_norm_modulation = (
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(
                        num_channels,
                        num_channels * 6,
                        bias=True,
                    ),
                )
            )

    def forward(
        self,
        hidden_states,
        time_conditioning,
    ):

        adaptive_norm_parameters = (
            self.adaptive_norm_modulation(
                time_conditioning
            )
        )

        (
            attention_shift,
            attention_scale,
            attention_gate,
            feed_forward_shift,
            feed_forward_scale,
            feed_forward_gate,
        ) = adaptive_norm_parameters.chunk(
            6,
            dim=-1,
        )

        # Attention
        hidden_states = (
            hidden_states
            + attention_gate.unsqueeze(1)
            * self.attention(
                apply_adaptive_modulation(
                    self.norm1(hidden_states),
                    attention_shift,
                    attention_scale,
                ),
                attn_mask=None,
            )
        )

        # Feed-forward
        hidden_states = (
            hidden_states
            + feed_forward_gate.unsqueeze(1)
            * self.feed_forward(
                apply_adaptive_modulation(
                    self.norm2(hidden_states),
                    feed_forward_shift,
                    feed_forward_scale,
                )
            )
        )

        return hidden_states


# ============================================================
# Adaptive output projection
# ============================================================

class AdaptiveOutputProjection(nn.Module):

    def __init__(
        self,
        model_channels,
        output_channels,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(
            model_channels,
            elementwise_affine=False,
            eps=1e-6,
        )

        self.projection = nn.Linear(
            model_channels,
            output_channels,
            bias=True,
        )

        self.adaptive_norm_modulation = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    model_channels,
                    2 * model_channels,
                    bias=True,
                ),
            )
        )

    def forward(
        self,
        inputs,
        conditioning,
    ):

        shift, scale = (
            self.adaptive_norm_modulation(
                conditioning
            ).chunk(2, dim=-1)
        )

        outputs = apply_adaptive_modulation(
            self.norm(inputs),
            shift,
            scale,
        )

        return self.projection(outputs)


# ============================================================
# RiboSphere Diffusion Transformer
# ============================================================

class DiffusionTransformer(nn.Module):

    def __init__(
        self,
        num_channels=512,
        input_channels=30,
        num_layers=8,
        num_heads=8,
        conditioning_type="cat",
        mlp_factor=4,
        normalize_queries_and_keys=False,
        share_adaln=False,
        attention_backend="sdpa",
    ):
        super().__init__()

        self.input_projection = (
            AdaptiveInputProjection(
                input_channels,
                num_channels,
            )
        )

        self.share_adaln = share_adaln

        if share_adaln:

            self.shared_adaln_modulation = (
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(
                        num_channels,
                        num_channels * 6,
                        bias=True,
                    ),
                )
            )

        self.blocks = nn.ModuleList(
            [
                DiffusionTransformerBlock(
                    num_channels=num_channels,
                    num_heads=num_heads,
                    mlp_factor=mlp_factor,
                    normalize_queries_and_keys=(
                        normalize_queries_and_keys
                    ),
                    attention_backend=(
                        attention_backend
                    ),
                    shared_adaln=(
                        self.shared_adaln_modulation
                        if share_adaln
                        else None
                    ),
                )
                for _ in range(num_layers)
            ]
        )

        self.timestep_embedding = (
            SinusoidalTimestepEmbedding(
                num_channels
            )
        )

        self.conditioning_type = (
            conditioning_type
        )

        self.condition_embedding = nn.Embedding(
            2,
            num_channels,
        )

        self.output_projection = (
            AdaptiveOutputProjection(
                num_channels,
                input_channels,
            )
        )

    def forward(
        self,
        input_states,
        times,
        conditioning_states=None,
    ):

        if conditioning_states is None:
            raise ValueError(
                "conditioning_states must be provided."
            )

        # --------------------------------------------
        # Time embedding
        # --------------------------------------------

        time_conditioning = (
            self.timestep_embedding(times)
        )

        # --------------------------------------------
        # Noisy coordinates
        # 30 -> 512
        # --------------------------------------------

        hidden_states = (
            self.input_projection(
                input_states,
                time_conditioning,
            )
        )

        # --------------------------------------------
        # Condition token IDs
        # --------------------------------------------

        condition_shape = (
            conditioning_states.shape[:-1]
        )

        device = conditioning_states.device

        condition_type_ids = torch.cat(
            (
                torch.zeros(
                    condition_shape,
                    dtype=torch.long,
                    device=device,
                ),

                torch.ones(
                    condition_shape,
                    dtype=torch.long,
                    device=device,
                ),
            ),
            dim=-1,
        )

        # --------------------------------------------
        # Concatenate noisy structure + condition
        # --------------------------------------------

        hidden_states = torch.cat(
            [
                hidden_states,
                conditioning_states,
            ],
            dim=-2,
        )

        hidden_states = (
            hidden_states
            + self.condition_embedding(
                condition_type_ids
            )
        )

        # --------------------------------------------
        # 8 Transformer blocks
        # --------------------------------------------

        for block in self.blocks:

            hidden_states = block(
                hidden_states,
                time_conditioning,
            )

        # --------------------------------------------
        # Keep only coordinate tokens
        # --------------------------------------------

        sequence_length = (
            input_states.size(1)
        )

        hidden_states = (
            hidden_states[
                :, :sequence_length, :
            ]
        )

        # --------------------------------------------
        # 512 -> 30
        # --------------------------------------------

        return self.output_projection(
            hidden_states,
            time_conditioning,
        )