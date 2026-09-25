import torch
import torch.nn as nn
import torch.nn.functional as F

from .rotary_embedding import RotaryEmbedding


class SelfAttention(nn.Module):
    """
    Multi-head self-attention with rotary
    positional embeddings.
    """

    def __init__(
        self,
        model_dimension,
        num_heads,
        attention_backend="sdpa",
        dropout=0.1,
    ):

        super().__init__()

        if model_dimension % num_heads != 0:
            raise ValueError(
                "model_dimension must be divisible "
                "by num_heads."
            )

        self.model_dimension = model_dimension
        self.num_heads = num_heads

        self.head_dimension = (
            model_dimension // num_heads
        )

        self.attention_dropout = nn.Dropout(
            dropout
        )

        self.dropout = dropout

        self.rotary_embedding = (
            RotaryEmbedding(
                self.head_dimension
            )
        )

        self.qkv_projection = nn.Linear(
            model_dimension,
            3 * model_dimension,
            bias=True
        )

        self.output_projection = nn.Linear(
            model_dimension,
            model_dimension
        )

        self.residual_dropout = nn.Dropout(
            dropout
        )

        self.attention_backend = (
            attention_backend
        )

    def forward(
        self,
        hidden_states,
        **attention_arguments
    ):

        batch_size, sequence_length, _ = (
            hidden_states.shape
        )

        query, key, value = (
            self.qkv_projection(
                hidden_states
            ).split(
                self.model_dimension,
                dim=-1
            )
        )

        def split_heads(tensor):

            return (
                tensor.reshape(
                    batch_size,
                    sequence_length,
                    self.num_heads,
                    self.head_dimension,
                )
                .transpose(1, 2)
            )

        query, key, value = map(
            split_heads,
            (query, key, value)
        )

        query, key = (
            self.rotary_embedding(
                query,
                key
            )
        )

        if self.attention_backend == "sdpa":

            attention_output = (
                F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p=(
                        self.dropout
                        if self.training
                        else 0.0
                    ),
                    **attention_arguments
                )
            )

        else:

            raise ValueError(
                f"Unsupported attention backend: "
                f"{self.attention_backend}"
            )

        attention_output = (
            attention_output.transpose(
                1, 2
            )
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                self.model_dimension
            )
        )

        attention_output = (
            self.residual_dropout(
                self.output_projection(
                    attention_output
                )
            )
        )

        return attention_output