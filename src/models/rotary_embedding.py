import torch
import torch.nn as nn


def rotate_half_dimension(inputs):
    """
    Rotate the last dimension by swapping halves.
    """

    if inputs.shape[-1] % 2:
        raise ValueError(
            "The final input dimension must be even."
        )

    first_half, second_half = inputs.chunk(
        2,
        dim=-1
    )

    return torch.cat(
        (-second_half, first_half),
        dim=-1
    )


def apply_rotary_embedding(
    inputs,
    cosine,
    sine
):
    """
    Apply rotary positional embedding.
    """

    sequence_length = inputs.shape[-2]

    cosine = cosine[:, :sequence_length, :]
    sine = sine[:, :sequence_length, :]

    return (
        inputs * cosine
        + rotate_half_dimension(inputs) * sine
    )


class RotaryEmbedding(nn.Module):
    """
    Rotary positional embedding for attention queries and keys.
    """

    def __init__(self, embedding_dimension):

        super().__init__()

        if (
            embedding_dimension <= 0
            or embedding_dimension % 2
        ):
            raise ValueError(
                "embedding_dimension must be "
                "a positive even integer."
            )

        inverse_frequencies = 1.0 / (
            10000
            ** (
                torch.arange(
                    0,
                    embedding_dimension,
                    2
                ).float()
                / embedding_dimension
            )
        )

        self.register_buffer(
            "inv_freq",
            inverse_frequencies
        )

        self._cached_sequence_length = 0
        self._cached_cosine = None
        self._cached_sine = None

    def _get_cosine_sine_tables(
        self,
        inputs,
        sequence_dimension=-2,
    ):

        sequence_length = inputs.shape[
            sequence_dimension
        ]

        cache_invalid = (
            sequence_length
            != self._cached_sequence_length
            or self._cached_cosine is None
            or self._cached_sine is None
            or self._cached_cosine.device
            != inputs.device
            or self._cached_cosine.dtype
            != inputs.dtype
        )

        if cache_invalid:

            self._cached_sequence_length = (
                sequence_length
            )

            positions = torch.arange(
                sequence_length,
                device=inputs.device,
                dtype=self.inv_freq.dtype,
            )

            frequencies = torch.einsum(
                "i,j->ij",
                positions,
                self.inv_freq
            )

            embedding = torch.cat(
                (frequencies, frequencies),
                dim=-1
            ).to(dtype=inputs.dtype)

            self._cached_cosine = (
                embedding.cos()[None, :, :]
            )

            self._cached_sine = (
                embedding.sin()[None, :, :]
            )

        return (
            self._cached_cosine,
            self._cached_sine
        )

    def forward(self, query, key):

        if query.shape != key.shape:
            raise ValueError(
                "query and key must have "
                "identical shapes."
            )

        cosine, sine = (
            self._get_cosine_sine_tables(
                key
            )
        )

        return (
            apply_rotary_embedding(
                query,
                cosine,
                sine
            ),
            apply_rotary_embedding(
                key,
                cosine,
                sine
            ),
        )