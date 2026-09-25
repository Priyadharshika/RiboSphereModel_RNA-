import torch
import torch.nn as nn
from torch import Tensor, int32

def straight_through_round(inputs: Tensor) -> Tensor:

    rounded = inputs.round()

    return inputs + (rounded - inputs).detach()

class FiniteScalarQuantizer(nn.Module):

    def __init__(
        self,
        levels,
        input_dimension=None,
        output_dimension=None,
        num_codebooks=1,
        keep_codebook_dimension=None,
        scale=None,
        jitter_spread=0.0,
    ):

        super().__init__()

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if not levels or any(level < 2 for level in levels):
            raise ValueError(
                "levels must contain integers greater than one."
            )

        if num_codebooks <= 0:
            raise ValueError(
                "num_codebooks must be positive."
            )

        if jitter_spread < 0:
            raise ValueError(
                "jitter_spread must be non-negative."
            )

        # ----------------------------------------------------
        # Levels
        # ----------------------------------------------------

        _levels = torch.tensor(
            levels,
            dtype=int32
        )

        self.register_buffer(
            "_levels",
            _levels,
            persistent=False
        )

        # ----------------------------------------------------
        # Basis
        # ----------------------------------------------------

        _basis = torch.cumprod(
            torch.tensor(
                [1] + levels[:-1]
            ),
            dim=0,
            dtype=int32
        )

        self.register_buffer(
            "_basis",
            _basis,
            persistent=False
        )

        self.scale = scale

        # ----------------------------------------------------
        # Dimensions
        # ----------------------------------------------------

        codebook_dimension = len(levels)

        self.codebook_dimension = (
            codebook_dimension
        )

        self.jitter_spread = jitter_spread

        effective_codebook_dimension = (
            codebook_dimension
            * num_codebooks
        )

        self.num_codebooks = num_codebooks

        self.effective_codebook_dimension = (
            effective_codebook_dimension
        )

        if keep_codebook_dimension is None:

            keep_codebook_dimension = (
                num_codebooks > 1
            )

        if (
            num_codebooks > 1
            and not keep_codebook_dimension
        ):

            raise ValueError(
                "keep_codebook_dimension must "
                "be true with multiple codebooks."
            )

        self.keep_codebook_dimension = (
            keep_codebook_dimension
        )

        # ----------------------------------------------------
        # Input dimension
        # ----------------------------------------------------

        self.input_dimension = (
            input_dimension
            if input_dimension is not None
            else len(_levels) * num_codebooks
        )

        if self.input_dimension <= 0:

            raise ValueError(
                "input_dimension must be positive."
            )

        # ----------------------------------------------------
        # Input projection
        # ----------------------------------------------------

        has_projections = (
            self.input_dimension
            != effective_codebook_dimension
        )

        self.input_projection = (

            nn.Linear(
                self.input_dimension,
                effective_codebook_dimension
            )

            if has_projections

            else nn.Identity()
        )

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        if output_dimension is not None:

            self.output_projection = nn.Linear(
                effective_codebook_dimension,
                output_dimension
            )

        else:

            self.output_projection = (

                nn.Linear(
                    effective_codebook_dimension,
                    self.input_dimension
                )

                if has_projections

                else nn.Identity()
            )

        self.has_projections = has_projections

        # ----------------------------------------------------
        # Codebook size
        # ----------------------------------------------------

        self.codebook_size = int(
            self._levels.prod().item()
        )

        # ----------------------------------------------------
        # Implicit codebook
        # ----------------------------------------------------

        implicit_codebook = self.indices_to_codes(
            torch.arange(
                self.codebook_size
            ),
            apply_output_projection=False,
        )

        self.register_buffer(
            "implicit_codebook",
            implicit_codebook,
            persistent=False
        )

    # ========================================================
    # Bound inputs
    # ========================================================

    def bound_inputs(
        self,
        inputs,
        epsilon=1e-3
    ):

        if self.training and self.jitter_spread:

            inputs = (
                inputs
                + torch.randn_like(inputs)
                * self.jitter_spread
            )

        half_width = (
            (self._levels - 1)
            * (1 - epsilon)
            / 2
        )

        offset = torch.where(
            self._levels % 2 == 0,
            0.5,
            0.0
        )

        shift = (
            offset / half_width
        ).tan()

        return (
            (inputs + shift).tanh()
            * half_width
            - offset
        )

    # ========================================================
    # Quantize
    # ========================================================

    def quantize(self, inputs):

        quantized = (
            straight_through_round(
                self.bound_inputs(inputs)
            )
        )

        half_width = (
            self._levels // 2
        )

        # Normalize to [-1, 1]
        return (
            quantized / half_width
        )

    # ========================================================
    # Normalized code -> integer coordinates
    # ========================================================

    def _normalized_to_code_coordinates(
        self,
        normalized_codes
    ):

        half_width = (
            self._levels // 2
        )

        return (
            normalized_codes
            * half_width
            + half_width
        )

    # ========================================================
    # Integer coordinates -> normalized code
    # ========================================================

    def _code_coordinates_to_normalized(
        self,
        codes
    ):

        half_width = (
            self._levels // 2
        )

        return (
            codes - half_width
        ) / half_width

    # ========================================================
    # Codes -> token IDs
    # ========================================================

    def codes_to_indices(
        self,
        normalized_codes
    ):

        if (
            normalized_codes.shape[-1]
            != self.codebook_dimension
        ):

            raise ValueError(
                f"Expected code dimension "
                f"{self.codebook_dimension}, "
                f"received "
                f"{normalized_codes.shape[-1]}."
            )

        code_coordinates = (
            self._normalized_to_code_coordinates(
                normalized_codes
            )
        )

        return (
            code_coordinates
            * self._basis
        ).sum(dim=-1).to(int32)

    # ========================================================
    # Token IDs -> codes
    # ========================================================

    def indices_to_codes(
        self,
        indices,
        apply_output_projection=True
    ):

        indices = indices[..., None]

        code_coordinates = (
            indices // self._basis
        ) % self._levels

        codes = (
            self._code_coordinates_to_normalized(
                code_coordinates
            )
        )

        if self.keep_codebook_dimension:

            codes = codes.reshape(
                *codes.shape[:-2],
                -1
            )

        if apply_output_projection:

            codes = self.output_projection(
                codes
            )

        return codes

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, inputs):

        if inputs.shape[-1] != self.input_dimension:

            raise ValueError(
                f"Expected input dimension "
                f"{self.input_dimension}, "
                f"received "
                f"{inputs.shape[-1]}."
            )

        # ----------------------------------------------------
        # 256 -> 5
        # ----------------------------------------------------

        projected_inputs = (
            self.input_projection(inputs)
        )

        # ----------------------------------------------------
        # [B,L,5] -> [B,L,1,5]
        # ----------------------------------------------------

        projected_inputs = projected_inputs.reshape(
            *projected_inputs.shape[:-1],
            self.num_codebooks,
            self.codebook_dimension
        )

        # ----------------------------------------------------
        # Quantization
        # ----------------------------------------------------

        codes = self.quantize(
            projected_inputs
        )

        # ----------------------------------------------------
        # Codes -> token IDs
        # ----------------------------------------------------

        indices = self.codes_to_indices(
            codes
        )

        # ----------------------------------------------------
        # [B,L,1,5] -> [B,L,5]
        # ----------------------------------------------------

        codes = codes.reshape(
            *codes.shape[:-2],
            -1
        )

        # ----------------------------------------------------
        # 5 -> 512
        # ----------------------------------------------------

        outputs = self.output_projection(
            codes
        )

        # num_codebooks = 1
        if not self.keep_codebook_dimension:

            indices = indices.squeeze(-1)

        return outputs, indices