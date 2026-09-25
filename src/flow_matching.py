import torch


class ConditionalFlowMatcher:

    def __init__(
        self,
        sigma=0.0,
        time_sampling_mode="uniform",
    ):
        self.sigma = sigma
        self.time_sampling_mode = time_sampling_mode

    def sample_flow(self, source, target):

        batch_size = source.shape[0]

        # Sample time t
        if self.time_sampling_mode == "uniform":

            times = torch.rand(
                batch_size,
                device=source.device,
            )

        else:

            raise ValueError(
                f"Unknown time sampling mode: "
                f"{self.time_sampling_mode}"
            )

        # [B] -> [B,1,1,1]
        t = times[:, None, None, None]

        # Linear interpolation
        intermediate_coordinates = (
            (1.0 - t) * source
            + t * target
        )

        # Target vector field
        target_vector_field = target - source

        return (
            times,
            intermediate_coordinates,
            target_vector_field,
        )