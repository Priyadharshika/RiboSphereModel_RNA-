import torch.nn.functional as F


def root_mean_square_norm(tensor):
    """
    Apply RMS normalization over the final dimension.
    """

    return F.rms_norm(
        tensor,
        (tensor.shape[-1],)
    )