import torch


def random_rotation(x):
    """
    Apply a random 3D rotation to RNA coordinates.

    Parameters
    ----------
    x : torch.Tensor
        Coordinates with shape [B, L, A, 3].

    Returns
    -------
    torch.Tensor
        Rotated coordinates with shape [B, L, A, 3].
    """

    B = x.shape[0]

    # Generate random matrices
    M = torch.randn(
        B,
        3,
        3,
        device=x.device,
        dtype=x.dtype
    )

    # QR decomposition
    Q, R = torch.linalg.qr(M)

    # Ensure proper rotation
    # determinant should be +1
    det = torch.det(Q)

    signs = torch.where(
        det < 0,
        -torch.ones(
            B,
            device=x.device,
            dtype=x.dtype
        ),
        torch.ones(
            B,
            device=x.device,
            dtype=x.dtype
        )
    )

    Q[:, :, 0] *= signs[:, None]

    # Apply rotation
    x_rotated = torch.einsum(
        "blij,bjk->blik",
        x,
        Q
    )

    return x_rotated