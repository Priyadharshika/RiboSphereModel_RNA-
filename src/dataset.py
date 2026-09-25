from pathlib import Path
import torch

from .preprocessing import process_rna_cif


def process_cif_dataset(cif_files, max_samples=500):
    """
    Process multiple RNA CIF files.

    Parameters
    ----------
    cif_files : list
        List of CIF file paths.
    max_samples : int
        Maximum number of successfully processed structures.

    Returns
    -------
    processed_data : list[torch.Tensor]
        RNA coordinate tensors.
    processed_names : list[str]
        Corresponding CIF filenames.
    """

    processed_data = []
    processed_names = []

    for cif_file in cif_files:

        if len(processed_data) >= max_samples:
            break

        x = process_rna_cif(cif_file)

        if x is not None:

            processed_data.append(x)
            processed_names.append(
                Path(cif_file).name
            )

            print(
                f"{len(processed_data):3d}/{max_samples} | "
                f"{Path(cif_file).name} | "
                f"Shape: {tuple(x.shape)}"
            )

    print(
        f"\nSuccessfully processed: "
        f"{len(processed_data)}"
    )

    return processed_data, processed_names


def save_processed_dataset(
    processed_data,
    processed_names,
    output_path
):
    """
    Save processed RNA structures to a PyTorch file.
    """

    torch.save(
        {
            "coordinates": processed_data,
            "names": processed_names
        },
        output_path
    )

    print(f"Saved to: {output_path}")


def load_processed_dataset(input_path):
    """
    Load a previously processed RNA dataset.
    """

    data = torch.load(
        input_path,
        weights_only=False
    )

    processed_data = data["coordinates"]
    processed_names = data["names"]

    print(
        f"Loaded {len(processed_data)} "
        f"RNA structures."
    )

    return processed_data, processed_names