import numpy as np
import torch
from biotite.structure import io as pdbx

from .config import ATOM_NAMES


def process_rna_cif(cif_path):
    """
    Process an RNA CIF file into a centered coordinate tensor.

    Parameters
    ----------
    cif_path : str or Path
        Path to the CIF file.

    Returns
    -------
    torch.Tensor or None
        Tensor of shape [L, 10, 3], where:
        L = number of valid RNA residues
        10 = selected backbone atoms
        3 = XYZ coordinates

        Returns None if the structure cannot be processed.
    """

    try:
        # Read CIF file
        cif_file = pdbx.CIFFile.read(str(cif_path))

        # Extract first model
        structure = pdbx.get_structure(cif_file, model=1)

        # Keep only required RNA backbone atoms
        mask = np.isin(structure.atom_name, ATOM_NAMES)
        structure = structure[mask]

        if len(structure) == 0:
            return None

        # Get residue IDs
        residue_ids = structure.res_id
        unique_residues = np.unique(residue_ids)

        coordinates = []

        for residue_id in unique_residues:

            residue_atoms = structure[
                residue_ids == residue_id
            ]

            atom_coords = []
            valid = True

            for atom_name in ATOM_NAMES:

                atom = residue_atoms[
                    residue_atoms.atom_name == atom_name
                ]

                # Residue is incomplete
                if len(atom) == 0:
                    valid = False
                    break

                # Take first occurrence
                atom_coords.append(atom.coord[0])

            # Keep only complete residues
            if valid:
                coordinates.append(atom_coords)

        if len(coordinates) == 0:
            return None

        # Convert to tensor
        # Shape: [L, 10, 3]
        x = torch.tensor(
            np.array(coordinates),
            dtype=torch.float32
        )

        # Mean-center coordinates
        center = x.mean(
            dim=(0, 1),
            keepdim=True
        )

        x_centered = x - center

        return x_centered

    except Exception as e:

        print(
            f"Error processing {cif_path.name}: {e}"
        )

        return None