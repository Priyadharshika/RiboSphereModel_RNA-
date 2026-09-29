import numpy as np
import torch
from biotite.structure as struc
import io as pdbx
from biotite.structure.io.pdbx import CIFFile, get_structure
from .config import ATOM_NAMES


def process_rna_cif(cif_path):
    try:
        cif_file = CIFFile.read(str(cif_path))
        structure = get_structure(cif_file, model=1)

        # Keep RNA nucleotides
        mask = np.isin(structure.res_name, ["A", "U", "G", "C"])
        structure = structure[mask]

        if len(structure) == 0:
            return None

        atom_names = [
            "P", "C5'", "C4'", "C3'", "C2'",
            "C1'", "O5'", "O4'", "O3'", "O2'"
        ]

        coords = []

        for residue in struc.residue_iter(structure):
            residue_coords = []

            for atom_name in atom_names:
                idx = np.where(residue.atom_name == atom_name)[0]

                if len(idx) == 0:
                    residue_coords = []
                    break

                residue_coords.append(residue.coord[idx[0]])

            if len(residue_coords) == 10:
                coords.append(residue_coords)

        if not coords:
            return None

        coords = np.asarray(coords, dtype=np.float32)

        # Mean-center only — NO random rotation
        coords -= coords.mean(axis=(0, 1), keepdims=True)

        return torch.tensor(coords, dtype=torch.float32)

    except Exception as e:
        print(f"Error processing {cif_path.name}: {e}")
        return None
