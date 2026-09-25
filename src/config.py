from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "rna_structures"

MAX_SAMPLES = 10

SEED = 42

ATOM_NAMES = [
    "P",
    "C5'",
    "C4'",
    "C3'",
    "C2'",
    "C1'",
    "O5'",
    "O4'",
    "O3'",
    "O2'"
]