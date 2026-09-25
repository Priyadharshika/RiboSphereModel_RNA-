from pathlib import Path
from src.config import DATA_DIR, MAX_SAMPLES
from src.dataset import process_cif_dataset


def main():

    # Find all CIF files in the dataset folder
    cif_files = list(Path(DATA_DIR).rglob("*.cif"))

    print(f"Found {len(cif_files)} CIF files.")

    # Process the dataset
    processed_data, processed_names = process_cif_dataset(
        cif_files,
        max_samples=MAX_SAMPLES
    )

    print(f"Successfully processed: {len(processed_data)} RNA structures.")


if __name__ == "__main__":
    main()