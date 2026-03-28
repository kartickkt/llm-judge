"""Download the Prometheus 2 Preference Collection from HuggingFace.

Saves the dataset in HuggingFace Arrow format to data/raw/, then creates
a reproducible 95/5 train/val split saved to data/splits/.

Usage:
    python data/download_datasets.py
    python data/download_datasets.py --output-dir data/raw --seed 42
"""

import argparse
import logging
from pathlib import Path

from datasets import load_dataset, DatasetDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_ID = "prometheus-eval/Preference-Collection"
DEFAULT_OUTPUT_DIR = Path("data/raw")
DEFAULT_SPLITS_DIR = Path("data/splits")
RANDOM_SEED = 42
VAL_FRACTION = 0.05


def download_preference_collection(output_dir: Path) -> None:
    """Download the Preference Collection dataset and save to disk.

    Args:
        output_dir: Directory where the raw dataset will be saved in Arrow format.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s from HuggingFace...", DATASET_ID)
    ds = load_dataset(DATASET_ID)
    logger.info("Dataset downloaded. Splits: %s", list(ds.keys()))
    for split, dataset in ds.items():
        logger.info("  %s: %d rows", split, len(dataset))
    ds.save_to_disk(str(output_dir))
    logger.info("Saved raw dataset to %s", output_dir)


def create_train_val_split(raw_dir: Path, splits_dir: Path, seed: int) -> None:
    """Create a reproducible 95/5 train/val split from the raw dataset.

    Uses the 'train' split of the raw download. If the raw dataset has no
    'train' split, falls back to the first available split.

    Args:
        raw_dir: Directory containing the raw Arrow dataset.
        splits_dir: Directory where the split dataset will be saved.
        seed: Random seed for reproducibility.
    """
    from datasets import load_from_disk

    splits_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading raw dataset from %s...", raw_dir)
    ds = load_from_disk(str(raw_dir))

    # Resolve which split to use as the training corpus
    if isinstance(ds, DatasetDict):
        if "train" in ds:
            full = ds["train"]
        else:
            first_key = next(iter(ds))
            logger.warning(
                "No 'train' split found; using '%s' split instead.", first_key
            )
            full = ds[first_key]
    else:
        full = ds  # already a single Dataset

    logger.info("Full dataset size: %d rows", len(full))

    split_result = full.train_test_split(
        test_size=VAL_FRACTION, seed=seed, shuffle=True
    )
    train_ds = split_result["train"]
    val_ds = split_result["test"]

    logger.info(
        "Split sizes — train: %d, val: %d (%.1f%% val)",
        len(train_ds),
        len(val_ds),
        100 * len(val_ds) / len(full),
    )

    final = DatasetDict({"train": train_ds, "val": val_ds})
    final.save_to_disk(str(splits_dir))
    logger.info("Saved train/val split to %s", splits_dir)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download Preference Collection and create train/val split."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save raw dataset (default: data/raw)",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=DEFAULT_SPLITS_DIR,
        help="Directory to save train/val split (default: data/splits)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step and only (re)create splits from existing raw data.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    if not args.skip_download:
        download_preference_collection(args.output_dir)
    else:
        logger.info("Skipping download (--skip-download set).")

    create_train_val_split(args.output_dir, args.splits_dir, args.seed)
    logger.info("Done.")


if __name__ == "__main__":
    main()
