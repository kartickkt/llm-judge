"""EDA script for the Preference Collection dataset.

Prints:
  - Total row count and schema verification
  - Distribution of orig_preference (A vs B balance)
  - Distribution of orig_score_A and orig_score_B
  - Sequence length distribution (tokenize instruction + output with Llama tokenizer)
  - Percentage of examples exceeding 4096 tokens
  - 3 random sample examples printed in full

Usage:
    python data/inspect_data.py
    python data/inspect_data.py --splits-dir data/splits --tokenizer meta-llama/Llama-3.1-8B-Instruct
    python data/inspect_data.py --no-tokenizer   # skip slow tokenization step
"""

import argparse
import random
from pathlib import Path
from collections import Counter
from typing import Optional

import numpy as np

SPLITS_DIR = Path("data/splits")
DEFAULT_TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 4096
RANDOM_SEED = 42
NUM_SAMPLES = 3

EXPECTED_FIELDS = [
    "orig_instruction",
    "orig_response_A",
    "orig_response_B",
    "orig_reference_answer",
    "orig_criteria",
    "orig_score_A",
    "orig_score_B",
    "orig_preference",
    "orig_feedback_A",
    "orig_feedback_B",
    "instruction",
    "output",
    "orig_feedback",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def hr(char: str = "-", width: int = 72) -> None:
    """Print a horizontal rule."""
    print(char * width)


def section(title: str) -> None:
    hr("=")
    print(f"  {title}")
    hr("=")


def print_distribution(counter: Counter, label: str, total: int) -> None:
    """Pretty-print a Counter as a percentage table."""
    print(f"\n{label} distribution (n={total}):")
    for key in sorted(counter):
        count = counter[key]
        pct = 100 * count / total if total else 0
        bar = "#" * int(pct / 2)
        print(f"  {key!s:>6}  {count:>7,}  {pct:5.1f}%  {bar}")


def percentile_summary(values: list[int], name: str) -> None:
    """Print percentile summary for a list of ints."""
    arr = np.array(values)
    print(f"\n{name} summary (n={len(arr):,}):")
    print(f"  min={arr.min():,}  p25={int(np.percentile(arr, 25)):,}  "
          f"median={int(np.median(arr)):,}  p75={int(np.percentile(arr, 75)):,}  "
          f"p90={int(np.percentile(arr, 90)):,}  p95={int(np.percentile(arr, 95)):,}  "
          f"p99={int(np.percentile(arr, 99)):,}  max={arr.max():,}")


# ---------------------------------------------------------------------------
# Inspection functions
# ---------------------------------------------------------------------------


def check_schema(dataset) -> None:
    """Verify all expected fields are present in the dataset."""
    section("1. SCHEMA VERIFICATION")
    actual_fields = set(dataset.column_names)
    print(f"\nActual columns ({len(actual_fields)}): {sorted(actual_fields)}")

    missing = [f for f in EXPECTED_FIELDS if f not in actual_fields]
    extra = [f for f in actual_fields if f not in EXPECTED_FIELDS]

    if missing:
        print(f"\n  [WARN] Missing expected fields: {missing}")
    else:
        print("\n  [OK] All expected fields present.")

    if extra:
        print(f"  [INFO] Extra fields not in spec: {extra}")

    print(f"\nTotal rows: {len(dataset):,}")


def check_preference_balance(dataset) -> None:
    """Check A vs B balance in orig_preference."""
    section("2. PREFERENCE BALANCE (orig_preference)")
    prefs = dataset["orig_preference"]
    counter = Counter(prefs)
    total = len(prefs)
    print_distribution(counter, "orig_preference", total)

    a_count = counter.get("A", 0) + counter.get(1, 0)
    b_count = counter.get("B", 0) + counter.get(2, 0)
    if a_count and b_count:
        ratio = a_count / b_count
        print(f"\n  A/B ratio: {ratio:.3f}  (ideal = 1.000)")


def check_score_distributions(dataset) -> None:
    """Check score distributions for orig_score_A and orig_score_B."""
    section("3. SCORE DISTRIBUTIONS")
    for field in ("orig_score_A", "orig_score_B"):
        scores = dataset[field]
        counter = Counter(scores)
        total = len(scores)
        print_distribution(counter, field, total)


def check_sequence_lengths(
    dataset, tokenizer_name: Optional[str], max_len: int
) -> None:
    """Tokenize instruction+output and report length distribution."""
    section("4. SEQUENCE LENGTH DISTRIBUTION")

    if tokenizer_name is None:
        print("\n  [SKIPPED] --no-tokenizer flag set.")
        return

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("\n  [SKIPPED] transformers not installed.")
        return

    print(f"\n  Loading tokenizer: {tokenizer_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print("  Tokenizing instruction + output fields (this may take a minute)...")
    instructions = dataset["instruction"]
    outputs = dataset["output"]

    lengths = []
    for inst, out in zip(instructions, outputs):
        combined = str(inst) + str(out)
        ids = tokenizer(combined, add_special_tokens=False)["input_ids"]
        lengths.append(len(ids))

    percentile_summary(lengths, "Sequence length (tokens)")

    over_limit = sum(1 for l in lengths if l > max_len)
    pct_over = 100 * over_limit / len(lengths) if lengths else 0
    print(f"\n  Examples exceeding {max_len:,} tokens: "
          f"{over_limit:,} / {len(lengths):,}  ({pct_over:.2f}%)")


def print_samples(dataset, n: int, seed: int) -> None:
    """Print n random examples in full."""
    section(f"5. RANDOM SAMPLE EXAMPLES (n={n})")
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), k=min(n, len(dataset)))

    for rank, idx in enumerate(indices, start=1):
        row = dataset[idx]
        hr("-")
        print(f"\n--- Example {rank} of {n}  (index {idx}) ---\n")
        for field in EXPECTED_FIELDS:
            value = row.get(field, "<field not found>")
            print(f"[{field}]\n{value}\n")
    hr("-")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="EDA for the Preference Collection dataset."
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=SPLITS_DIR,
        help="Directory containing the train/val split (default: data/splits)",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val"],
        help="Which split to inspect (default: train)",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"HuggingFace tokenizer ID for length analysis (default: {DEFAULT_TOKENIZER})",
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Skip sequence-length tokenization step.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for sample selection (default: 42)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_SAMPLES,
        help="Number of random examples to print (default: 3)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    from datasets import load_from_disk

    print(f"\nLoading dataset from {args.splits_dir} (split='{args.split}')...")
    ds_dict = load_from_disk(str(args.splits_dir))
    dataset = ds_dict[args.split]
    print(f"Loaded {len(dataset):,} rows.\n")

    check_schema(dataset)
    check_preference_balance(dataset)
    check_score_distributions(dataset)

    tokenizer_name = None if args.no_tokenizer else args.tokenizer
    check_sequence_lengths(dataset, tokenizer_name, MAX_SEQ_LENGTH)

    print_samples(dataset, args.num_samples, args.seed)

    print("\nInspection complete.")


if __name__ == "__main__":
    main()
