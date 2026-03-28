"""Tokenize Preference Collection splits into SFT-ready format with label masking.

For each example:
  - Wraps `instruction` (user) and `output` (assistant) in the Llama 3.1 chat template
  - Tokenizes the full conversation
  - Masks system + user tokens as -100 so only assistant tokens contribute to loss
  - Drops examples whose total token length exceeds --max-seq-length

Output fields per example:
  - input_ids      : full tokenized sequence
  - attention_mask : 1 for real tokens, 0 for padding
  - labels         : -100 for system/user tokens; real token IDs for assistant tokens

Usage:
    huggingface-cli login          # required for Llama 3.1 access
    python data/prepare_sft_data.py
    python data/prepare_sft_data.py --splits-dir data/splits --output-dir data/processed
"""

import argparse
import functools
import logging
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_from_disk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a fair evaluator language model."
DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MAX_SEQ_LENGTH = 4096
IGNORE_INDEX = -100
LOG_CHUNK_SIZE = 10_000


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def build_labels(
    full_input_ids: list[int],
    prefix_length: int,
) -> list[int]:
    """Return a labels list with system/user tokens masked as IGNORE_INDEX.

    Args:
        full_input_ids: Token IDs for the complete conversation (system + user + assistant).
        prefix_length: Number of tokens that belong to the system + user portion.
            Everything from this index onward is the assistant response and keeps
            its real token ID.

    Returns:
        labels list of the same length as full_input_ids.
    """
    labels = [IGNORE_INDEX] * prefix_length + full_input_ids[prefix_length:]
    return labels


def process_example(
    example: dict,
    tokenizer,
) -> dict:
    """Tokenize one example and build the label-masked output.

    Args:
        example: A single row from the dataset containing ``instruction`` and
            ``output`` fields.
        tokenizer: A HuggingFace ``PreTrainedTokenizer`` / ``PreTrainedTokenizerFast``
            instance for Llama-3.1-8B-Instruct.

    Returns:
        A dict with ``input_ids``, ``attention_mask``, ``labels``, and ``_seq_len``
        (used downstream to filter over-length examples).
    """
    messages_without_assistant = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["instruction"]},
    ]
    messages_full = messages_without_assistant + [
        {"role": "assistant", "content": example["output"]},
    ]

    # Tokenize the prefix (system + user) to determine where assistant starts.
    # add_generation_prompt=True appends the "<|start_header_id|>assistant<|end_header_id|>\n\n"
    # turn-start tokens so the prefix length is exactly where the assistant content begins.
    prefix_ids: list[int] = tokenizer.apply_chat_template(
        messages_without_assistant,
        tokenize=True,
        add_generation_prompt=True,
    )

    # Tokenize the full conversation (no generation prompt needed — assistant turn is complete).
    full_ids: list[int] = tokenizer.apply_chat_template(
        messages_full,
        tokenize=True,
        add_generation_prompt=False,
    )

    labels = build_labels(full_ids, len(prefix_ids))
    attention_mask = [1] * len(full_ids)

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "_seq_len": len(full_ids),
    }


def process_split(
    dataset,
    split_name: str,
    tokenizer,
    max_seq_length: int,
) -> tuple:
    """Tokenize and filter all examples in a single split.

    Args:
        dataset: A HuggingFace ``Dataset`` for one split.
        split_name: Human-readable name used only for logging ("train" / "val").
        tokenizer: The Llama-3.1-8B-Instruct tokenizer.
        max_seq_length: Maximum allowed token count; longer examples are dropped.

    Returns:
        A tuple (processed_dataset, n_dropped, pct_dropped).
    """
    n_before = len(dataset)
    logger.info("[%s] Processing %d examples in chunks of %d...", split_name, n_before, LOG_CHUNK_SIZE)

    tokenize_fn = functools.partial(process_example, tokenizer=tokenizer)
    chunks = []
    for chunk_start in range(0, n_before, LOG_CHUNK_SIZE):
        chunk_end = min(chunk_start + LOG_CHUNK_SIZE, n_before)
        chunk = dataset.select(range(chunk_start, chunk_end))
        processed_chunk = chunk.map(
            tokenize_fn,
            remove_columns=dataset.column_names,
            desc=f"{split_name} [{chunk_start}–{chunk_end}]",
        )
        chunks.append(processed_chunk)
        logger.info("[%s] Tokenized %d / %d examples", split_name, chunk_end, n_before)

    processed = concatenate_datasets(chunks)

    n_dropped = sum(1 for length in processed["_seq_len"] if length > max_seq_length)
    processed = processed.filter(
        lambda x: x["_seq_len"] <= max_seq_length,
        desc=f"Filtering {split_name}",
    )
    processed = processed.remove_columns(["_seq_len"])

    pct_dropped = 100.0 * n_dropped / n_before if n_before else 0.0
    logger.info(
        "[%s] Before: %d  After: %d  Dropped: %d (%.2f%%)",
        split_name,
        n_before,
        len(processed),
        n_dropped,
        pct_dropped,
    )

    return processed, n_dropped, pct_dropped


def print_length_stats(dataset, split_name: str) -> None:
    """Log sequence length percentiles for the processed dataset.

    Args:
        dataset: A processed HuggingFace ``Dataset`` containing ``input_ids``.
        split_name: Human-readable name used for logging.
    """
    import numpy as np

    lengths = [len(ids) for ids in dataset["input_ids"]]
    arr = np.array(lengths)
    logger.info(
        "[%s] Length stats — min=%d  p25=%d  median=%d  p75=%d  p90=%d  p95=%d  p99=%d  max=%d",
        split_name,
        arr.min(),
        int(np.percentile(arr, 25)),
        int(np.median(arr)),
        int(np.percentile(arr, 75)),
        int(np.percentile(arr, 90)),
        int(np.percentile(arr, 95)),
        int(np.percentile(arr, 99)),
        arr.max(),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare SFT training data with chat template and label masking."
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=DEFAULT_SPLITS_DIR,
        help=f"Directory containing train/val splits (default: {DEFAULT_SPLITS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save processed datasets (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"HuggingFace tokenizer ID (default: {DEFAULT_TOKENIZER})",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help=f"Maximum token length; longer examples are dropped (default: {DEFAULT_MAX_SEQ_LENGTH})",
    )
    return parser.parse_args()


def verify_label_boundary(tokenizer, example: dict, max_seq_length: int) -> None:
    """Process one example and log the label boundary for visual inspection.

    Decodes the last 50 characters of the masked (system+user) portion and the
    first 100 characters of the unmasked (assistant) portion so the boundary can
    be confirmed at a glance.

    Args:
        tokenizer: The Llama-3.1-8B-Instruct tokenizer.
        example: A single raw dataset row with ``instruction`` and ``output`` fields.
        max_seq_length: If the tokenized sequence exceeds this, the check is skipped.
    """
    result = process_example(example, tokenizer)
    if result["_seq_len"] > max_seq_length:
        logger.warning("[verify] First example exceeds max_seq_length — boundary check skipped.")
        return

    input_ids = result["input_ids"]
    labels = result["labels"]

    # Find the first index where the label is not IGNORE_INDEX
    boundary = next((i for i, lbl in enumerate(labels) if lbl != IGNORE_INDEX), len(labels))

    masked_text = tokenizer.decode(input_ids[:boundary], skip_special_tokens=False)
    unmasked_text = tokenizer.decode(input_ids[boundary:], skip_special_tokens=False)

    logger.info(
        "[verify] Label boundary at token index %d / %d total tokens",
        boundary,
        len(input_ids),
    )
    logger.info("[verify] ...last 50 chars of masked portion:  %r", masked_text[-50:])
    logger.info("[verify] first 100 chars of unmasked portion: %r", unmasked_text[:100])


def main() -> None:
    """Entry point."""
    args = parse_args()

    from transformers import AutoTokenizer

    logger.info("Loading tokenizer: %s", args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logger.info("Loading splits from %s...", args.splits_dir)
    raw = load_from_disk(str(args.splits_dir))

    if "train" in raw and len(raw["train"]) > 0:
        logger.info("--- Label boundary verification (first train example) ---")
        verify_label_boundary(tokenizer, raw["train"][0], args.max_seq_length)
        logger.info("--- End verification ---")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    processed_splits: dict = {}
    for split_name in ("train", "val"):
        if split_name not in raw:
            logger.warning("Split '%s' not found in %s — skipping.", split_name, args.splits_dir)
            continue

        processed_ds, n_dropped, pct_dropped = process_split(
            raw[split_name],
            split_name,
            tokenizer,
            args.max_seq_length,
        )
        print_length_stats(processed_ds, split_name)
        processed_splits[split_name] = processed_ds

    final = DatasetDict(processed_splits)
    final.save_to_disk(str(args.output_dir))
    logger.info("Saved processed dataset to %s", args.output_dir)

    # Summary
    logger.info("=== Summary ===")
    for split_name, ds in processed_splits.items():
        logger.info("  %s: %d examples", split_name, len(ds))
    logger.info("Done.")


if __name__ == "__main__":
    main()
