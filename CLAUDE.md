# LLM-as-Judge: Fine-Tuning Llama-3.1-8B for Pairwise Evaluation

## Project Overview

Fine-tune Llama-3.1-8B-Instruct as a pairwise LLM judge using QDoRA SFT + DPO. The model evaluates two responses to an instruction and decides which is better based on a specific rubric. Training on the Prometheus 2 Preference Collection (200K pairwise examples). Evaluation against MT-Bench Human Judgments, HHH Alignment, and Auto-J. Serving via vLLM on AWS SageMaker (single A10G GPU).

## Key Architectural Decisions

- **Base model**: meta-llama/Llama-3.1-8B-Instruct (strongest 8B instruction-follower)
- **Fine-tuning**: QDoRA = 4-bit NF4 quantized base + DoRA adapters (rank 16, alpha 32)
- **Pipeline**: SFT (learn the judging task) → DPO (refine with preference optimization)
- **Judging format**: Pairwise (A vs B) with rubric-conditioned evaluation
- **GPU**: Single A10G (24GB VRAM), budget ~$100-150
- **Serving**: vLLM with LoRA adapter loading

## Development Environment

- **Local**: Windows, Python 3.10, VS Code + Claude Code
- **Training/Eval/Serving**: AWS SageMaker ml.g5.2xlarge (1x A10G)
- All data scripts and configs are developed locally, then run on AWS
- Use type hints, docstrings, and clear logging in all scripts

## Folder Structure

```
llm-judge/
├── CLAUDE.md                          # This file
├── README.md                          # Project overview, results, how to reproduce
├── requirements.txt                   # Pinned dependencies
├── configs/
│   ├── sft_config.yaml                # SFT training hyperparameters
│   └── dpo_config.yaml                # DPO training hyperparameters
├── data/
│   ├── download_datasets.py           # Download from HuggingFace
│   ├── prepare_sft_data.py            # Format for SFT training
│   ├── prepare_dpo_data.py            # Construct DPO preference pairs
│   └── inspect_data.py                # EDA: distribution checks, sample viewing
├── training/
│   ├── sft_train.py                   # QDoRA SFT training script
│   ├── dpo_train.py                   # DPO training script
│   └── merge_adapter.py              # Merge LoRA weights into base model
├── evaluation/
│   ├── run_benchmark.py               # Run model on MT-Bench, HHH, Auto-J
│   ├── run_gpt4_baseline.py           # GPT-4 as judge baseline
│   ├── run_base_model_baseline.py     # Prompted base model baseline
│   ├── bias_analysis.py               # Position, verbosity, self-enhancement
│   ├── compute_metrics.py             # Agreement, Cohen's Kappa, Krippendorff's alpha
│   └── generate_report.py            # Create results tables and charts
├── serving/
│   ├── vllm_serve.py                  # vLLM server launch script
│   ├── fastapi_gateway.py             # API wrapper with structured I/O
│   ├── benchmark_latency.py           # Latency/throughput benchmarks
│   └── cost_analysis.py              # Cost per judgment vs GPT-4
└── notebooks/                         # Jupyter notebooks for exploration
    ├── 01_data_exploration.ipynb
    ├── 02_training_monitoring.ipynb
    └── 03_results_analysis.ipynb
```

## Dataset: Preference Collection

**Source**: `prometheus-eval/Preference-Collection` on HuggingFace
**Size**: 200K pairwise examples (100K A-wins, 100K B-wins — balanced)
**License**: CC-BY-4.0

### Schema (raw fields)

| Field | Description |
|-------|-------------|
| `orig_instruction` | The user task/question |
| `orig_response_A` | First candidate response |
| `orig_response_B` | Second candidate response |
| `orig_reference_answer` | Gold-standard answer for comparison |
| `orig_criteria` | One of ~1,000 fine-grained evaluation rubrics |
| `orig_score_A` | Score for response A (1-5) |
| `orig_score_B` | Score for response B (1-5) |
| `orig_preference` | Verdict: "A" or "B" |
| `orig_feedback_A` | Why response A got its score |
| `orig_feedback_B` | Why response B got its score |
| `instruction` | **Pre-formatted full prompt** (task desc + all fields filled into template) |
| `output` | **Target**: "Feedback: (comparison)... [RESULT] A or B" |
| `orig_feedback` | Combined comparison feedback (same content as `output`) |

### How 20K instructions become 200K rows
One instruction can pair with multiple response combinations, each evaluated under different rubrics. The ~1,000 rubrics are the main multiplier: same response pair, different evaluation criterion = different training example.

### Pre-formatted training fields
The dataset provides `instruction` (model input) and `output` (model target) already assembled with the prompt template. The `[RESULT]` token is the parsing anchor — at inference we extract the verdict from what follows it.

## Prompt Template

### System Prompt
```
You are a fair evaluator language model.
```

### Pairwise Judging Template (used in `instruction` field)
```
###Task Description:
An instruction (might include an Input inside it), two responses to evaluate,
a reference answer, and a score rubric are given.

1. Write a detailed feedback comparing both responses based strictly on the
   given score rubric, not general quality.
2. After writing feedback, select which response is better: "A" or "B".
3. Output format: "Feedback: (write comparison) [RESULT] A or B"

###Instruction:
{instruction}

###Response A:
{response_a}

###Response B:
{response_b}

###Reference Answer:
{reference_answer}

###Score Rubric:
{rubric}

###Feedback:
```

## Training Configs

### Stage 1: QDoRA SFT
```yaml
base_model: "meta-llama/Llama-3.1-8B-Instruct"
load_in_4bit: true
bnb_4bit_quant_type: "nf4"
bnb_4bit_compute_dtype: "bfloat16"
bnb_4bit_use_double_quant: true
use_dora: true
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
num_epochs: 2
per_device_train_batch_size: 4
gradient_accumulation_steps: 4        # effective batch size = 16
learning_rate: 2e-4
lr_scheduler: "cosine"
warmup_ratio: 0.05
max_seq_length: 4096
bf16: true
gradient_checkpointing: true
logging_steps: 10
eval_steps: 200
save_steps: 200
```

Memory estimate: ~10-11GB on 24GB A10G.

### Stage 2: DPO
```yaml
base_model: "meta-llama/Llama-3.1-8B-Instruct"
sft_adapter: "./outputs/sft_checkpoint/"
beta: 0.1
loss_type: "sigmoid"
load_in_4bit: true
bnb_4bit_quant_type: "nf4"
bnb_4bit_compute_dtype: "bfloat16"
bnb_4bit_use_double_quant: true
use_dora: true
lora_rank: 16
lora_alpha: 32
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
num_epochs: 1
per_device_train_batch_size: 2
gradient_accumulation_steps: 8        # effective batch size = 16
learning_rate: 5e-5
lr_scheduler: "cosine"
warmup_ratio: 0.1
max_seq_length: 4096
bf16: true
gradient_checkpointing: true
```

Memory estimate: ~15-16GB on 24GB A10G (policy + reference model).

## Evaluation Benchmarks (NEVER train on these)

| Benchmark | Format | Size | What it tests |
|-----------|--------|------|---------------|
| MT-Bench Human Judgments | Pairwise | ~3.3K | Agreement with expert human preferences |
| HHH Alignment | Pairwise | ~220 | Helpfulness, Honesty, Harmlessness |
| Auto-J Eval | Pairwise | ~4K | General quality comparison |

### Metrics
- **Agreement rate**: % matching human judgments (primary metric, target >75%)
- **Cohen's Kappa**: Inter-rater reliability accounting for chance agreement
- **Krippendorff's Alpha**: Consistency across repeated runs
- **Position bias rate**: % of verdicts that flip when A/B order is swapped (target <15%)
- **Verbosity bias rate**: tendency to prefer longer responses regardless of quality
- **Self-enhancement bias rate**: tendency to prefer Llama-family outputs

### Models to compare
1. Base Llama-3.1-8B-Instruct (prompted, no fine-tune) — baseline
2. SFT-only model — shows what supervised learning alone achieves
3. SFT + DPO model — our final model
4. GPT-4 as judge (API) — gold standard comparison

## DPO Data Construction Strategy

**Approach 1 (primary)**: Run SFT model on held-out validation set. Where the model disagrees with humans, use (correct_judgment, model's_wrong_judgment) as a DPO pair. Where the model is correct, skip.

**Approach 3 (if needed)**: Target position-biased judgments. Where the model flips its verdict when A/B are swapped, the consistent judgment is "chosen" and the position-biased one is "rejected."

## Phase-by-Phase Execution

### PHASE 1: Data Download & Exploration (current)
**Goal**: Download Preference Collection, validate schema, check distributions, verify sequence lengths fit 4096 tokens, create train/val split (95/5).

**Files to create**:
1. `data/download_datasets.py` — download Preference Collection from HuggingFace, save to `data/raw/`
2. `data/inspect_data.py` — EDA script that prints:
   - Total row count and schema verification
   - Distribution of orig_preference (A vs B balance)
   - Distribution of orig_score_A and orig_score_B
   - Sequence length distribution (tokenize `instruction` + `output` with Llama tokenizer)
   - Percentage of examples exceeding 4096 tokens
   - 3 random sample examples printed in full
3. `requirements.txt` — initial dependencies: datasets, transformers, tokenizers, pyyaml, numpy

**Important notes**:
- Use HuggingFace `datasets` library (Arrow format, memory-mapped)
- Save processed data in HuggingFace dataset format, NOT CSV
- Do NOT download the Feedback Collection — we are only using Preference Collection
- Do NOT download evaluation benchmarks yet — that is a later phase
- The train/val split should be 95/5 (10K validation examples is sufficient)
- Set random seed 42 for reproducibility

### PHASE 2: SFT Data Preparation
**Goal**: Tokenize the raw splits into training-ready format with proper chat template and label masking.

**Files to create**:
1. `data/prepare_sft_data.py` — processes raw splits into tokenized, label-masked datasets

**What the script does**:
1. Load raw train/val splits from `data/splits/`
2. Apply Llama 3.1 chat template to each example:
   - System prompt: "You are a fair evaluator language model."
   - User message: the `instruction` field (already contains task desc + responses + rubric)
   - Assistant message: the `output` field (comparative feedback + [RESULT] verdict)
3. Tokenize using `meta-llama/Llama-3.1-8B-Instruct` tokenizer
4. Build `labels` array: set all tokens from system prompt and user message to -100 (no loss computed). Only assistant response tokens keep their real token IDs.
5. Filter out any examples where total sequence length exceeds `max_seq_length` (default 4096). Log how many are dropped and the percentage.
6. Save processed dataset to `data/processed/` in HuggingFace Arrow format
7. Print summary stats: total examples before/after filtering, sequence length percentiles of processed data

**Output fields per example**:
- `input_ids`: full tokenized sequence (system + user + assistant)
- `attention_mask`: 1 for real tokens, 0 for padding
- `labels`: -100 for system/user tokens, real token IDs for assistant tokens

**Important notes**:
- Use `tokenizer.apply_chat_template()` for correct Llama 3.1 special tokens — do NOT manually construct the template
- The `instruction` field is ALREADY the fully assembled prompt — do NOT re-template from orig_* fields
- The `output` field is ALREADY the target — do NOT use orig_feedback_A, orig_feedback_B, or orig_scores
- Use `tokenizer.apply_chat_template()` with the conversation in the standard messages format: [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]
- To find where the assistant response starts for label masking: tokenize the conversation WITHOUT the assistant message, note that length, and mask everything up to that point as -100
- Process both train and val splits
- Add CLI args: --splits-dir, --output-dir, --max-seq-length, --tokenizer
- Requires HuggingFace token with Llama 3.1 access (huggingface-cli login)

### PHASE 3: SFT Training
(To be detailed when Phase 2 is complete)

### PHASE 4: DPO Data Construction
(To be detailed when Phase 3 is complete)

### PHASE 5: DPO Training
(To be detailed when Phase 4 is complete)

### PHASE 6: Evaluation & Bias Analysis
(To be detailed when Phase 5 is complete)

### PHASE 7: Serving & Deployment
(To be detailed when Phase 6 is complete)

### PHASE 8: Documentation & README
(To be detailed when Phase 7 is complete)

## Code Style Guidelines

- Python 3.10+
- Type hints on all function signatures
- Docstrings on all public functions (Google style)
- Use `logging` module, not print statements (except in inspect_data.py where prints are fine for EDA)
- Use `pathlib.Path` for file paths
- Use `argparse` for CLI arguments where appropriate
- Config files in YAML format
- All random operations seeded with 42
- Scripts should be runnable standalone: `python data/download_datasets.py`
