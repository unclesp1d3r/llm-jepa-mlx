# LLM-JEPA on MLX

An Apple Silicon port of the reference code for
[LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures](https://arxiv.org/abs/2509.14252)
(Huang, LeCun, Balestriero, 2025). The original is PyTorch + Hugging Face and targets multi-GPU
NVIDIA machines: <https://github.com/rbalestr-lab/llm-jepa>. This version runs on a Mac with
[MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm).

It is not a line-for-line port. Every place it behaves differently from upstream is listed under
[Differences from the PyTorch original](#differences-from-the-pytorch-original); two of those
change accuracy numbers.

## What LLM-JEPA is

Ordinary fine-tuning teaches a model to predict the next token. LLM-JEPA adds a second objective in
embedding space. Many datasets pair two views of the same thing, such as a description in English
(Text) and the regular expression it describes (Code). The model encodes each view separately, and
the extra loss pulls the two embeddings together:

```
loss = gamma * L_LLM + lbd * d(Pred(Enc(Text)), Enc(Code))
```

- `L_LLM` is the usual next-token cross-entropy on the answer.
- `Enc(x)` is the last-layer hidden state at the last token of `x`.
- `Pred` reuses the model itself: `k` special `<|predictor_i|>` tokens are appended to Text and the
  embedding is read at the last of them. With `k = 0` the predictor is the identity.
- `d` is one minus cosine similarity by default.

The paper reports higher accuracy than plain fine-tuning across several models and datasets.

## Status

Read this before trusting any number that comes out of it.

| | |
|---|---|
| Unit tests | 84 pass, offline, in a few seconds |
| LoRA + JEPA end to end | Ran on Llama-3.2-1B: trained, saved, reloaded, evaluated |
| Token layout | Checked against upstream on a real example |
| Full fine-tuning | Three steps ran and the loss fell; a complete run was never finished |
| Paper accuracy | **Not reproduced.** No full-length training run has been done with this code |
| Other model families | Gemma, Qwen and OLMo-2 paths are written but untested |

## Requirements

- An Apple Silicon Mac
- [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 or 3.13 and the dependencies)
- About 6 GB of free memory for LoRA on a 1B model, and more than 11.5 GB for full fine-tuning (see Memory and speed)
- Free disk of about twice the model's size, used while a save is being written

## Quick start

```sh
uv sync
uv run pytest

# LLM-JEPA fine-tuning: one predictor token, lambda = 1
uv run python finetune.py --train_file datasets/synth_train.jsonl --output_dir ./fine-tuned \
  --lora --num_epochs 4 --predictors 1 --lbd 1.0 --last_token -2 --finetune_seed 82

# Baseline for comparison: next-token loss only
uv run python finetune.py --train_file datasets/synth_train.jsonl --output_dir ./fine-tuned \
  --lora --num_epochs 4 --regular --finetune_seed 82

# Accuracy on the test split
uv run python evaluate.py --model_name ./fine-tuned \
  --original_model_name mlx-community/Llama-3.2-1B-Instruct-bf16 \
  --input_file datasets/synth_test.jsonl --output_file eval.jsonl
```

The first run downloads `mlx-community/Llama-3.2-1B-Instruct-bf16` (about 2.5 GB). The model is
saved to `--output_dir` at the end of every epoch, replacing the previous save. Nothing else in that
folder is deleted, except model weight files that the new save has replaced.

`finetune.py` prints three numbers per step. `llm_loss` is the next-token loss, `jepa_loss` is the
embedding distance, and `loss` is the weighted sum being minimized. A falling loss shows the
optimizer is working; it does not show the model is more accurate. Accuracy comes only from
`evaluate.py`, which prints a line such as `Success Rate: ./fine-tuned, 0.57`.

### Running the paper's protocol

The paper trains every configuration with five seeds (82, 23, 37, 84, 4) for four epochs and
reports mean and standard deviation. `run.sh` does that for the baseline and for LLM-JEPA and
appends the `Success Rate:` lines to `output.txt`:

```sh
./run.sh [dataset] [learning_rate] [lbd] [predictors] [extra finetune.py flags...]
./run.sh synth 2e-5 1.0 1 --lora --lora_rank 512
```

Set `MODEL=` and `LAST_TOKEN=` in the environment to change the model.

## Options

### finetune.py

| Flag | Default | Meaning |
|---|---|---|
| `--train_file` | required | Training JSONL |
| `--model_name` | `mlx-community/Llama-3.2-1B-Instruct-bf16` | Unquantized mlx-lm repo or local path |
| `--output_dir` | `./fine-tuned` | Where the latest model is saved, at the end of every epoch |
| `--keep_epochs` | off | Also keep each epoch's model in an `epoch-N` folder inside `--output_dir` |
| `--regular` | off | Baseline: no JEPA term |
| `--lbd` | 0.1 | λ, weight of the JEPA loss |
| `--gamma` | 1.0 | γ, weight of the next-token loss |
| `--predictors` | 0 | k, number of predictor tokens (0 to 10) |
| `--last_token` | -1 | Which token's embedding to read, counted from the end of each view |
| `--lora`, `--lora_rank` | off, 16 | LoRA on every linear layer of every block; merged into the weights on save |
| `--learning_rate` | 2e-5 | Peak rate, decays linearly to zero |
| `--num_epochs` | 3 | |
| `--batch_size`, `--grad_accum` | 4, 4 | Examples per update = their product |
| `--max_length` | 512 | Truncation length in tokens |
| `--finetune_seed` | 42 | |
| `--jepa_ratio` | -1 | When above 0, apply the JEPA loss to this fraction of batches (the paper's loss dropout, `LD = 1 - ratio`) |
| `--jepa_l2`, `--jepa_mse`, `--infonce` | off | Replace the cosine distance (ablations) |
| `--front_pred` | off | Put predictor tokens before Text instead of after |
| `--reverse_pred` | off | Let Code predict Text |
| `--pretrain` | off | Start from random weights |
| `--plain` | off | No chat template; join question and answer with `<|perception|>` |
| `--train_all` | off | Next-token loss on every token |
| `--unmask_assistant_special_tokens` | off | Also train on the answer's closing tokens; pass it to `evaluate.py` too |

`--last_token` depends on the chat template, because it has to land on the last predictor token
(or the last content token). Upstream uses `-2` for Llama and Gemma, `-1` for OLMo-2 and DeepSeek,
and `-3` for Qwen.

### evaluate.py

| Flag | Meaning |
|---|---|
| `--model_name` | Fine-tuned model directory, or any mlx-lm model |
| `--original_model_name` | The base model's name, used to pick the chat layout (needed for Gemma) |
| `--input_file` | Test JSONL |
| `--output_file` | Per-example results as JSONL: generated text, expected text, correct or not |
| `--max_examples` | Evaluate only the first N |
| `--max_new_tokens` | Generation limit, default 128 |
| `--startswith` | Also report how often the output starts with the expected answer |
| `--similarity` | Also report the Text/Code cosine similarity for right and wrong answers |
| `--spider_path` | Directory of Spider sqlite databases |
| `--plain`, `--unmask_assistant_special_tokens` | Must match training |

## Datasets

`datasets/` is copied from the upstream repository, which preprocessed them from the original
publications. Each line is a three-message chat: system prompt, user turn (Text), assistant turn (Code).

| Files | Task | Metric |
|---|---|---|
| `synth_*`, `turk_*` | English to regular expression | Exact match |
| `gsm8k_*` | Math word problems | Final answer after `####` matches |
| `spider_*` | English to SQL | The query runs and returns the reference's rows; order counts only when the reference has `ORDER BY` |
| `nq_open_*` | Open-domain QA | A reference answer appears in the output |
| `hellaswag_*` | Sentence completion, multiple choice | Most likely of A/B/C/D |
| `paraphrase_train` | Pretraining only | |
| `rotten_tomatoes_*`, `yelp_*` | Sentiment labels after pretraining | Use `--startswith` |

The metric is chosen from the file name, so keep the prefixes. Spider also needs `spider_data.zip`
from the upstream repo unzipped, with `--spider_path spider_data/database`. Model-written SQL runs
on a read-only connection that permits only reading, so it cannot change the databases or run
commands.

## Memory and speed

Measured with Llama-3.2-1B in bf16 on `synth`, on a 24 GB M5, at `--batch_size 4 --grad_accum 1`:

| Mode | Peak memory | Time per step |
|---|---|---|
| LoRA, rank 16 | about 6 GB | about 1.1 s |
| Full fine-tuning | about 11.3 GB | about 4 s when it fits in RAM |

The default `--grad_accum 4` holds one more copy of the gradients between steps, about 2.5 GB more
for full fine-tuning; that setting was not measured. Full fine-tuning that spills into swap runs tens
of times slower, so check free memory first. One epoch of `synth` is 2,000 steps at batch size 4.

A save writes the complete new model before replacing the old one, so it needs free disk of about
twice the model's size. `--keep_epochs` adds one model per epoch.

## Differences from the PyTorch original

This is not a line-for-line port. Everything below is a place where this code does something
different from <https://github.com/rbalestr-lab/llm-jepa>. Unless a line says otherwise, the
difference is deliberate.

### Scoring: numbers that are not comparable with upstream

Scores on NQ-Open and Spider are not comparable with upstream's scripts or with the numbers the
paper reports. Upstream's checks were measured on the real test sets and found wrong.

- **NQ-Open.** A reference answer must appear in the output, as the paper states. Upstream checks
  the reverse, so on the 1,769 test questions an empty output scores 100% there and
  `The answer is <right answer>.` scores 0%.
- **Spider: a query must run and return the reference's rows.** Upstream compares what the
  `sqlite3` command-line tool prints, and a failed query prints nothing, so it matches any reference
  that also prints nothing. With `sqlite3` 3.54.1 the text `this is not sql` scores correct on 365
  of 1,447 test examples there. Here it scores correct on none, and so does an output that is
  empty or only a comment.
- **Spider: double-quoted text values are read as text.** 326 references use them. Upstream leaves
  this to the installed `sqlite3` tool, and newer versions reject them.
- **Spider: row order counts only when the reference has an `ORDER BY`** (336 of 1,447 do).
  Upstream compares printed output line by line.
- **Spider: a reference that cannot run makes its example wrong** and is named in the report. One
  test reference (`document_management`) is malformed. Upstream scores any failing output correct
  on it.
- **Spider: a missing, empty or corrupt database file is an error.** Upstream scores every output
  correct when the database path is wrong.
- **Spider: model-written SQL runs on a read-only connection** through Python's SQLite library.
  Upstream passes it to the `sqlite3` command-line tool, which also executes dot-commands such as
  `.shell`. The command-line tool is not needed here.
- **Datasets are recognised by file name,** wherever the file is. Upstream matches the start of the
  path as given, so its scripts must run from the dataset folder.
- **HellaSwag prompts are cut at `--max_length`.** Upstream cuts them at `--max_new_tokens`.
- **Generation stops at any of the model's end-of-sequence tokens.** Upstream stops at the
  tokenizer's single `eos_token_id`.

### Training

- **No 4D additive attention mask.** Upstream packs Text and Code into one row behind a
  block-causal mask because it pads every row to `--max_length`. Here each view is its own row,
  padded only to the longest view in the batch. The embeddings are the same and the attention work
  is smaller. `--additive_mask` and `--same_flop` do not exist.
- **Hand-written training loop.** It follows the Hugging Face `Trainer` defaults upstream relies on:
  AdamW without weight decay, linear decay to zero, gradient clipping at 1.0, and a partial
  accumulation group applied at the end of each epoch. Shuffle order differs, so runs are not
  bit-for-bit comparable. Losses are logged per batch, not per optimizer update.
- **Effective batch size.** The paper ran 8 GPUs × batch 4 × 4 accumulation steps, 128 examples per
  update. The defaults here give 16. Use `--grad_accum 32` to match.
- **`--jepa_ratio` always applies.** Upstream only honours it together with `--additive_mask`.
- **`--gamma` also scales `--regular` runs.** Upstream's baseline ignores it.
- **Training checks itself.** It stops at a loss that is not a finite number, and before the first
  step it checks that every embedding position lies inside its view. Upstream has neither check.
- **New-token embeddings start at the exact mean embedding.** Upstream's resize adds a very small
  random offset; this was recalled, not checked against the transformers source.
- **Gemma is recognised by `gemma` anywhere in the model name,** so MLX community repos work.
  Upstream looks for `google/gemma`.

### Saving

- **The latest model replaces the previous one at the end of every epoch.** Upstream keeps a
  `checkpoint-N` folder per epoch; `--keep_epochs` does the same here with `epoch-N` folders.
- **The output folder is never deleted.** Upstream deletes it at the start of a run. Here a save
  removes only model weight files it has replaced.
- **A save cannot be resumed.** Optimizer state is not saved; upstream's checkpoints include it.

### Evaluation output

- **`evaluate.py --output_file`** writes per-example results as each example is scored. Upstream
  leaves the file empty.
- **`--similarity`** uses the last token of the last layer only. Upstream also offers mean and
  first-token pooling and a choice of layer.

### Not ported

- `stp.py` (Semantic Tube Prediction, a different paper) and `finetune8bh200.py`.
- Multi-GPU training, FLOP tracking, gradient checkpointing, and 4-bit or 8-bit loading.
- The t-SNE dump, train/test splitting, and the `--eval_file`, `--data_file` and `--debug` flags.
- OpenELM and phi support.

## Known caveats

- **bf16 weights.** Like upstream, full fine-tuning updates bf16 weights directly, which rounds
  away very small updates. LoRA adapters are float32 and are not affected.
- **GPU matmul precision.** On the development machine (M5, macOS 27.2, MLX 0.32.4) a float32
  matrix multiply on the GPU agreed with a float64 reference only to about 3e-3, against 1e-7 on
  the CPU. Whether that is expected for this MLX version is unknown. It may affect how closely
  results track the PyTorch code.
- **Stopping.** By default the loss covers the answer text but not the end-of-turn token, as
  upstream does. A lightly trained model can therefore run on past the answer and fail exact match.
  `--unmask_assistant_special_tokens` trains on the closing token too.
- **New tokens.** The 11 added special tokens start from the mean embedding. Under `--lora` the
  embeddings are frozen, so with `--predictors` above 1 the predictor tokens differ only by position.
- **Quantized models** are rejected; training needs full-precision weights.
- **Spider limits.** One test reference (`document_management`) is malformed and cannot run, so that
  example scores wrong for every model and is named in the report. On the 63 examples whose
  reference returns no rows, a valid but wrong query that also returns no rows still scores correct.
  A generated query that never finishes stalls the evaluation; there is no time limit.
- **Interrupted saves.** A save interrupted while it is being written leaves the previous model in
  place. For a model stored as several weight files (over 5 GB), a crash during the final renames
  can leave a mix of old and new weights.
- **A saved model is not a resumable run.** Optimizer state is not saved.

## Layout

| File | Contents |
|---|---|
| `finetune.py` | Training CLI and loop |
| `evaluate.py` | Generation and metrics |
| `jepa.py` | Losses, batching, the training step |
| `data.py` | Chat formatting, label masking, Text/Code views |
| `model.py` | mlx-lm loading, new-token embeddings, LoRA, saving |
| `tests/` | Unit tests on a tiny random model and tokenizer; no downloads |
| `run.sh` | Five-seed baseline versus LLM-JEPA driver |

## Credit and license

The method, the original implementation and the preprocessed datasets are by the paper's authors.
This port is distributed under the same Apache-2.0 license (see `LICENSE`).

```bibtex
@article{huang2025llmjepa,
  title   = {LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures},
  author  = {Huang, Hai and LeCun, Yann and Balestriero, Randall},
  journal = {arXiv preprint arXiv:2509.14252},
  year    = {2025}
}
```
