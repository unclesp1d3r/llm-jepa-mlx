# AGENTS.md

MLX port of LLM-JEPA (arXiv:2509.14252). The PyTorch reference is
<https://github.com/rbalestr-lab/llm-jepa> (`finetune.py`, `evaluate.py`). README.md covers usage,
flags, and the list of differences from upstream.

## Commands

```sh
uv sync                                   # install (Python 3.12-3.13, mlx-lm)
uv run pytest                             # all tests, offline, a few seconds
uv run pytest tests/test_jepa.py -k lora  # one test
uv run python finetune.py --help
uv run python evaluate.py --help
```

There is no linter or formatter configured. Lines are kept under 110 columns.

## Goal: behave like upstream

The point of this repo is to compute what the PyTorch code computes. When changing behavior, read
the matching upstream function first and keep flag names and semantics. Anything that deviates on
purpose is listed under "Differences from the PyTorch original" in README.md; add to that list when
you add a deviation.

Two metrics deliberately do not match upstream, by the repo owner's decision, because upstream's
checks were measured wrong on the real test sets. Do not restore upstream's behavior:

- NQ-Open: a reference answer must appear in the output (upstream checks the reverse).
- Spider: queries run through Python's `sqlite3` on a read-only connection that permits only
  reading, with double-quoted text values accepted. A query that fails is wrong. Row order counts
  only when the reference has `ORDER BY`. A reference that cannot run makes its example wrong and is
  reported. Never route model-written SQL through the `sqlite3` command-line tool: it executes
  dot-commands such as `.shell`.

These look like bugs but are upstream behavior kept for parity. Do not "fix" them silently:

- The chat template text is tokenized with `add_special_tokens=True`, so Llama inputs start with
  two `<|begin_of_text|>` tokens. `--unmask_assistant_special_tokens` turns that off.
- Default labels cover only the assistant's content, not the closing `<|eot_id|>`.
- Assistant content is located by comparing decoded token strings, not ids, and only the first
  occurrence is unmasked.
- `get_assistant_messages` relabels the assistant turn as `user` for Gemma.
- GSM8K scoring treats two missing `#### answer` lines as equal.

## Architecture

Flat modules imported by name (`pythonpath = ["."]` in pyproject.toml), no package.

- `data.py`: pure Python, no MLX. `prepare_example` returns unpadded token lists:
  `input_ids`, `labels`, and for JEPA runs `text_ids` and `code_ids`.
- `jepa.py`: `collate` pads a batch, `total_loss` combines the two terms, `make_train_step`
  owns gradient accumulation, clipping and the optimizer update.
- `model.py`: everything that touches mlx-lm internals.
- `finetune.py`, `evaluate.py`: argument parsing and orchestration only.

Per training step there are two forward passes: `model(input_ids)` for logits and the next-token
loss, and `model.model(views)` for hidden states. `views` stacks all Text rows and then all Code
rows, so the first half of the batch is Text. `view_last` is `len(view) + last_token`.

## Things that are easy to get wrong

- **Right padding only.** Nothing passes an attention mask. Padding is safe because the model is
  causal and every position read (labels, `view_last`) is before the padding. Left padding, or
  reading a position past a row's length, breaks this silently.
- **Views instead of a 4D mask.** Encoding Text and Code as separate rows equals upstream's
  block-causal mask only for models with relative (rotary) positions.
  `test_view_embeddings_ignore_padding` guards the padding half of that claim.
- **mlx-lm internals.** The code relies on `model.model(ids)` returning post-norm hidden states,
  `model.model.embed_tokens`, an optional `model.lm_head` (absent when weights are tied), and
  `TokenizerWrapper._tokenizer` for the Hugging Face tokenizer. Check
  `.venv/lib/python*/site-packages/mlx_lm/models/<arch>.py` before supporting a new architecture
  or bumping mlx-lm.
- **Dataset detection is by file name prefix** (`hellaswag`, `gsm8k`, `spider`, `nq_open`) on the
  basename, in both `finetune.py` and `evaluate.py`.
- **Embedding resize changes `vocab_size`.** `model.load` returns an updated config; that config,
  not the original, must be the one saved.
- **Saves go through a staging folder.** `model.save` writes a complete model into `.saving` inside
  the output folder, then renames the files into place, weight files last. It never deletes the
  output folder; the only files it removes are `model*.safetensors` files the new save replaced,
  because `mlx_lm.load` loads every file matching that pattern.
- **LoRA saves swap, then restore.** Each save temporarily replaces the adapters with merged layers
  and puts the same adapter modules back afterwards, also on failure, so a saved model is always
  plain weights and training keeps its adapters and optimizer state.
- **MLX is lazy.** The training step calls `mx.eval` on parameters, optimizer state and pending
  gradients every step. Dropping that lets the graph grow without bound.
- **bf16.** Model weights are bf16. Losses and the gradient norm are computed in float32 on purpose.
- **GPU float32 matmul is only accurate to about 3e-3 on the development machine.** Tests that
  compare two numerically different routes to the same result run under `with mx.stream(mx.cpu):`.

## Tests

Tests build a two-layer random Llama (`tiny_model` in `tests/test_jepa.py`) and a character-level
fake tokenizer (`CharTokenizer` in `tests/test_data.py`), so they need no network and no model
download. Save and reload tests build a small real tokenizer in-process (`tests/test_model.py`), and
Spider tests build a small SQLite database in a temporary folder. Write the test first, then the change. New behavior in `data.py` or `jepa.py` needs a test.

## Running real training

Do not start a real fine-tune or evaluation to check a change unless asked. It downloads a 2.5 GB
model, a full fine-tune of a 1B model peaks near 11.3 GB, and on a machine without that much free
memory it swaps and slows by an order of magnitude. LoRA peaks near 6 GB. For a quick end-to-end
check, use `--lora` on the first 200 lines of `datasets/synth_train.jsonl` with `--num_epochs 1`
and `evaluate.py --max_examples 40`. The model is saved to `--output_dir` after every epoch.

No full-length run has been done with this code, so there is no accuracy baseline to compare to.
