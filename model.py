"""Model plumbing on top of mlx-lm: loading, new-token embeddings, LoRA, saving."""

import os
import shutil
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import save_config, save_model

from data import SPECIAL_TOKENS

INIT_STD = 0.02  # transformers' initializer_range default
LORA_DROPOUT = 0.1
STAGING_DIR = ".saving"
WEIGHT_PATTERN = "model*.safetensors"  # what mlx_lm.load reads from a model folder
LORA_SCALE = 2.0  # upstream uses lora_alpha = 2 * rank


def resize_embeddings(model, vocab_size):
    """Grow the token embeddings (and untied lm_head); new rows start at the mean row."""

    def grow(weight):
        extra = vocab_size - weight.shape[0]
        mean = weight.astype(mx.float32).mean(axis=0, keepdims=True).astype(weight.dtype)
        return mx.concatenate([weight, mx.repeat(mean, extra, axis=0)])

    model.model.embed_tokens.weight = grow(model.model.embed_tokens.weight)
    if "lm_head" in model:
        model.lm_head.weight = grow(model.lm_head.weight)
    model.args.vocab_size = vocab_size


def reinitialize(model):
    """Random weights for pretraining from scratch (normal matrices, unit norms, zero biases)."""
    def init(name, param):
        if param.ndim >= 2:
            return (INIT_STD * mx.random.normal(param.shape)).astype(param.dtype)
        return mx.zeros_like(param) if name.endswith("bias") else mx.ones_like(param)

    model.update(tree_unflatten([(n, init(n, p)) for n, p in tree_flatten(model.parameters())]))


def load(model_name, *, pretrain=False):
    """Load an unquantized mlx-lm model and add the LLM-JEPA special tokens.

    Returns (model, Hugging Face tokenizer, config).
    """
    model, wrapper, config = mlx_lm.load(model_name, return_config=True)
    if "quantization" in config:
        raise ValueError(f"{model_name} is quantized; training needs full-precision weights.")
    if not hasattr(getattr(model, "model", None), "embed_tokens"):
        raise ValueError(f"Unsupported architecture {config.get('model_type')}: no model.embed_tokens.")
    if pretrain:
        # shortcut: downloads the pretrained weights only to discard them,
        # build from config alone if that download ever matters
        reinitialize(model)

    tokenizer = wrapper._tokenizer
    new_tokens = [token for token in SPECIAL_TOKENS if token not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_tokens(new_tokens, special_tokens=True)
        print(f"Added {len(new_tokens)} new special tokens")
    if len(tokenizer) > model.model.embed_tokens.weight.shape[0]:
        resize_embeddings(model, len(tokenizer))
        config = {**config, "vocab_size": len(tokenizer)}
    return model, tokenizer, config


def apply_lora(model, rank):
    """Freeze the model and add LoRA adapters to every linear layer of every block."""
    model.freeze()
    linear_to_lora_layers(
        model, len(model.layers), {"rank": rank, "scale": LORA_SCALE, "dropout": LORA_DROPOUT}
    )


def fuse_lora(model):
    """Swap every LoRA layer for its merged plain layer; returns the adapters it replaced."""
    adapters = [(name, m) for name, m in model.named_modules() if hasattr(m, "fuse")]
    if adapters:
        model.update_modules(tree_unflatten([(name, m.fuse()) for name, m in adapters]))
    return adapters


@contextmanager
def merged_adapters(model):
    """Run with merged layers, then put the same adapter modules back, even on failure."""
    adapters = fuse_lora(model)
    try:
        yield
    finally:
        if adapters:
            model.update_modules(tree_unflatten(adapters))


def _write_model(staging, model, tokenizer, config):
    with merged_adapters(model):
        save_model(staging, model)
    save_config(dict(config), staging / "config.json")
    tokenizer.save_pretrained(staging)


def _weights_last(files):
    return sorted(files, key=lambda path: (path.match(WEIGHT_PATTERN), path.name))


def save(output_dir, model, tokenizer, config, *, epoch=None, keep_epoch=False):
    """Write a complete model into a staging folder, then rename it into place.

    An interrupted save leaves the previous model untouched. The weight files are renamed last;
    for a model stored as several weight files, a crash between those renames can leave a mix.
    """
    output_dir = Path(output_dir)
    staging = output_dir / STAGING_DIR
    if staging.exists():
        shutil.rmtree(staging)
    _write_model(staging, model, tokenizer, config)
    staged = _weights_last(staging.iterdir())
    if keep_epoch:
        shutil.copytree(staging, output_dir / f"epoch-{epoch}", dirs_exist_ok=True)
    for path in staged:
        os.replace(path, output_dir / path.name)
    new_names = {path.name for path in staged}
    for stale in output_dir.glob(WEIGHT_PATTERN):
        if stale.name not in new_names:
            stale.unlink()
    staging.rmdir()
