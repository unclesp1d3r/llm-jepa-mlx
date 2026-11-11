import json
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import mlx_lm
import pytest
from mlx.utils import tree_flatten
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

import jepa
import model as model_lib
from tests.test_jepa import EXAMPLES, VOCAB, tiny_model

TOKENIZER_WORDS = 44  # fewer than VOCAB, so adding the special tokens grows the vocabulary
TINY_CONFIG = {
    "model_type": "llama", "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 64,
    "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-5,
    "vocab_size": VOCAB, "tie_word_embeddings": False,
}
STALE_SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")


def tiny_tokenizer():
    vocab = {"<unk>": 0, **{f"w{i}": i for i in range(1, TOKENIZER_WORDS)}}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")


def snapshot(folder):
    return {p.name: p.read_bytes() for p in Path(folder).iterdir() if p.is_file()}


def weights_of(model):
    return {name: array for name, array in tree_flatten(model.parameters())}


def assert_same_weights(first, second):
    a, b = weights_of(first), weights_of(second)
    assert a.keys() == b.keys()
    assert all(mx.array_equal(a[name], b[name]) for name in a)


def perturbed(model):
    model.model.norm.weight = model.model.norm.weight + 1.0
    return model


def raise_disk_error(*args, **kwargs):
    raise OSError("disk full")


def test_save_leaves_unrelated_files_in_the_output_folder_alone(tmp_path):
    (tmp_path / "notes.txt").write_text("keep me")
    model_lib.save(tmp_path, tiny_model(), tiny_tokenizer(), TINY_CONFIG)
    assert (tmp_path / "notes.txt").read_text() == "keep me"


def test_interrupted_save_leaves_the_previous_model_byte_for_byte_unchanged(tmp_path, monkeypatch):
    model_lib.save(tmp_path, tiny_model(), tiny_tokenizer(), TINY_CONFIG)
    before = snapshot(tmp_path)
    monkeypatch.setattr(model_lib, "save_config", raise_disk_error)
    with pytest.raises(OSError):
        model_lib.save(tmp_path, perturbed(tiny_model()), tiny_tokenizer(), TINY_CONFIG)
    assert snapshot(tmp_path) == before
    assert_same_weights(mlx_lm.load(str(tmp_path))[0], tiny_model())


def test_next_save_removes_the_staging_folder_an_interrupted_save_left(tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(model_lib, "save_config", raise_disk_error)
        with pytest.raises(OSError):
            model_lib.save(tmp_path, tiny_model(), tiny_tokenizer(), TINY_CONFIG)
    assert len(list(tmp_path.iterdir())) == 1  # only the leftover staging folder
    (next(tmp_path.iterdir()) / "leftover.bin").write_bytes(b"from the interrupted save")
    model_lib.save(tmp_path, tiny_model(), tiny_tokenizer(), TINY_CONFIG)
    assert not any(p.is_dir() for p in tmp_path.iterdir())
    assert not (tmp_path / "leftover.bin").exists()
    assert_same_weights(mlx_lm.load(str(tmp_path))[0], tiny_model())


def test_save_moves_the_weight_file_into_place_after_every_other_file(tmp_path, monkeypatch):
    moved = []
    real_replace = model_lib.os.replace

    def record_then_replace(source, destination):
        moved.append(Path(destination).name)
        real_replace(source, destination)

    monkeypatch.setattr(model_lib.os, "replace", record_then_replace)
    model_lib.save(tmp_path, tiny_model(), tiny_tokenizer(), TINY_CONFIG)
    assert len(moved) > 1
    assert moved[-1] == "model.safetensors"


def test_save_removes_stale_weight_files_it_replaced_and_keeps_other_files(tmp_path):
    for name in STALE_SHARDS:
        (tmp_path / name).write_bytes(b"stale")
    (tmp_path / "notes.txt").write_text("keep me")
    model_lib.save(tmp_path, tiny_model(), tiny_tokenizer(), TINY_CONFIG)
    assert not any((tmp_path / name).exists() for name in STALE_SHARDS)
    assert (tmp_path / "notes.txt").read_text() == "keep me"
    assert_same_weights(mlx_lm.load(str(tmp_path))[0], tiny_model())


def test_save_that_fails_before_its_renames_keeps_stale_weight_files(tmp_path, monkeypatch):
    for name in STALE_SHARDS:
        (tmp_path / name).write_bytes(b"stale")
    monkeypatch.setattr(model_lib, "save_config", raise_disk_error)
    with pytest.raises(OSError):
        model_lib.save(tmp_path, tiny_model(), tiny_tokenizer(), TINY_CONFIG)
    assert all((tmp_path / name).read_bytes() == b"stale" for name in STALE_SHARDS)


def test_keep_epoch_copies_each_epoch_next_to_the_latest_model(tmp_path):
    for epoch, model in ((1, tiny_model()), (2, perturbed(tiny_model()))):
        model_lib.save(tmp_path, model, tiny_tokenizer(), TINY_CONFIG, epoch=epoch, keep_epoch=True)
    assert_same_weights(mlx_lm.load(str(tmp_path / "epoch-1"))[0], tiny_model())
    assert_same_weights(mlx_lm.load(str(tmp_path / "epoch-2"))[0], perturbed(tiny_model()))
    assert_same_weights(mlx_lm.load(str(tmp_path))[0], perturbed(tiny_model()))


def test_save_without_keep_epoch_creates_no_epoch_folder(tmp_path):
    model_lib.save(tmp_path, tiny_model(), tiny_tokenizer(), TINY_CONFIG, epoch=1)
    assert not list(tmp_path.glob("epoch-*"))


def test_model_with_added_tokens_reloads_with_the_enlarged_vocabulary(tmp_path):
    source, out = tmp_path / "source", tmp_path / "out"
    model_lib.save(source, tiny_model(), tiny_tokenizer(), TINY_CONFIG)
    model, tokenizer, config = model_lib.load(str(source))
    assert len(tokenizer) > VOCAB
    model_lib.save(out, model, tokenizer, config)

    reloaded, reloaded_tokenizer, _ = model_lib.load(str(out))
    highest = len(reloaded_tokenizer) - 1
    logits = reloaded(mx.array([[1, highest]]))
    assert logits.shape == (1, 2, len(tokenizer))
    assert json.loads((out / "config.json").read_text())["vocab_size"] == len(tokenizer)


def trained_lora_model(steps=3):
    model = tiny_model()
    model_lib.apply_lora(model, rank=4)
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    step = jepa.make_train_step(model, optimizer, gamma=1.0, lbd=1.0, kind="cosine", grad_accum=1)
    batch = jepa.collate(EXAMPLES, last_token=-1)
    for _ in range(steps):
        step(batch, use_jepa=True)
    return model, step, batch


def test_lora_mid_run_save_has_no_adapter_entries_and_reloads_to_the_same_output(tmp_path):
    model, _, _ = trained_lora_model()
    model_lib.save(tmp_path, model, tiny_tokenizer(), TINY_CONFIG)
    saved = mx.load(str(tmp_path / "model.safetensors"))
    assert not any("lora" in name for name in saved)
    reloaded, _ = mlx_lm.load(str(tmp_path))
    model.eval()
    ids = mx.array([[1, 5, 6]])
    with mx.stream(mx.cpu):
        expected, actual = model(ids), reloaded(ids)
        mx.eval(expected, actual)
    # the merged weights were computed on the GPU during the save, whose float32 matmul is ~3e-3
    assert mx.allclose(actual, expected, atol=3e-3)


def test_lora_training_continues_with_the_same_adapters_after_a_mid_run_save(tmp_path):
    model, step, batch = trained_lora_model()
    adapters = {name: module for name, module in model.named_modules() if hasattr(module, "fuse")}
    names = [name for name, _ in tree_flatten(model.trainable_parameters())]

    model_lib.save(tmp_path, model, tiny_tokenizer(), TINY_CONFIG)

    assert [n for n, _ in tree_flatten(model.trainable_parameters())] == names
    assert all(dict(model.named_modules())[name] is module for name, module in adapters.items())
    before = model.layers[0].self_attn.q_proj.lora_b * 1
    step(batch, use_jepa=True)
    assert not mx.array_equal(model.layers[0].self_attn.q_proj.lora_b, before)


def test_lora_save_that_raises_puts_the_adapters_back(tmp_path, monkeypatch):
    model, _, _ = trained_lora_model()
    adapters = {name: module for name, module in model.named_modules() if hasattr(module, "fuse")}
    monkeypatch.setattr(model_lib, "save_config", raise_disk_error)
    with pytest.raises(OSError):
        model_lib.save(tmp_path, model, tiny_tokenizer(), TINY_CONFIG)
    assert all(dict(model.named_modules())[name] is module for name, module in adapters.items())
