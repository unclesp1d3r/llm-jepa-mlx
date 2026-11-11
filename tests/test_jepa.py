import mlx.core as mx
import mlx.optimizers as optim
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.llama import Model, ModelArgs

import jepa
import model as model_lib

VOCAB = 50


def tiny_model(tie=False):
    mx.random.seed(0)
    return Model(ModelArgs(
        model_type="llama", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=VOCAB,
        tie_word_embeddings=tie,
    ))


EXAMPLES = [
    {"input_ids": [1, 5, 6, 7, 8, 9, 2], "labels": [-100, -100, -100, 7, 8, 9, -100],
     "text_ids": [1, 5, 6, 2], "code_ids": [1, 7, 8, 9, 2]},
    {"input_ids": [1, 11, 12, 2], "labels": [-100, -100, 12, -100],
     "text_ids": [1, 11, 2], "code_ids": [1, 12, 2]},
]


def test_collate_right_pads_and_indexes_last_tokens():
    batch = jepa.collate(EXAMPLES, last_token=-2)
    assert batch["input_ids"].shape == (2, 7)
    assert batch["labels"][1].tolist() == [-100, -100, 12, -100, -100, -100, -100]
    assert batch["views"].shape == (4, 5)  # text rows then code rows
    assert batch["view_last"].tolist() == [2, 1, 3, 1]


def test_collate_without_views_for_regular_training():
    examples = [{k: e[k] for k in ("input_ids", "labels")} for e in EXAMPLES]
    assert "views" not in jepa.collate(examples, last_token=-1)


def test_view_embeddings_ignore_padding():
    model = tiny_model()
    batch = jepa.collate(EXAMPLES, last_token=-1)
    text, code = jepa.view_embeddings(model, batch)
    for row, example in enumerate(EXAMPLES):
        alone_text = model.model(mx.array([example["text_ids"]]))[0, -1]
        alone_code = model.model(mx.array([example["code_ids"]]))[0, -1]
        assert mx.allclose(text[row], alone_text, atol=1e-5)
        assert mx.allclose(code[row], alone_code, atol=1e-5)


def test_lm_loss_ignores_masked_labels():
    logits = mx.random.normal((1, 4, VOCAB))
    labels = mx.array([[-100, 3, -100, 7]])
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    # position t predicts label t+1
    expected = -(logp[0, 0, 3] + logp[0, 2, 7]) / 2
    assert jepa.lm_loss(logits, labels).item() == pytest.approx(expected.item(), abs=1e-5)


def test_lm_loss_is_finite_when_every_label_is_masked():
    loss = jepa.lm_loss(mx.random.normal((1, 3, VOCAB)), mx.full((1, 3), -100))
    assert loss.item() == 0.0


@pytest.mark.parametrize("kind,expected", [("cosine", 0.0), ("mse", 0.0), ("l2", 0.0)])
def test_jepa_loss_is_zero_for_identical_embeddings(kind, expected):
    emb = mx.random.normal((3, 8))
    assert jepa.jepa_loss(emb, emb, kind).item() == pytest.approx(expected, abs=1e-5)


def test_jepa_loss_values():
    text, code = mx.array([[1.0, 0.0], [0.0, 2.0]]), mx.array([[0.0, 1.0], [0.0, 1.0]])
    assert jepa.jepa_loss(text, code, "cosine").item() == pytest.approx(0.5)
    assert jepa.jepa_loss(text, code, "mse").item() == pytest.approx(0.75)
    assert jepa.jepa_loss(text, code, "l2").item() == pytest.approx((2**0.5 + 1) / 2)
    # row 0 is orthogonal to both codes, row 1 matches both: both rows are a coin flip
    assert jepa.jepa_loss(text, code, "infonce").item() == pytest.approx(0.693147, abs=1e-4)


def test_jepa_loss_rejects_unknown_kind():
    with pytest.raises(ValueError):
        jepa.jepa_loss(mx.ones((1, 2)), mx.ones((1, 2)), "nope")


def test_total_loss_combines_gamma_and_lambda():
    model = tiny_model()
    batch = jepa.collate(EXAMPLES, last_token=-1)
    total, lm, jp = jepa.total_loss(model, batch, gamma=0.5, lbd=2.0, kind="cosine", use_jepa=True)
    assert total.item() == pytest.approx(0.5 * lm.item() + 2.0 * jp.item(), rel=1e-5)
    regular, lm2, jp2 = jepa.total_loss(model, batch, gamma=0.5, lbd=2.0, kind="cosine", use_jepa=False)
    assert regular.item() == pytest.approx(0.5 * lm2.item(), rel=1e-5)
    assert jp2.item() == 0.0


def test_training_reduces_the_jepa_term():
    model = tiny_model()
    batch = jepa.collate(EXAMPLES, last_token=-1)
    step = jepa.make_train_step(model, optim.AdamW(learning_rate=1e-2, weight_decay=0.0),
                                gamma=1.0, lbd=1.0, kind="cosine", grad_accum=1)
    history = [step(batch, use_jepa=True) for _ in range(20)]
    assert history[-1]["jepa"] < history[0]["jepa"]
    assert history[-1]["lm"] < history[0]["lm"]


def test_grad_accum_only_updates_on_the_last_micro_step():
    model = tiny_model()
    batch = jepa.collate(EXAMPLES, last_token=-1)
    step = jepa.make_train_step(model, optim.AdamW(learning_rate=1e-2, weight_decay=0.0),
                                gamma=1.0, lbd=1.0, kind="cosine", grad_accum=2)
    before = model.model.norm.weight * 1
    step(batch, use_jepa=True)
    assert mx.array_equal(model.model.norm.weight, before)
    step(batch, use_jepa=True)
    assert not mx.array_equal(model.model.norm.weight, before)


def test_flush_applies_a_partial_accumulation_group():
    model = tiny_model()
    batch = jepa.collate(EXAMPLES, last_token=-1)
    step = jepa.make_train_step(model, optim.AdamW(learning_rate=1e-2, weight_decay=0.0),
                                gamma=1.0, lbd=1.0, kind="cosine", grad_accum=4)
    before = model.model.norm.weight * 1
    step(batch, use_jepa=True, flush=True)
    assert not mx.array_equal(model.model.norm.weight, before)


@pytest.mark.parametrize("tie", [False, True])
def test_resize_embeddings_adds_mean_rows(tie):
    model = tiny_model(tie)
    old = model.model.embed_tokens.weight * 1
    model_lib.resize_embeddings(model, VOCAB + 3)
    new = model.model.embed_tokens.weight
    assert new.shape == (VOCAB + 3, 32)
    assert mx.array_equal(new[:VOCAB], old)
    assert mx.allclose(new[-1], old.mean(axis=0), atol=1e-6)
    assert model(mx.array([[VOCAB + 2, 1]])).shape == (1, 2, VOCAB + 3)


def test_lora_trains_only_adapters_and_fuses_back_to_plain_weights():
    model = tiny_model()
    names = {name for name, _ in tree_flatten(model.parameters())}
    model_lib.apply_lora(model, rank=4)
    trainable = [name for name, _ in tree_flatten(model.trainable_parameters())]
    assert trainable and all("lora_" in name for name in trainable)

    batch = jepa.collate(EXAMPLES, last_token=-1)
    step = jepa.make_train_step(model, optim.AdamW(learning_rate=1e-2, weight_decay=0.0),
                                gamma=1.0, lbd=1.0, kind="cosine", grad_accum=1)
    for _ in range(3):
        step(batch, use_jepa=True)
    model.eval()
    ids = mx.array([[1, 5, 6]])
    # CPU: float32 matmul on the Metal GPU is only good to ~3e-3 here
    with mx.stream(mx.cpu):
        tuned = model(ids)
        mx.eval(tuned)
        model_lib.fuse_lora(model)
        fused = model(ids)
        mx.eval(fused)
    assert {name for name, _ in tree_flatten(model.parameters())} == names
    assert mx.allclose(fused, tuned, atol=1e-5)


def test_reinitialize_changes_weights_and_resets_norms():
    model = tiny_model()
    model.model.norm.weight = model.model.norm.weight * 3
    old = model.model.embed_tokens.weight * 1
    model_lib.reinitialize(model, seed=1)
    assert not mx.array_equal(model.model.embed_tokens.weight, old)
    assert mx.array_equal(model.model.norm.weight, mx.ones(32))
