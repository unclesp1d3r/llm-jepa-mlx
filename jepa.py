"""LLM-JEPA objective: next-token loss plus an embedding-space loss between two views."""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from data import IGNORE

INFONCE_TEMPERATURE = 0.07
MAX_GRAD_NORM = 1.0  # transformers.Trainer default
PAD_ID = 0  # never attended to by real tokens (right padding + causal mask)


def _pad(rows, value):
    width = max(len(row) for row in rows)
    return mx.array([row + [value] * (width - len(row)) for row in rows])


def collate(examples, last_token):
    """Right-pad a batch. Views are stacked as [Text rows..., Code rows...]."""
    batch = {
        "input_ids": _pad([e["input_ids"] for e in examples], PAD_ID),
        "labels": _pad([e["labels"] for e in examples], IGNORE),
    }
    if "text_ids" not in examples[0]:
        return batch
    views = [e["text_ids"] for e in examples] + [e["code_ids"] for e in examples]
    return {
        **batch,
        "views": _pad(views, PAD_ID),
        "view_last": mx.array([len(view) + last_token for view in views]),
    }


def view_embeddings(model, batch):
    """Last-layer hidden state at each view's last token: (Enc(Text), Enc(Code)).

    Upstream packs both views into one row behind a block-causal 4D mask because it pads
    every row to max_length. Rows padded to the batch's longest view give the same
    embeddings (rotary positions are relative) for less attention work.
    """
    views = batch["views"]
    hidden = model.model(views)
    last = hidden[mx.arange(views.shape[0]), batch["view_last"]].astype(mx.float32)
    half = views.shape[0] // 2
    return last[:half], last[half:]


def lm_loss(logits, labels):
    logits = logits[:, :-1].astype(mx.float32)
    targets = labels[:, 1:]
    keep = targets != IGNORE
    losses = nn.losses.cross_entropy(logits, mx.where(keep, targets, 0), reduction="none")
    return (losses * keep).sum() / mx.maximum(keep.sum(), 1)


def _normalize(x):
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def jepa_loss(text, code, kind="cosine"):
    if kind == "cosine":
        return 1.0 - (_normalize(text) * _normalize(code)).sum(axis=-1).mean()
    if kind == "l2":
        return mx.linalg.norm(text - code, axis=-1).mean()
    if kind == "mse":
        return ((text - code) ** 2).mean()
    if kind == "infonce":
        logits = (_normalize(text) @ _normalize(code).T) / INFONCE_TEMPERATURE
        return nn.losses.cross_entropy(logits, mx.arange(logits.shape[0]), reduction="mean")
    raise ValueError(f"Unknown JEPA loss: {kind}")


def total_loss(model, batch, *, gamma, lbd, kind, use_jepa):
    lm = lm_loss(model(batch["input_ids"]), batch["labels"])
    jepa = jepa_loss(*view_embeddings(model, batch), kind) if use_jepa else mx.array(0.0)
    return gamma * lm + lbd * jepa, lm, jepa


def _clip(grads, max_norm):
    # float32 accumulation: a bf16 sum over a billion squares is not trustworthy
    squares = [(g.astype(mx.float32) ** 2).sum() for _, g in tree_flatten(grads)]
    scale = mx.minimum(max_norm / (mx.sqrt(sum(squares)) + 1e-6), 1.0)
    return tree_map(lambda g: g * scale.astype(g.dtype), grads)


def make_train_step(model, optimizer, *, gamma, lbd, kind, grad_accum):
    """Return step(batch, use_jepa, flush=False) -> metrics.

    An update is applied every grad_accum calls, or early when flush is set (epoch end).
    """
    loss_and_grad = nn.value_and_grad(
        model,
        lambda batch, use_jepa: total_loss(
            model, batch, gamma=gamma, lbd=lbd, kind=kind, use_jepa=use_jepa
        ),
    )
    pending = {"grads": None, "count": 0}

    def step(batch, use_jepa, flush=False):
        (total, lm, jepa), grads = loss_and_grad(batch, use_jepa)
        grads = tree_map(lambda g: g / grad_accum, grads)
        if pending["grads"] is not None:
            grads = tree_map(lambda a, b: a + b, pending["grads"], grads)
        count = pending["count"] + 1
        if count == grad_accum or flush:
            optimizer.update(model, _clip(grads, MAX_GRAD_NORM))
            grads, count = None, 0
        pending.update(grads=grads, count=count)
        mx.eval(model.parameters(), optimizer.state, grads, total, lm, jepa)
        return {"loss": total.item(), "lm": lm.item(), "jepa": jepa.item()}

    return step
