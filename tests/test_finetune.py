import pytest

import finetune
from tests.test_data import prepare
from tests.test_jepa import EXAMPLES, tiny_model


def test_count_truncated_counts_examples_that_reached_max_length():
    short = {"input_ids": [1, 2], "labels": [1, 2], "text_ids": [1, 2], "code_ids": [1, 2]}
    examples = [prepare(max_length=5), short, prepare(max_length=5)]
    assert finetune.count_truncated(examples, max_length=5) == 2


def test_count_truncated_is_zero_when_no_example_reaches_max_length():
    assert finetune.count_truncated([prepare(max_length=512)], max_length=512) == 0


def tiny_training(num_epochs, batch_size=1, grad_accum=4, examples=10):
    args = finetune.parse_args([
        "--train_file", "unused.jsonl", "--num_epochs", str(num_epochs),
        "--batch_size", str(batch_size), "--grad_accum", str(grad_accum), "--eval_steps", "1000",
    ])
    return args, [EXAMPLES[i % len(EXAMPLES)] for i in range(examples)]


def test_train_calls_the_save_callback_at_the_end_of_each_epoch_until_one_raises():
    args, examples = tiny_training(num_epochs=3)
    saved = []

    def on_epoch_end(epoch):
        if epoch == 3:
            raise RuntimeError("interrupted")
        saved.append(epoch)

    with pytest.raises(RuntimeError):
        finetune.train(args, tiny_model(), examples, on_epoch_end)
    assert saved == [1, 2]


def test_train_returns_the_number_of_optimizer_updates_it_applied():
    args, examples = tiny_training(num_epochs=3, grad_accum=4)  # 10 batches per epoch
    assert finetune.train(args, tiny_model(), examples, lambda epoch: None) == 9


def test_main_stops_before_training_when_an_embedding_position_is_outside_its_view(monkeypatch):
    short_view = [{"input_ids": [1, 2], "labels": [-100, 2], "text_ids": [1, 2], "code_ids": [1, 2, 3]}]
    monkeypatch.setattr(finetune.model_lib, "load", lambda *args, **kwargs: (tiny_model(), None, {}))
    monkeypatch.setattr(finetune, "prepare_dataset", lambda args, tokenizer: short_view)
    monkeypatch.setattr(finetune, "train", lambda *args: pytest.fail("training started"))
    with pytest.raises(ValueError, match="example 0"):
        finetune.main(["--train_file", "unused.jsonl", "--last_token", "-3"])
