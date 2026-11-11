import data

IGNORE = data.IGNORE
MESSAGES = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "lines with dog"},
    {"role": "assistant", "content": ".*dog.*"},
]


class CharTokenizer:
    """One token per character, BOS id 1; enough to exercise label masking offline."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role'][0]}>{m['content']}\n" for m in messages)
        return text + ("<a>" if add_generation_prompt else "")

    def __call__(self, text, truncation=True, max_length=None, add_special_tokens=True):
        return {"input_ids": self.encode(text, add_special_tokens)[:max_length]}

    def encode(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + [ord(c) for c in text]

    def decode(self, ids):
        ids = [ids] if isinstance(ids, int) else ids
        return "".join(chr(i) for i in ids if i != 1)


TOK = CharTokenizer()


def prepare(**kwargs):
    options = {"model_name": "llama", "max_length": 512, **kwargs}
    return data.prepare_example({"messages": MESSAGES}, TOK, **options)


def unmasked(example):
    return TOK.decode([t for t in example["labels"] if t != IGNORE])


def test_default_labels_unmask_only_assistant_content():
    example = prepare()
    assert unmasked(example) == ".*dog.*"
    assert len(example["labels"]) == len(example["input_ids"])


def test_unmask_assistant_special_tokens_keeps_everything_after_generation_prompt():
    assert unmasked(prepare(unmask_assistant_special_tokens=True)) == ".*dog.*\n"


def test_labels_match_inputs_when_truncation_cuts_into_the_prompt():
    example = prepare(unmask_assistant_special_tokens=True, max_length=5)
    assert example["labels"] == [IGNORE] * 5


def test_train_all_unmasks_every_token():
    example = prepare(train_all=True)
    assert example["labels"] == example["input_ids"]


def test_views_are_text_and_code():
    example = prepare()
    assert TOK.decode(example["text_ids"]) == "<u>lines with dog\n"
    assert TOK.decode(example["code_ids"]) == "<a>.*dog.*\n"


def test_predictor_tokens_are_appended_to_text_in_descending_order():
    text = TOK.decode(prepare(predictors=2)["text_ids"])
    assert text == "<u>lines with dog<|predictor_2|><|predictor_1|>\n"


def test_front_pred_prepends_predictor_tokens():
    text = TOK.decode(prepare(predictors=2, front_pred=True)["text_ids"])
    assert text == "<u><|predictor_1|><|predictor_2|>lines with dog\n"


def test_reverse_pred_swaps_views():
    example = prepare(reverse_pred=True, predictors=1)
    assert TOK.decode(example["text_ids"]) == "<a>.*dog.*<|predictor_1|>\n"
    assert TOK.decode(example["code_ids"]) == "<u>lines with dog\n"


def test_plain_format_joins_with_perception_token():
    example = prepare(plain=True)
    assert TOK.decode(example["input_ids"]) == "lines with dog<|perception|>.*dog.*<|eot_id|>"
    assert TOK.decode(example["text_ids"]) == "lines with dog"


def test_gemma_folds_system_prompt_into_user_turn():
    example = prepare(model_name="mlx-community/gemma-2-2b-it")
    assert TOK.decode(example["input_ids"]) == "<u>sys\n\nlines with dog\n<a>.*dog.*\n"
    assert TOK.decode(example["code_ids"]) == "<u>.*dog.*\n"


def test_regular_mode_skips_views():
    assert "text_ids" not in prepare(regular=True)


def test_hellaswag_views_come_from_text_and_code_fields():
    example = {
        "messages": MESSAGES,
        "text": [{"role": "user", "content": "ctx"}],
        "code": [{"role": "user", "content": "ending"}],
    }
    out = data.prepare_example(example, TOK, model_name="llama", max_length=512, use_view_fields=True)
    assert TOK.decode(out["text_ids"]) == "<u>ctx\n"
    assert TOK.decode(out["code_ids"]) == "<u>ending\n"


def test_prepare_does_not_mutate_input_messages():
    before = [dict(m) for m in MESSAGES]
    prepare(predictors=3, model_name="gemma")
    assert MESSAGES == before
