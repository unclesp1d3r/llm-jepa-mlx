"""Dataset preparation for LLM-JEPA: chat formatting, label masking, and the Text/Code views."""

import json

IGNORE = -100
MAX_PREDICTORS = 10
SPECIAL_TOKENS = [f"<|predictor_{i}|>" for i in range(1, MAX_PREDICTORS + 1)] + [
    "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|perception|>",
]


def load_jsonl(path):
    with open(path) as lines:
        return [json.loads(line) for line in lines if line.strip()]


def is_gemma(model_name):
    return "gemma" in model_name.lower()


def get_messages(model_name, messages):
    """Gemma has no system role: fold the system prompt into the user turn."""
    if not is_gemma(model_name):
        return messages
    system, user, assistant = messages[:3]
    return [{**user, "content": system["content"] + "\n\n" + user["content"]}, assistant]


def get_user_messages(messages):
    return [dict(messages[1])]


def get_assistant_messages(model_name, messages):
    role = "user" if is_gemma(model_name) else messages[2]["role"]
    return [{**messages[2], "role": role}]


def _with_predictors(messages, predictors, front_pred):
    ascending = "".join(f"<|predictor_{i}|>" for i in range(1, predictors + 1))
    descending = "".join(f"<|predictor_{i}|>" for i in range(predictors, 0, -1))
    content = messages[0]["content"]
    content = ascending + content if front_pred else content + descending
    return [{**messages[0], "content": content}]


def _format(tokenizer, messages, plain, add_generation_prompt=False):
    if plain:
        return messages[0]["content"]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def _assistant_content_labels(messages, tokenizer, input_ids):
    """Unmask only the first occurrence of each assistant message's content."""
    labels = [IGNORE] * len(input_ids)
    decoded_input = [tokenizer.decode(token) for token in input_ids]
    for message in messages:
        if message["role"] != "assistant":
            continue
        # Compare decoded pieces: the same text can get different ids in context.
        target = [
            tokenizer.decode(token)
            for token in tokenizer.encode(message["content"], add_special_tokens=False)
        ]
        for start in range(len(input_ids) - len(target) + 1):
            if decoded_input[start:start + len(target)] == target:
                labels[start:start + len(target)] = input_ids[start:start + len(target)]
                break
    return labels


def prepare_example(
    example, tokenizer, *, model_name, max_length, predictors=0, regular=False,
    train_all=False, plain=False, front_pred=False, reverse_pred=False,
    unmask_assistant_special_tokens=False, use_view_fields=False,
):
    """Tokenize one example into LM inputs/labels plus the Text and Code views (unpadded)."""
    add_special_tokens = not unmask_assistant_special_tokens

    def encode(text):
        return tokenizer(
            text, truncation=True, max_length=max_length, add_special_tokens=add_special_tokens
        )["input_ids"]

    messages = example["messages"]
    full_messages = get_messages(model_name, messages)
    user, assistant = messages[1]["content"], messages[2]["content"]
    if not plain:
        prompt_text = _format(tokenizer, full_messages[:-1], plain, add_generation_prompt=True)
        full_text = _format(tokenizer, full_messages, plain)
    elif train_all:
        prompt_text, full_text = "", user + "<|eot_id|>"
    else:
        prompt_text = user + "<|perception|>"
        full_text = prompt_text + assistant + "<|eot_id|>"

    input_ids = encode(full_text)
    if train_all:
        labels = list(input_ids)
    elif unmask_assistant_special_tokens:
        prompt_length = min(
            len(tokenizer.encode(prompt_text, add_special_tokens=False)), len(input_ids)
        )
        labels = [IGNORE] * prompt_length + input_ids[prompt_length:]
    else:
        labels = _assistant_content_labels(messages, tokenizer, input_ids)

    prepared = {"input_ids": input_ids, "labels": labels}
    if regular:
        return prepared

    if use_view_fields:  # hellaswag ships its views explicitly
        text, code = example["text"], example["code"]
    else:
        text, code = get_user_messages(messages), get_assistant_messages(model_name, messages)
        if reverse_pred:
            text, code = code, text
    text = _with_predictors(text, predictors, front_pred)
    return {
        **prepared,
        "text_ids": encode(_format(tokenizer, text, plain)),
        "code_ids": encode(_format(tokenizer, code, plain)),
    }
