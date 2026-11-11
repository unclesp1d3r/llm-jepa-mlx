"""Evaluation for LLM-JEPA on Apple Silicon (MLX port of upstream evaluate.py)."""

import argparse
import json
import os
import re
import sqlite3
import statistics
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import mlx.core as mx
import mlx_lm
from tqdm import tqdm

from data import get_assistant_messages, get_messages, get_user_messages, load_jsonl

GSM8K_ANSWER = re.compile(r"\n#### (.+)$")
SPIDER_DB = re.compile(r"For db_id:\[(.+)\]")
OPTIONS = ["A", "B", "C", "D"]
READ_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}


def _run_sql(db_file, query):
    """Rows of a query, or None when it cannot run."""
    conn = sqlite3.connect(f"{Path(db_file).resolve().as_uri()}?mode=ro", uri=True)
    try:
        conn.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DML, True)
        conn.set_authorizer(_read_only_authorizer)
        conn.text_factory = bytes  # results that are not valid UTF-8 must still compare
        cursor = conn.execute(query)
        # an empty or comment-only string executes without being a query
        return cursor.fetchall() if cursor.description else None
    except (sqlite3.Error, ValueError):
        return None
    finally:
        conn.close()


def _read_only_authorizer(action, *_):
    return sqlite3.SQLITE_OK if action in READ_ACTIONS else sqlite3.SQLITE_DENY


def _spider_db_file(messages, spider_path):
    db_id = SPIDER_DB.search(messages[1]["content"]).group(1)
    db_file = os.path.join(spider_path, db_id, db_id + ".sqlite")
    if not os.path.exists(db_file):
        raise FileNotFoundError(f"{db_file} not found; unzip spider_data.zip and pass --spider_path")
    return db_id, db_file


def _same_rows(generated_rows, reference_rows, reference_sql):
    if re.search("ORDER BY", reference_sql, re.IGNORECASE):
        return generated_rows == reference_rows
    return Counter(generated_rows) == Counter(reference_rows)


def spider_score(generated, messages, spider_path):
    """(whether the output is correct, the database id when the reference query cannot run)."""
    db_id, db_file = _spider_db_file(messages, spider_path)
    reference_sql = messages[2]["content"]
    reference_rows = _run_sql(db_file, reference_sql)
    if reference_rows is None:
        return False, db_id
    generated_rows = _run_sql(db_file, generated)
    if generated_rows is None:
        return False, None
    return _same_rows(generated_rows, reference_rows, reference_sql), None


def _gsm8k_answer(text):
    match = GSM8K_ANSWER.search(text)
    return match.group(1) if match else None


def is_correct(generated, messages, input_file, spider_path="", startswith=False):
    expected = messages[2]["content"]
    dataset = os.path.basename(input_file)
    if startswith:
        return generated.startswith(expected)
    if dataset.startswith("gsm8k"):
        return _gsm8k_answer(generated) == _gsm8k_answer(expected)
    if dataset.startswith("spider"):
        return spider_score(generated, messages, spider_path)[0]
    if dataset.startswith("nq_open"):
        return any(answer in generated for answer in expected.split("; "))
    return generated == expected


def build_prompt(tokenizer, messages, plain, include_assistant=False, similarity=False):
    if not include_assistant:
        messages = [m for m in messages if m["role"] != "assistant"]
    if not plain:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return messages[0]["content"] if similarity else messages[1]["content"] + "<|perception|>"


def generate_response(model, wrapper, prompt_ids, max_new_tokens):
    """Greedy decoding (mlx-lm's default sampler is argmax)."""
    tokens = [
        response.token
        for response in mlx_lm.stream_generate(model, wrapper, prompt_ids, max_tokens=max_new_tokens)
    ]
    return wrapper._tokenizer.decode(tokens, skip_special_tokens=True).strip()


def choose_option(model, tokenizer, prompt_ids):
    """Multiple choice (HellaSwag): the most likely of A/B/C/D as the next token."""
    logits = model(mx.array([prompt_ids]))[0, -1]
    option_ids = mx.array(tokenizer.convert_tokens_to_ids(OPTIONS))
    return OPTIONS[mx.argmax(logits[option_ids]).item()]


def cosine_similarity(model, text_ids, code_ids):
    text, code = (
        model.model(mx.array([ids]))[0, -1].astype(mx.float32) for ids in (text_ids, code_ids)
    )
    return (mx.sum(text * code) / (mx.linalg.norm(text) * mx.linalg.norm(code))).item()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a (fine-tuned) model by greedy generation (MLX)")
    parser.add_argument("--model_name", required=True, help="Model path or repo")
    parser.add_argument("--original_model_name", help="Base model name, used to pick the chat layout")
    parser.add_argument("--input_file", required=True, help="Test JSONL file")
    parser.add_argument("--output_file", help="Write per-example results here as JSONL")
    parser.add_argument("--max_examples", type=int, help="Only evaluate the first N examples")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=512, help="Maximum prompt length")
    parser.add_argument("--similarity", action="store_true",
                        help="Also report cos(Enc(Text), Enc(Code)) for right and wrong answers")
    parser.add_argument("--startswith", action="store_true",
                        help="Also report the rate of outputs that start with the ground truth")
    parser.add_argument("--plain", action="store_true", help="No chat template; append <|perception|>")
    parser.add_argument("--spider_path", default="", help="Path to the Spider sqlite databases")
    parser.add_argument("--unmask_assistant_special_tokens", action="store_true",
                        help="Must match the training configuration")
    return parser.parse_args(argv)


def evaluate_example(args, model, wrapper, messages):
    tokenizer = wrapper._tokenizer
    original = args.original_model_name or args.model_name

    def encode(text):
        return tokenizer(
            text, truncation=True, max_length=args.max_length,
            add_special_tokens=not args.unmask_assistant_special_tokens,
        )["input_ids"]

    prompt_ids = encode(build_prompt(tokenizer, get_messages(original, messages), args.plain))
    if os.path.basename(args.input_file).startswith("hellaswag"):
        generated = choose_option(model, tokenizer, prompt_ids)
    else:
        generated = generate_response(model, wrapper, prompt_ids, args.max_new_tokens)

    if os.path.basename(args.input_file).startswith("spider"):
        correct, unscorable_db_id = spider_score(generated, messages, args.spider_path)
    else:
        correct, unscorable_db_id = is_correct(generated, messages, args.input_file), None
    result = {
        "generated": generated,
        "expected": messages[2]["content"],
        "correct": correct,
        "startswith": is_correct(generated, messages, args.input_file, startswith=True),
    }
    if unscorable_db_id:
        result = {**result, "unscorable_db_id": unscorable_db_id}
    if not args.similarity:
        return result
    views = (get_user_messages(messages), get_assistant_messages(original, messages))
    text_ids, code_ids = (
        encode(build_prompt(tokenizer, view, args.plain, include_assistant=True, similarity=True))
        for view in views
    )
    return {**result, "similarity": cosine_similarity(model, text_ids, code_ids)}


def report(args, results):
    total = len(results)
    correct = [r for r in results if r["correct"]]
    line = f"Success Rate: {args.model_name}, {len(correct) / total}"
    if args.startswith:
        line += f", {sum(r['startswith'] for r in results) / total}"
    print(line)
    unscorable = [r["unscorable_db_id"] for r in results if r.get("unscorable_db_id")]
    if unscorable:
        print(f"{len(unscorable)} examples have a reference query that cannot run (scored wrong): "
              f"{', '.join(unscorable)}")
    if args.similarity:
        wrong = [r for r in results if not r["correct"]]
        for name, group in (("correct", correct), ("incorrect", wrong)):
            sims = [r["similarity"] for r in group]
            if sims:
                print(f"cosine similarity ({name}, n={len(sims)}): "
                      f"mean {statistics.mean(sims):.4f} std {statistics.pstdev(sims):.4f}")


def run_examples(rows, score, out=None):
    results = []
    for row in rows:
        result = score(row["messages"])
        results.append(result)
        if out:
            out.write(json.dumps(result) + "\n")
            out.flush()
    return results


def main(argv=None):
    args = parse_args(argv)
    model, wrapper = mlx_lm.load(args.model_name)
    model.eval()
    rows = load_jsonl(args.input_file)[:args.max_examples]
    if not rows:
        raise ValueError(f"No examples in {args.input_file}")
    print(f"Loaded {len(rows)} examples from {args.input_file}")

    with open(args.output_file, "w") if args.output_file else nullcontext() as out:
        results = run_examples(
            tqdm(rows, desc="Generating responses"),
            lambda messages: evaluate_example(args, model, wrapper, messages),
            out,
        )
    report(args, results)


if __name__ == "__main__":
    main()
