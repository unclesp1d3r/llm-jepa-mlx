import json
import sqlite3

import pytest

import evaluate

SYNTH = [{}, {"content": "q"}, {"content": ".*dog.*"}]
GSM8K = [{}, {"content": "q"}, {"content": "2+2=4\n#### 4"}]


def test_default_metric_is_exact_match():
    assert evaluate.is_correct(".*dog.*", SYNTH, "synth_test.jsonl")
    assert not evaluate.is_correct(".*dog.*.*", SYNTH, "synth_test.jsonl")


def test_startswith_metric():
    assert evaluate.is_correct(".*dog.*.*", SYNTH, "synth_test.jsonl", startswith=True)


def test_gsm8k_compares_final_answer_only_and_accepts_a_directory_prefix():
    assert evaluate.is_correct("two and two\n#### 4", GSM8K, "datasets/gsm8k_test.jsonl")
    assert not evaluate.is_correct("two and two\n#### 5", GSM8K, "datasets/gsm8k_test.jsonl")
    assert not evaluate.is_correct("4", GSM8K, "datasets/gsm8k_test.jsonl")


def _nq(reference):
    return [{}, {"content": "q"}, {"content": reference}]


def test_nq_open_requires_a_reference_answer_to_appear_in_the_output():
    truth = _nq("Nine")
    assert not evaluate.is_correct("", truth, "nq_open_test.jsonl")
    assert not evaluate.is_correct("e", truth, "nq_open_test.jsonl")
    assert evaluate.is_correct("Nine", truth, "nq_open_test.jsonl")
    assert evaluate.is_correct("The answer is Nine.", truth, "nq_open_test.jsonl")


def test_nq_open_accepts_any_one_of_several_reference_answers():
    truth = _nq("759; 527")
    assert evaluate.is_correct("527", truth, "nq_open_test.jsonl")
    assert not evaluate.is_correct("5", truth, "nq_open_test.jsonl")


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_examples_writes_one_line_per_result_before_a_later_row_fails(tmp_path):
    rows = [{"messages": i} for i in range(5)]

    def score(messages):
        if messages == 2:
            raise RuntimeError("boom")
        return {"n": messages}

    out = tmp_path / "out.jsonl"
    with out.open("w") as handle, pytest.raises(RuntimeError):
        evaluate.run_examples(rows, score, handle)
    assert _lines(out) == [{"n": 0}, {"n": 1}]


def test_run_examples_without_an_output_file_returns_results_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results = evaluate.run_examples([{"messages": 1}, {"messages": 2}], lambda m: {"n": m})
    assert results == [{"n": 1}, {"n": 2}]
    assert list(tmp_path.iterdir()) == []


def _spider_example(tmp_path, reference, rows=(("a", 1), ("b", 2))):
    db_dir = tmp_path / "shop"
    db_dir.mkdir()
    conn = sqlite3.connect(db_dir / "shop.sqlite")
    conn.execute("CREATE TABLE item (name TEXT, qty INTEGER)")
    conn.executemany("INSERT INTO item VALUES (?, ?)", rows)
    conn.commit()
    conn.close()
    messages = [{}, {"content": "For db_id:[shop] list items"}, {"content": reference}]
    return messages, str(tmp_path)


def _spider_score(generated, messages, spider_path):
    return evaluate.is_correct(generated, messages, "spider_test.jsonl", spider_path)


def test_spider_invalid_sql_is_wrong_even_when_reference_returns_no_rows(tmp_path):
    messages, path = _spider_example(tmp_path, "SELECT name FROM item WHERE qty > 99")
    assert not _spider_score("this is not sql", messages, path)


def test_spider_valid_query_returning_no_rows_is_correct_when_reference_returns_no_rows(tmp_path):
    messages, path = _spider_example(tmp_path, "SELECT name FROM item WHERE qty > 99")
    assert _spider_score("SELECT name FROM item WHERE qty > 98", messages, path)


def test_spider_unordered_reference_accepts_rows_in_another_order(tmp_path):
    messages, path = _spider_example(tmp_path, "SELECT name FROM item")
    assert _spider_score("SELECT name FROM item ORDER BY name DESC", messages, path)


def test_spider_unordered_comparison_counts_duplicates(tmp_path):
    messages, path = _spider_example(tmp_path, "SELECT name FROM item")
    assert not _spider_score("SELECT name FROM item UNION ALL SELECT 'a'", messages, path)


def test_spider_ordered_reference_rejects_rows_in_another_order(tmp_path):
    messages, path = _spider_example(tmp_path, "SELECT name FROM item ORDER BY name")
    assert not _spider_score("SELECT name FROM item ORDER BY name DESC", messages, path)


def test_spider_lower_case_order_by_in_reference_still_counts_order(tmp_path):
    messages, path = _spider_example(tmp_path, "select name from item order by name")
    assert not _spider_score("select name from item order by name desc", messages, path)


def test_spider_shell_command_is_wrong_and_runs_nothing(tmp_path):
    messages, path = _spider_example(tmp_path, "SELECT name FROM item")
    marker = tmp_path / "marker"
    assert not _spider_score(f".shell touch {marker}", messages, path)
    assert not marker.exists()


def test_spider_delete_is_wrong_and_leaves_rows_unchanged(tmp_path):
    messages, path = _spider_example(tmp_path, "SELECT name FROM item")
    assert not _spider_score("DELETE FROM item", messages, path)
    conn = sqlite3.connect(tmp_path / "shop" / "shop.sqlite")
    assert conn.execute("SELECT count(*) FROM item").fetchone() == (2,)


def test_spider_attach_database_is_wrong_and_creates_no_file(tmp_path):
    messages, path = _spider_example(tmp_path, "SELECT name FROM item")
    target = tmp_path / "attached.db"
    assert not _spider_score(f"ATTACH DATABASE '{target}' AS x", messages, path)
    assert not target.exists()


def test_spider_two_statements_in_one_output_are_wrong(tmp_path):
    messages, path = _spider_example(tmp_path, "SELECT name FROM item")
    assert not _spider_score("SELECT name FROM item; SELECT name FROM item", messages, path)


def test_spider_text_that_is_not_valid_utf8_scores_against_itself(tmp_path):
    messages, path = _spider_example(tmp_path, "SELECT CAST(x'ff' AS TEXT)")
    assert _spider_score("SELECT CAST(x'ff' AS TEXT)", messages, path)


def test_spider_double_quoted_text_value_is_read_as_text(tmp_path):
    messages, path = _spider_example(tmp_path, 'SELECT name FROM item WHERE name = "a"')
    assert _spider_score("SELECT name FROM item WHERE name = 'a'", messages, path)


def test_spider_unrunnable_reference_makes_every_output_wrong_and_is_reported(tmp_path):
    messages, path = _spider_example(tmp_path, "SELECT nope FROM nowhere")
    assert not _spider_score("SELECT nope FROM nowhere", messages, path)
    assert not _spider_score("SELECT name FROM item", messages, path)
    assert evaluate.spider_unscorable_db_id(messages, path) == "shop"


def test_spider_runnable_reference_is_not_reported_unscorable(tmp_path):
    messages, path = _spider_example(tmp_path, "SELECT name FROM item")
    assert evaluate.spider_unscorable_db_id(messages, path) is None


def test_spider_missing_database_file_raises_error_naming_the_path(tmp_path):
    messages = [{}, {"content": "For db_id:[ghost] q"}, {"content": "SELECT 1"}]
    with pytest.raises(FileNotFoundError, match="ghost.sqlite"):
        _spider_score("SELECT 1", messages, str(tmp_path))
