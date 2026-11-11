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


def test_nq_open_accepts_any_listed_answer_as_substring():
    truth = [{}, {"content": "q"}, {"content": "Paris; City of Paris"}]
    assert evaluate.is_correct("Rome; Paris", truth, "nq_open_test.jsonl")
    assert not evaluate.is_correct("Rome", truth, "nq_open_test.jsonl")
