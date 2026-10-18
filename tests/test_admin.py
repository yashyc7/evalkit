import pytest

from evalkit.admin import parse_jsonl


def test_parse_valid_cases():
    text = '{"input": "Q1", "expected": ["a"], "tags": ["x"]}\n{"input": "Q2"}\n'
    assert parse_jsonl(text) == [
        {"input": "Q1", "expected": ["a"], "tags": ["x"]},
        {"input": "Q2", "expected": None, "tags": []},
    ]


def test_parse_skips_blank_lines():
    text = '{"input": "Q1"}\n\n   \n{"input": "Q2"}\n'
    assert len(parse_jsonl(text)) == 2


def test_parse_reports_bad_json_line_number():
    with pytest.raises(ValueError, match="Line 2"):
        parse_jsonl('{"input": "Q1"}\nnot json\n')


def test_parse_requires_input():
    with pytest.raises(ValueError, match="Line 1"):
        parse_jsonl('{"question": "Q1"}\n')