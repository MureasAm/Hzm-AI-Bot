"""可选标注必须能向人解释，且不能被二元判据实验偷换成负例。"""
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import judge_experiment
import problem_cases


def test_optional_required_and_forbidden_constraints_are_all_described():
    description = problem_cases.describe_expect({"preference": {
        "should": ["food"], "allowed": ["food", "routine"], "must_not": ["routine"]}})
    assert "food" in description and "routine" in description
    assert "允许" in description and "禁止" in description and "该命中" in description
    assert "没有期望" not in description


@pytest.mark.parametrize("condition", [{"allowed": ["food"]}, {"allowed": []}])
def test_optional_constraint_is_not_reported_as_missing_expectation(condition):
    description = problem_cases.describe_expect({"preference": condition})
    assert "可空" in description and "允许" in description
    assert "没有期望" not in description


def test_facts_are_described_as_all_required_information():
    description = problem_cases.describe_expect({"facts": {"contains": ["澳洲", "袋鼠"]}})
    assert "最终可用信息" in description and "全部" in description
    assert "澳洲" in description and "袋鼠" in description


def test_understanding_constraints_are_described_without_becoming_missing_expectations():
    description = problem_cases.describe_expect({"understanding": {
        "focus": "character", "exclusions_contains": [["亲身经历", "见过"]],
        "premises_contains": ["男学霸"], "third_parties_empty": True}})
    assert all(text in description for text in ("character", "亲身经历", "男学霸", "third_parties"))
    assert "为空" in description and "没有期望" not in description


def test_allowed_understanding_focus_is_described_as_a_bounded_set():
    description = problem_cases.describe_expect({"understanding": {
        "focus_allowed": ["user", "general"]}})
    assert "user" in description and "general" in description and "主体" in description
    assert "没有期望" not in description


def test_optional_or_forbidden_only_cases_do_not_turn_into_binary_negatives(monkeypatch, tmp_path):
    cases = [
        {"query": "optional", "expect": {"preference": {"allowed": ["food"]}}},
        {"query": "forbidden-only", "expect": {"preference": {"must_not": ["food"]}}},
        {"query": "both-optional", "expect": {"preference": {"allowed": ["food"], "must_not": ["routine"]}}},
        {"query": "required", "expect": {"preference": {"should": ["food"], "allowed": ["food"]}}},
        {"query": "empty", "expect": {"preference": {"should_not_hit": True}}},
        {"query": "facts-only", "expect": {"facts": {"contains": ["东野圭吾"]}}},
        {"query": "understanding-only", "expect": {"understanding": {"focus": "character"}}},
    ]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(judge_experiment, "CASES_FILE", path)
    tasks = judge_experiment.load_tasks()
    assert {case["q"]: case["want"] for case in tasks["preference"]["cases"]} == {
        "required": True, "empty": False}
    assert all(case["q"] != "facts-only" for task in tasks.values() for case in task["cases"])
    assert all(case["q"] != "understanding-only" for task in tasks.values() for case in task["cases"])
