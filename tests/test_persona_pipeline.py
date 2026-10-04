"""persona pipeline 的纯函数测试（不调用 LLM）。"""
import sys
from pathlib import Path

import pytest
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import persona_pipeline as pp  # noqa: E402


class TestMergeCorpus:
    def test_append_unique_and_preserve_schema(self):
        existing = {
            "_readme": "x",
            "statements": [{"statement": "旧事件"}],
        }
        merged, added = pp._merge_corpus(
            existing,
            [{"statement": "旧事件"}, {"statement": "新事件"}],
        )
        assert merged == [{"statement": "旧事件"}, {"statement": "新事件"}]
        assert added == [{"statement": "新事件"}]


class TestSelfPronounAudit:
    def test_detects_third_person_self_reference_candidate(self):
        assert pp._has_self_pronoun("灰泽满不想走捷径，她宁愿慢慢来。")

    def test_detects_pronoun_in_object_position(self):
        assert pp._has_self_pronoun("有人定期给灰泽满科普，也愿意给她讲避雷。")

    def test_detects_possessive_self_reference(self):
        assert pp._has_self_pronoun("灰泽满刚下课，她的情绪切换得很快。")

    def test_no_self_name_is_not_flagged(self):
        assert not pp._has_self_pronoun("女同学在教室里等她一起走。")


class TestMergeVoice:
    def test_append_with_id_and_dedupe(self):
        existing = {
            "_readme": "x",
            "samples": [{
                "id": "a", "type": "daily", "length": "short",
                "user": "粉丝问：在吗", "reply": "在。",
            }],
        }
        merged, added = pp._merge_voice(
            existing,
            [
                {"user": "粉丝问：在吗", "reply": "在。"},
                {"user": "粉丝问：吃了吗", "reply": "吃了。"},
            ],
            "session",
        )
        assert len(merged) == 2
        assert added[0]["user"] == "粉丝问：吃了吗"
        assert added[0]["id"].startswith("session_")


class TestMergePhrases:
    def test_merge_same_meaning(self):
        existing = {
            "_readme": "x",
            "phrase_groups": [{
                "id": "brag_deny",
                "meaning": "被夸时否认",
                "trigger": "",
                "phrases": ["也没有啦"],
                "usage": "",
            }],
        }
        incoming = [{
            "id": "phrase_001",
            "meaning": "被夸时否认",
            "trigger": "被夸奖时",
            "phrases": ["也没有啦", "别夸了"],
            "usage": "先否认",
        }]
        merged, added = pp._merge_phrases(existing, incoming)
        assert merged[0]["id"] == "brag_deny"
        assert merged[0]["phrases"] == ["也没有啦", "别夸了"]
        assert merged[0]["trigger"] == "被夸奖时"
        assert added == [{"id": "brag_deny", "meaning": "被夸时否认",
                          "phrases": ["别夸了"]}]


class TestMergeBehaviors:
    def test_merge_samples_and_evidence(self):
        existing = {
            "_readme": "x",
            "behaviors": [{
                "name": "被夸时嘴硬否认",
                "trigger": "被夸时",
                "response": "先否认",
                "samples": [{"user": "夸夸", "reply": "没有啦"}],
                "evidence": [],
            }],
        }
        incoming = [{
            "name": "被夸时嘴硬否认",
            "trigger": "",
            "response": "",
            "samples": [
                {"user": "夸夸", "reply": "没有啦"},
                {"user": "再夸", "reply": "别夸了"},
            ],
            "evidence": ["素材样本：再夸 → 别夸了"],
        }]
        merged, added = pp._merge_behaviors(existing, incoming)
        assert len(merged[0]["samples"]) == 2
        assert merged[0]["evidence"] == ["素材样本：再夸 → 别夸了"]
        assert len(added[0]["samples"]) == 1


class TestCli:
    def test_run_tool_registers_pipeline(self):
        import run_tool

        args = run_tool.build_parser().parse_args([
            "persona-pipeline", "corpus",
            "-i", "a.json", "b.json",
            "--session", "s1",
        ])
        assert args.target == "corpus"
        assert args.input == [["a.json", "b.json"]]

    def test_precompute_excludes_retired_targets(self):
        import run_tool

        # voice-samples / phrases 已于 2026-10-04 下线（并入 behaviors）→ 两个都该被 argparse 拒
        for retired in ("voice-samples", "phrases"):
            with pytest.raises(SystemExit):
                run_tool.build_parser().parse_args(["precompute", retired])


class TestStatementPrompt:
    def test_json_braces_are_escaped_for_format(self):
        import generate_statements

        prompt = generate_statements.STATEMENT_PROMPT.format(text="素材")
        assert '{"statement": "灰泽满当时……"}' in prompt


class TestToolModulesImport:
    def test_modified_tools_import(self):
        import analyze_pace
        import clean_transcript
        import convert_to_chat
        import extract_persona
        import generate_statements
        import mine_phrases

        assert analyze_pace.MODEL
        assert clean_transcript.MODEL
        assert convert_to_chat.MODEL
        assert extract_persona.MODEL
        assert generate_statements.MODEL
        assert mine_phrases.MODEL


class TestReuseApply:
    async def test_reuse_candidate_applies_corpus_without_regeneration(
        self, tmp_path, monkeypatch
    ):
        stage_root = tmp_path / "pipeline"
        stage = stage_root / "session"
        stage.mkdir(parents=True)
        (stage / "corpus_candidates.json").write_text(
            '{"statements": [{"statement": "新事件"}]}',
            encoding="utf-8",
        )
        target = tmp_path / "statement_final.json"
        target.write_text(
            '{"_readme": "x", "statements": [{"statement": "旧事件"}]}',
            encoding="utf-8",
        )
        monkeypatch.setattr(pp, "PIPELINE_DIR", stage_root)
        monkeypatch.setattr(pp, "STATEMENT_FILE", target)
        hook_calls = []
        monkeypatch.setattr(pp, "_build_corpus_hooks", lambda: hook_calls.append(True))
        args = SimpleNamespace(
            target="corpus",
            session="session",
            audio=None,
            input=None,
            apply=True,
            refresh=False,
            no_vectorize=True,
            no_hooks=False,
            sanitize_existing=False,
        )
        await pp.run(args)
        data = target.read_text(encoding="utf-8")
        assert "旧事件" in data
        assert "新事件" in data
        assert hook_calls == [True]
