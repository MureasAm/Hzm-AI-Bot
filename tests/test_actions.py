# -*- coding: utf-8 -*-
"""动作标记（[[语音]]/[[表情:类别]]）与动作回执。"""
from datetime import datetime, timedelta

from src.plugins.chatbot import core
from src.plugins.chatbot import reply_style
from src.plugins.chatbot import session_memory as sm


class TestActionTags:
    def test_voice_tag_is_stripped(self):
        text, actions = reply_style.extract_action_tags("晚安啦 [[语音]]")
        assert text == "晚安啦"
        assert actions == {"voice": True, "sticker": ""}

    def test_sticker_tag_carries_group(self):
        text, actions = reply_style.extract_action_tags("好耶，那我不客气了 [[表情:开心]]")
        assert text == "好耶，那我不客气了"
        assert actions == {"voice": False, "sticker": "开心"}

    def test_both_tags_and_trailing_comma(self):
        text, actions = reply_style.extract_action_tags("知道啦，[[表情：害羞]] [[语音]]")
        assert text == "知道啦"
        assert actions == {"voice": True, "sticker": "害羞"}

    def test_plain_text_unchanged(self):
        text, actions = reply_style.extract_action_tags("嗯，晚安")
        assert text == "嗯，晚安"
        assert actions == {"voice": False, "sticker": ""}

    def test_reply_is_a_str_with_actions(self):
        r = reply_style.Reply("好的", {"voice": True, "sticker": ""})
        assert r == "好的"
        assert r.actions["voice"] is True


class TestActionsBlock:
    def test_capability_block_is_opt_in(self):
        on = core.build_message_list("在吗", [], [], show_actions=True)
        off = core.build_message_list("在吗", [], [])
        assert any("你可以做的动作" in m["content"] for m in on)
        assert not any("你可以做的动作" in m["content"] for m in off)


class TestActionReceipts:
    def test_roundtrip_and_freshness(self):
        sm.add_last_action("u_actions", "sticker", "委屈")
        sm.add_last_action("u_actions", "feedback", "对方让你别发表情", ok=False)
        note = sm.actions_note("u_actions")
        assert "「委屈」表情包" in note and "别发表情" in note
        # 语音形态改由历史行的 [语音] 承载，不再进 last_actions
        assert "语音" not in note

    def test_stale_actions_are_not_injected(self):
        import json
        import os
        from pathlib import Path
        p = sm._file_for("u_actions")
        p.parent.mkdir(parents=True, exist_ok=True)
        old = (datetime.now() - timedelta(hours=2)).isoformat()
        p.write_text(json.dumps({
            "topic": "", "events": [], "last_active": datetime.now().isoformat(),
            "last_actions": [{"kind": "sticker", "detail": "旧表情", "t": old}],
        }, ensure_ascii=False), encoding="utf-8")
        assert sm.actions_note("u_actions") == ""


class TestVoicePreference:
    def test_long_term_request_sets_mode(self):
        sm.note_modality_feedback("u_voice", "以后都用语音跟我说话")
        assert sm.voice_preference_active("u_voice")
        assert "希望你用语音" in sm.voice_preference_note("u_voice")

    def test_current_turn_request_does_not_set_mode(self):
        sm.note_modality_feedback("u_voice2", "给我发条语音嘛")
        assert not sm.voice_preference_active("u_voice2")

    def test_cancel_clears_and_leaves_feedback(self):
        sm.note_modality_feedback("u_voice3", "以后都用语音")
        sm.note_modality_feedback("u_voice3", "还是打字吧")
        assert not sm.voice_preference_active("u_voice3")
        assert "别发语音" in sm.actions_note("u_voice3")

    def test_mode_expires(self):
        from datetime import datetime, timedelta
        now = datetime.now()
        sm.set_voice_preference("u_voice4", seconds=60)
        assert sm.voice_preference_active("u_voice4", now=now + timedelta(seconds=30))
        assert not sm.voice_preference_active("u_voice4", now=now + timedelta(seconds=61))

    def test_sticker_complaint_is_recorded(self):
        sm.note_modality_feedback("u_voice5", "别发表情包了")
        assert "别发表情" in sm.actions_note("u_voice5")

    def test_failure_outcomes_are_visible(self):
        sm.add_last_action("u_voice6", "sticker", "委屈", ok=False)
        note = sm.actions_note("u_voice6")
        assert "表情包发送失败" in note and "语音" not in note

    def test_voice_only_goes_to_audit_log(self):
        sm.log_action_event("u_voice7", "voice", ok=False)
        assert sm.actions_note("u_voice7") == ""
