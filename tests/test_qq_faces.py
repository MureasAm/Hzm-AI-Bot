"""QQ 内置表情（face）→ 含义文字的提取。

背景：NapCat 有时**不给 faceText**，只剩一个裸 id，于是消息变成 `[表情：QQ表情6]`，
模型完全不知道那是什么表情。`qq_faces.py` 那张表就是给这种情况兜底。

**表本身别再手抄**：从 QQ 客户端自己的表情配置提取
`<QQNT>/resources/app/resource/default-emojis/default_config.json`，
规则是 `face id = qzoneCode - 100`（经典 100-199）/ `qzoneCode - 10000`（新表情）。
**别用 `emojiId`**——那是面板自己的编号，和消息里的 face id 不是一回事（0/14 正好相反）。
"""
import pytest

from src.plugins import chatbot as entry  # 包 __init__ 本身就是事件入口模块
from src.plugins.chatbot.qq_faces import QQ_FACE_NAMES, face_name


class _Seg:
    def __init__(self, type_, data):
        self.type = type_
        self.data = data


class TestFaceNameTable:
    def test_table_not_empty(self):
        assert len(QQ_FACE_NAMES) > 200, "表太小了，多半是提取规则用错（比如用了 emojiId）"

    @pytest.mark.parametrize("fid,expect", [
        (6, "害羞"),        # 用户实际遇到的 case
        (0, "微笑"),
        (14, "惊讶"),       # 配置里 emojiId=14 是"微笑"；消息里的 14 必须是"惊讶"
        (13, "呲牙"),
        (74, "便便"),
        (76, "太阳"),
        (311, "打call"),
    ])
    def test_known_ids(self, fid, expect):
        assert face_name(fid) == expect

    def test_unknown_id_returns_empty(self):
        assert face_name(99999) == ""

    def test_bad_input_returns_empty(self):
        assert face_name("abc") == ""
        assert face_name(None) == ""
        assert face_name("") == ""

    def test_accepts_string_id(self):
        # OneBot 的 face id 常常是字符串
        assert face_name("6") == "害羞"


class TestExtractFaceText:
    """`_extract_face_text` 的三条路径：raw.faceText 优先 → id 查表 → 裸 id 兜底。"""

    def test_raw_dict_face_text_preferred(self):
        seg = _Seg("face", {"id": "6", "raw": {"faceIndex": 6, "faceText": "/比爱心"}})
        assert entry._extract_face_text([seg]) == "比爱心"

    def test_raw_python_repr_string(self):
        seg = _Seg("face", {"id": "6", "raw": "{'faceIndex': 6, 'faceText': '/害羞'}"})
        assert entry._extract_face_text([seg]) == "害羞"

    def test_raw_json_string(self):
        seg = _Seg("face", {"id": "6", "raw": '{"faceIndex": 6, "faceText": "/流泪"}'})
        assert entry._extract_face_text([seg]) == "流泪"

    def test_fallback_to_id_table(self):
        # 实测形态：NapCat 不给 raw/faceText，只剩 id → 以前变成"QQ表情6"
        seg = _Seg("face", {"id": "6"})
        assert entry._extract_face_text([seg]) == "害羞"

    def test_unknown_id_keeps_raw_id_text(self):
        seg = _Seg("face", {"id": "99999"})
        assert entry._extract_face_text([seg]) == "QQ表情99999"

    def test_no_face_segment_returns_empty(self):
        assert entry._extract_face_text([_Seg("text", {"text": "在吗"})]) == ""

    def test_first_face_wins(self):
        segs = [_Seg("face", {"id": "6"}), _Seg("face", {"id": "14"})]
        assert entry._extract_face_text(segs) == "害羞"
