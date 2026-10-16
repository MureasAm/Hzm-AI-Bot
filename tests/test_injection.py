# -*- coding: utf-8 -*-
"""注入层（persona/injection/ + injection.py）的结构测试。

为什么值得单独测：这一层是 2026-10-16 从四个 py 文件里抽出来的，**抽的时候靠
「输出逐字不变」验证过一次**（scripts/injection_snapshot.py 抓的快照）。
那之后没有任何东西拦着它再次漂移——改错一个块名、漏一个占位符、
把两个块的序号写重，都只会在某次真聊天里静默少注一段。所以这里把它钉住。
"""
import re
import string
from pathlib import Path

import pytest

from src.plugins.chatbot import injection as inj

EXPECTED_IDS = [
    "identity", "schedule", "profile", "preferences", "core_stories", "group",
    "terms", "session", "prev_session", "behaviors", "corpus", "commitment",
    "history", "same_request", "batch", "query_hint", "now", "mood", "daily",
]


# ---------------------------------------------------------------- 目录结构

def test_all_19_blocks_load():
    blocks = inj.load_blocks()
    assert sorted(blocks) == sorted(EXPECTED_IDS), "块的 id 集合变了"


def test_order_is_1_to_19_and_unique():
    orders = [inj.load_blocks()[b]["order"] for b in EXPECTED_IDS]
    assert sorted(orders) == list(range(1, 20)), "序号必须 1..19 且不重复"


def test_folder_prefix_matches_order():
    """文件夹名的序号 = block.json 的 order —— 否则「按名字排」和「按 order 排」会打架。"""
    for bid, b in inj.load_blocks().items():
        prefix = int(b["dir"].split("_")[0])
        assert prefix == b["order"], f"{bid}: 文件夹 {b['dir']} 与 order={b['order']} 对不上"


def test_order_function_follows_folder_order():
    assert inj.order() == EXPECTED_IDS


# ---------------------------------------------------------------- 每个块的必备字段

@pytest.mark.parametrize("bid", EXPECTED_IDS)
def test_block_has_required_metadata(bid):
    b = inj.block(bid)
    for field in ("id", "order", "title", "when", "source", "_readme"):
        assert b.get(field), f"{bid} 缺字段 {field}"
    assert len(b["_readme"]) > 40, f"{bid} 的 _readme 太短，等于没写"


@pytest.mark.parametrize("bid", EXPECTED_IDS)
def test_block_is_either_strings_or_raw_file(bid):
    b = inj.block(bid)
    if b.get("kind") == "raw_file":
        assert b.get("source"), f"{bid} 是 raw_file 但没有 source"
        return
    strings = b.get("strings")
    assert isinstance(strings, dict) and strings, f"{bid} 既不是 raw_file 也没有 strings"


# ---------------------------------------------------------------- 模板能不能渲染

def _placeholders(tpl: str) -> set:
    return {name for _, name, _, _ in string.Formatter().parse(tpl) if name}


@pytest.mark.parametrize("bid", EXPECTED_IDS)
def test_every_template_renders(bid):
    """占位符都能用假参数填上——**缺参数会在真聊天里 KeyError**，在这里先炸。"""
    b = inj.block(bid)
    if b.get("kind") == "raw_file":
        pytest.skip("raw_file 不套模板")
    for key, tpl in (b.get("strings") or {}).items():
        if key == "behaviors":
            continue          # 那是张表，不是模板
        dummy = {name: "X" for name in _placeholders(tpl)}
        out = tpl.format(**dummy)
        assert out and "{" not in out, f"{bid}.{key} 渲染后还留着花括号"


def test_no_block_template_has_stray_brace():
    """模板里不该有字面的花括号——format 会把它当占位符。"""
    for bid in EXPECTED_IDS:
        b = inj.block(bid)
        for key, tpl in (b.get("strings") or {}).items():
            if key == "behaviors":
                continue
            # 合法占位符之外的 { 或 }
            stripped = re.sub(r"\{\w+\}", "", tpl)
            assert "{" not in stripped and "}" not in stripped, f"{bid}.{key} 有裸花括号"


# ---------------------------------------------------------------- 段头与命名

def test_each_text_block_starts_with_its_header():
    """除了 raw_file，每块的第一句都该是【段头】——出问题时一眼能认出是哪块注的。"""
    for bid in EXPECTED_IDS:
        b = inj.block(bid)
        if b.get("kind") == "raw_file":
            continue
        for key, tpl in (b.get("strings") or {}).items():
            if key in ("behaviors", "item", "contradiction", "fallback", "tail"):
                continue       # 拼接用的碎片，不带头
            assert tpl.startswith("【"), f"{bid}.{key} 没有以段头开头：{tpl[:20]}"


def test_labels_mentioned_in_block_text_are_real():
    """块文案里提到的【X】必须是真存在的段头——历史上骨架写过【行为指令】而代码叫
    【当前情境下的行为指令】，两边谁也不知道对方改了名。"""
    known = set()
    for bid in EXPECTED_IDS:
        b = inj.block(bid)
        if b.get("kind") == "raw_file":
            continue
        for tpl in (b.get("strings") or {}).values():
            if isinstance(tpl, str):
                known.update(re.findall(r"【([^】]{1,12})】", tpl))
    for bid in EXPECTED_IDS:
        b = inj.block(bid)
        for key, tpl in (b.get("strings") or {}).items():
            if not isinstance(tpl, str):
                continue
            if tpl.startswith("【"):        # 段头本身当然要算
                continue
            for ref in re.findall(r"【([^】]{1,12})】", tpl):
                assert ref in known, f"{bid}.{key} 提了个不存在的段头【{ref}】"


# ---------------------------------------------------------------- 周表那行

def test_weekly_row_keeps_the_original_wording():
    """`time` **不归一**——图上写「动态通知」就是「动态通知」。

    旧识图脚本把「没有固定时间」和「动态通知」都写成「无固定时间」，等于把
    「还没定、看动态」这条信息扔掉，模型只会当成「这天不播」（2026-10-16 用户提的）。
    """
    assert inj.weekly_row({"day": "周二", "time": "动态通知"}) == "周二 动态通知"
    assert inj.weekly_row({"day": "周三", "time": "待定"}) == "周三 待定"


def test_weekly_row_appends_topic_then_note():
    assert inj.weekly_row({"day": "周一", "time": "18:00", "topic": "唱歌"}) == "周一 18:00（唱歌）"
    assert inj.weekly_row({"day": "周五", "time": "20:00", "topic": "游戏",
                           "note": "联动"}) == "周五 20:00（游戏／联动）"
    assert inj.weekly_row({"day": "周六", "time": "19:00", "topic": "", "note": ""}) == "周六 19:00"


def test_format_weekly_joins_rows():
    out = inj.format_weekly([
        {"day": "周一", "time": "18:00", "topic": "唱歌"},
        {"day": "周二", "time": "动态通知"},
    ])
    assert out == "周一 18:00（唱歌）、周二 动态通知"


def test_format_weekly_skips_rows_without_day():
    assert inj.format_weekly([{"time": "18:00"}, {"day": "周一", "time": "18:00"}]) == "周一 18:00"


# ---------------------------------------------------------------- 骨架与行为句表

def test_system_prompt_is_readable():
    text = inj.system_prompt()
    assert text.startswith("# 灰泽满")
    assert len(text) > 1500


def test_behavior_sentence_table_covers_the_13_used_keys():
    """相处方式那张表必须覆盖 memory_v2 会产出的键，缺一个就静默少一句。"""
    assert inj.behavior_sentence("support_style", "comfort")
    assert inj.behavior_sentence("self_reference", "first_person")
    assert inj.behavior_sentence("不存在的键", "x") == ""


# ---------------------------------------------------------------- 数据文件都还在

DATA_FILES = [
    ("02_schedule", "schedule.json"),
    ("04_preferences", "preferences.json"),
    ("05_core_stories", "core_stories.json"),
    ("05_core_stories", "core_story_vectors.json"),
    ("07_terms", "terms.json"),
    ("10_behaviors", "behaviors.json"),
    ("10_behaviors", "behavior_keywords.json"),
    ("11_corpus", "statement_final.json"),
    ("11_corpus", "corpus_vectors.json"),
    ("11_corpus", "corpus_keywords.json"),
    ("11_corpus", "corpus_asks.json"),
]


@pytest.mark.parametrize("folder,filename", DATA_FILES)
def test_data_file_lives_in_its_block_folder(folder, filename):
    p = Path(inj.INJECTION_DIR) / folder / filename
    assert p.exists(), f"{p} 不见了——数据必须跟着它服务的块走"


def test_old_folders_are_gone():
    """旧位置必须真的空掉，不然「一份数据两个家」迟早打架。"""
    persona = Path(inj.INJECTION_DIR).parent
    assert not (persona / "behavior").exists(), "persona/behavior/ 该空了"
    assert not (persona / "world").exists(), "persona/world/ 该整个删掉（数据进了 injection/，legendary 进了 decision/）"
    assert not (persona / "media").exists(), "persona/media/ 该整个删掉（stickers 进了 decision/）"
    assert {d.name for d in persona.iterdir() if d.is_dir()} == {"core", "injection", "decision"}