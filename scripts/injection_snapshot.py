#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓一份「19 个注入块实际注出了什么」的快照（重构前/后逐字对比用）。

为什么要它：把文案从 `core.py` 等 4 个 py 文件搬进 `persona/injection/` 时，
**唯一的硬判据是「输出逐字不变」**。人眼对着两段中文找不出少了一个逗号，
所以先把现状抓下来，改完再抓一次，`diff` 说话。

时间/天气/农历那几项**必须冻住**，否则两次跑的分钟数不同，比的就不是文案了。

用法：
    python scripts/injection_snapshot.py -o outputs/eval/injection_golden.json
    python scripts/injection_snapshot.py -o /tmp/after.json   # 改完再跑，然后 diff
    python scripts/injection_snapshot.py --group -o /tmp/group.json   # 群聊场景单独抓一份
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nonebot  # noqa: E402
from nonebot.adapters.onebot.v11 import Adapter as ONEBOT_V11Adapter  # noqa: E402

nonebot.init()
nonebot.get_driver().register_adapter(ONEBOT_V11Adapter)
nonebot.load_plugins("src/plugins")

from src.plugins.chatbot import core, context_probe  # noqa: E402
from src.plugins.chatbot.retrieval import RetrievalItem  # noqa: E402


def _freeze_clock():
    """把「现在几点/农历/天气/直播状态」钉死——它们是唯一会自己变的部分。

    ⚠️ `_format_time` 只管北京那半句；「澳洲当地时间 HH:MM」是函数里直接读 `datetime.now`
    拼的，**绕不过去**，所以最后还要用 `_normalize` 把那个分钟数抹掉。
    """
    context_probe._format_time = lambda dt: "FAKE_TIME"
    context_probe._format_lunar = lambda dt: "FAKE_LUNAR"
    context_probe._weather_line = lambda city: "天气：FAKE"
    context_probe._live_status_text = lambda: "灰泽满现在没有在直播"


_TIME_RE = re.compile(r"澳洲当地时间 \d{2}:\d{2}")


def _normalize(blocks):
    return [_TIME_RE.sub("澳洲当地时间 HH:MM", b) for b in blocks]


def _args(**over):
    base = dict(
        user_msg="今天几点的播呀",
        fused_items=[
            RetrievalItem(source="behavior", item_id="b1", score=0.9,
                          text="【行为】被追问时不绕圈子，给个干脆答案。"),
            RetrievalItem(source="corpus", item_id="c1", score=0.8,
                          text="灰泽满上周五鸽了直播，第二天自己先笑。"),
        ],
        user_history=["user：在吗", "灰泽满：干嘛", "user：问你个事"],
        batch_summary="用户连发了三条问直播时间",
        preference_items=[{"category": "食物", "text": "爱吃椰子鸡"}],
        core_stories=[{"category": "写作", "text": "乌色月是她写的小说"}],
        session_context="当前话题：问今晚直播时间（10 分钟前开始）",
        query_hint="用户问灰泽满今晚几点开播",
        group_context="在场成员：绿冻A、绿冻B；群内近况：在聊新游戏",
        prev_session_note="【上次聊过】3 小时前你们聊到「加粉丝群」。（隔了这么久，**这不是现在正在聊的话题**）",
        same_request={"count": 3, "what": "一个晚安吻"},
        profile_context="【和这个绿冻的相处方式】只调整相处方式。\n用户低落时先接住情绪。",
        commitment_context="【相关未完成约定】答应过给他念一段乌色月",
        emotion_context="【用户此刻的情绪】有点失落",
    )
    base.update(over)
    return base


def capture(group: bool = False) -> dict:
    _freeze_clock()
    out = {}

    msgs = core.build_message_list(**_args(is_group=group))
    out["全量(普通短消息)"] = _normalize([m["content"] for m in msgs if m["role"] == "system"])

    # 纯表情：user_msg 本身要是表情，才会走 emoji 变体（is_emoji_msg 看的是正文，不是 hint）
    msgs = core.build_message_list(**_args(user_msg="[表情：可怜]", query_hint="用户发了个可怜的表情",
                                           is_group=group))
    out["全量(纯表情)"] = _normalize([m["content"] for m in msgs if m["role"] == "system"])

    msgs = core.build_message_list(**_args(user_history="（上一条记忆）", is_group=group))
    out["历史=非列表分支"] = _normalize([m["content"] for m in msgs if m["role"] == "system"])

    msgs = core.build_message_list(**_args(prev_session_note="", is_group=group))
    out["无上次聊过"] = _normalize([m["content"] for m in msgs if m["role"] == "system"])

    msgs = core.build_message_list(**_args(same_request={"count": 1, "what": "x"}, is_group=group))
    out["没到连续索要"] = _normalize([m["content"] for m in msgs if m["role"] == "system"])

    # 关联背景（作品→作者→喜好）：只验"段头确实注进去"，内容由 retrieval 单测覆盖
    msgs = core.build_message_list(**_args(
        association_context="《空中杀人现场》是东野圭吾的作品。",
        association_entities=["空中杀人现场", "东野圭吾"],
        is_group=group))
    out["关联背景"] = _normalize([m["content"] for m in msgs if m["role"] == "system"])

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--group", action="store_true",
                    help="按群聊捕获（注入 00_scene 的群聊文案、group_context 等）")
    args = ap.parse_args()
    data = capture(group=args.group)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    n = sum(len(v) for v in data.values())
    print(f"[OK] {n} 块快照（{'群聊' if args.group else '私聊'}）→ {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())