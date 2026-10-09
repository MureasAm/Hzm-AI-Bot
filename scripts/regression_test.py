"""灰泽满回复回归：虚构弹幕浏览 + **真实翻车案例的自动判据**。

用法：
    python scripts/regression_test.py                 # 跑一遍内置弹幕，打印回复（人工看）
    python scripts/regression_test.py --ab            # A/B：V0(无样本) vs V1(有样本) 对比
    python scripts/regression_test.py --check         # ★判据回归：跑 scripts/regression_cases.json
    python scripts/regression_test.py --check --case no_old_fact_as_current_reason

`--check` 是这套工具的重点：它把**真实翻过的车**（记录.txt / docs/历史/待办清单.md 里归档的）
固化成**确定性判据**（禁词 / 开头同质率 / 逐字复读率），跑一次就知道有没有回退。

⚠️ 为什么判据要确定性、不用 LLM 当裁判：调研文档 §2.4 实测——LLM judge 对
"话题沾边但写错了"的答案接受率高达 62.81%，是"具体但错误"策略的约 6 倍。
**宽松的裁判比没有裁判更糟**，因为它给的是通过。

说明：
- 使用临时记忆文件，不污染线上 user_memory/short_term.json 与 long_term.json
- 长期记忆提取被禁用（只测回复质量，不测记忆副作用）
- **--check 每次跑都用不同 user_id**：否则同一会话的短期记忆会把"她上一轮说过的话"
  喂回来，同质率就测不准了（测出来的是复读自己，不是复读数据层）
"""
import asyncio
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_FILE = PROJECT_ROOT / "scripts" / "regression_cases.json"
# 她本人的真实私聊回复：判"措辞固化"时要拿它当基线（见 _baseline_openers）

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 虚构弹幕：覆盖典型触发场景
DANMAKU = [
    "小满你昨天唱歌好好听！声音好甜啊",
    "说好的八点直播呢？都八点四十了！",
    "今天有什么好玩的事吗？",
    "好几天没见你了，有点想你",
    "你怎么又鸽了，不是说这周不鸽吗",
    "小满当我女朋友吧（羞涩）",
    "你最近在忙什么啊，动态也不发",
    "感觉你今天声音有点累，好好休息啊",
    "你喜欢什么类型的呀？",
    "要不要一起去吃火锅？",
    "你怎么这么可爱！",
    "晚安小满，早点睡",
]


def _init():
    """初始化 NoneBot 并加载插件，返回 core 模块。

    顺序很关键：必须先 init + load_plugins，再 import core，
    否则 chatbot 包被提前导入会导致 load_plugins 报错。
    """
    import nonebot
    from nonebot.adapters.onebot.v11 import Adapter as ONEBOT_V11Adapter
    nonebot.init()
    driver = nonebot.get_driver()
    driver.register_adapter(ONEBOT_V11Adapter)
    nonebot.load_plugins("src/plugins")

    import src.plugins.chatbot.core as core
    # 禁用长期记忆提取副作用（只测回复质量）。
    # ⚠️ **v1 / v2 两路都要禁**：只禁 v1 的话，v2 提取（2026-10-10 起默认开）会往
    #    **真实** user_memory/memory_v2_shadow.json 里写一堆 regr_* 测试用户。
    core.update_memory_task = lambda *a, **k: asyncio.sleep(0)
    core.update_memory_v2_shadow_task = lambda *a, **k: asyncio.sleep(0)

    # 把记忆文件指向临时目录，避免污染线上数据
    tmp = tempfile.mkdtemp(prefix="hzm_test_")
    import memory_manager as mm
    mm.use_storage(Path(tmp))
    import src.plugins.chatbot.memory as mem
    mem.use_storage(Path(tmp))
    # v2 的影子库同样指到临时目录（否则注入会读到真实库里的真人记忆）
    import src.plugins.chatbot.memory_v2 as mv2
    core._MEMORY_V2_STORE = mv2.ShadowMemoryStore(
        state_file=Path(tmp) / "memory_v2_shadow.json",
        audit_file=Path(tmp) / "memory_v2_audit.jsonl",
    )
    return core


async def run_batch(core, user_id: str, danmaku: list = None) -> dict:
    results = {}
    for msg in (danmaku if danmaku else DANMAKU):
        reply = await core.handle_chat(user_id, msg)
        results[msg] = reply
    return results


def _load_danmaku(danmaku_file=None) -> list:
    """加载弹幕列表。缺省用内置虚构弹幕。支持 JSON 数组或每行一条的文本。"""
    global DANMAKU
    if not danmaku_file:
        return DANMAKU
    p = Path(danmaku_file)
    if p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x) for x in data]
        raise ValueError("弹幕 JSON 必须是字符串数组")
    with open(p, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ==================== 判据回归（--check） ====================

_SELF_RE = re.compile(r"^(灰泽满|hzm|小满|满姐)")
_BRACKET_RE = re.compile(r"^[（(][^）)]{0,14}[）)]\s*")
_PUNCT_RE = re.compile(r"^[，。！？~～、…\s]+")


def _opener(reply: str) -> str:
    """取"去掉自称/括号动作/语气标点后的头两个字"，用来量开头同质。

    （愣了一下）灰泽满收到咯 → 收到
    "灰泽满"要剥掉：用名字自称是她的人设，不是同质（曾有防措辞固化错杀过这个，别重蹈）。
    """
    t = _BRACKET_RE.sub("", (reply or "").strip())
    t = _SELF_RE.sub("", t)
    t = _PUNCT_RE.sub("", t)
    return t[:2]


_BASELINE_OPENERS = None


def _baseline_openers():
    """她本人真实私聊回复里**出现过的开头**（`_opener` 之后的头两个字）。

    ⚠️ **为什么需要它**（2026-09-30 实测，修一条会抖的判据）：

    光看"同一个开头占比高"**分不开病和正常**：
    ```
    正常：同一条消息生成 8 次，最高频开头占比  中位 38% ｜ 最高 62%
    当初的 bug：8/14 以「呃」开头 = 57%          → 两者重叠
    阈值 50% 卡在中间 → 同一版本连跑会时挂时过（实测 10 轮挂 1 轮）
    ```
    但那个 bug 有个**准特征**：「呃」在她 721 条真实回复里 **0 次**；
    而正常的高频开头（「啊？」3 次、「天呐」8 次）**她本人都说过**。

    → 所以判据要问的是「**她本人会不会这么开头**」，不是「占比高不高」。

    返回 None 表示算不出来（没有 chat_log / 条数太少）——**这时退回旧行为**
    （只看占比），并且调用方要**明说**这个降级，别让人以为新判据生效了。
    """
    global _BASELINE_OPENERS
    if _BASELINE_OPENERS is not None:
        return _BASELINE_OPENERS or None
    out = set()
    try:
        for line in _common.chatlog_lines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if str(r.get("session", "")).isdigit() and (r.get("reply") or "").strip():
                out.add(_opener(r["reply"]))
    except Exception:
        out = set()
    if len(out) < 100:          # 太少 → 不能用（会把"她没说过"当成"她不会说"）
        _BASELINE_OPENERS = ""
        return None
    _BASELINE_OPENERS = out
    return out


def _load_cases(path=None) -> list:
    p = Path(path) if path else CASES_FILE
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("cases", data) if isinstance(data, dict) else data


def _run_case(core, case: dict) -> dict:
    """跑一个 case N 次，每次换一个 user_id（空短期记忆），返回回复与指标。

    支持的可选字段：
    - `history`：[{"user": …, "reply": …}, …] 先垫进短期记忆再问最后那句。
      **多轮场景必须用它**——"被追问时会不会改口/复读"这类问题，单轮问不出来。
    - `is_group`：true 时按群会话跑（user_id 当群号），用来测群聊行为（如接话门）。
    """
    from src.plugins.chatbot import memory  # 懒导入：要在 _init() 之后
    cid = case["id"]
    runs = int(case.get("runs", 3))
    is_group = bool(case.get("is_group"))
    history = case.get("history") or []

    def one(i: int) -> str:
        uid = f"regr_{cid}_{i}"
        for h in history:  # 垫历史：她"记得"前面发生过什么
            memory.append_user_history(uid, h.get("user", ""), h.get("reply", ""))
        return asyncio.run(core.handle_chat(uid, case["user"], is_group=is_group))

    replies = [one(i) for i in range(runs)]
    openers = Counter(_opener(r) for r in replies)
    top_opener, top_n = openers.most_common(1)[0]
    return {
        "replies": replies,
        "opener": top_opener,
        "homogeneity": top_n / len(replies),
        "verbatim": max(Counter(replies).values()),
    }


def core_is_fallback(reply: str) -> bool:
    """这一条是不是"没生成出来"的兜底文案。懒导入，避免 _init() 之前碰 core。"""
    from src.plugins.chatbot.core import is_fallback_reply
    return is_fallback_reply(reply)


def _judge(case: dict, res: dict) -> list:
    """返回失败原因列表（空 = 通过）。"""
    fails = []

    # 兜底文案（生成失败）不是"回复质量差"，是**这一轮压根没生成出来**。
    # 不单独报的话，判据会拿一句故障文案当普通回复去判，结论是假的。
    fb = [r for r in res["replies"] if core_is_fallback(r)]
    if fb:
        fails.append(f"{len(fb)} 条是兜底文案（生成失败），这轮结论不可信：{fb[0][:30]}")
    for word in case.get("forbid", []):
        hit = [r for r in res["replies"] if word in r]
        if hit:
            fails.append(f"出现禁词「{word}」×{len(hit)}（例：{hit[0][:40]}）")

    # 正则判据：禁词表表达不了的（如"灰泽满…她"这种自指错误、"括号用了 3 个"）
    pat = case.get("forbid_pattern")
    if pat:
        rx = re.compile(pat)
        hit = [r for r in res["replies"] if rx.search(r)]
        if hit:
            fails.append(f"命中禁用模式 /{pat}/ ×{len(hit)}（例：{hit[0][:40]}）")

    req = case.get("require_any", [])
    if req:
        miss = [r for r in res["replies"] if not any(w in r for w in req)]
        if miss:
            fails.append(f"{len(miss)} 条回复一个都没含 {req}（例：{miss[0][:40]}）")

    cap = case.get("opener_homogeneity_max")
    if cap is not None and res["homogeneity"] > cap:
        base = _baseline_openers()
        if base is None:
            # 算不出她的开头基线（没 chat_log）→ **退回旧行为**，但要说清是降级了
            fails.append(f"开头同质 {res['homogeneity']:.0%} > 上限 {cap:.0%}"
                         f"（{res['opener']}；⚠️ 无 chat_log，只能按占比判——可能误伤）")
        elif res["opener"] not in base:
            # ★ 她**本人从来没用过**这个开头，生成里却反复出现 → 这才是"措辞固化"
            fails.append(f"开头同质 {res['homogeneity']:.0%}（{res['opener']}）"
                         f"**且「{res['opener']}」她本人从没这么开过头** → 措辞固化")
        # else：她本人也这么说 → 占比高只是这条消息的正常波动 → 放过（见 _baseline_openers）

    vr = case.get("verbatim_repeat_max")
    if vr is not None and res["verbatim"] > vr:
        fails.append(f"逐字复读 {res['verbatim']} 条 > 上限 {vr}")

    return fails


def check(case_id=None, cases_file=None, out_dir=None) -> int:
    """跑判据回归。返回失败数（0 = 全过）。"""
    cases = _load_cases(cases_file)
    if case_id:
        cases = [c for c in cases if c.get("id") == case_id]
        if not cases:
            print(f"❌ 没有 id 为 {case_id!r} 的 case")
            return 1

    out = Path(out_dir) if out_dir else Path("outputs/eval/regression")
    out.mkdir(parents=True, exist_ok=True)
    core = _init()

    lines, failed, failed_cases = [], 0, 0
    for case in cases:
        res = _run_case(core, case)
        fails = _judge(case, res)
        failed += len(fails)
        if fails:
            failed_cases += 1
        flag = "FAIL" if fails else "PASS"
        header = f"[{flag}] {case['id']}  ({case.get('desc', '')})"
        print(header)
        for r in res["replies"]:
            print(f"        · {r}")
        for f in fails:
            print(f"        ✗ {f}")
        print()
        lines += [header] + [f"    · {r}" for r in res["replies"]] \
                 + [f"    ✗ {f}" for f in fails] + [""]

    print("=" * 46)
    # ⚠️ 原来这里写的是 `len(cases) - int(bool(failed))` —— 任何一条失败都印成 (N-1)/N，
    # 看起来像"只挂了一条"，其实可能挂了一片。现在按**失败用例数**算。
    print(f"case 通过率: {len(cases) - failed_cases}/{len(cases)}"
          f"   失败判据数: {failed}")
    (out / "check.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ 明细已保存: {out / 'check.txt'}")
    return failed


def run(ab_mode=False, danmaku_file=None, out_dir=None):
    """参数化入口（供 run_tool 调用）。

    ⚠️ `ab_mode` 已于 2026-10-04 失效：它原来是"关掉声音样本 vs 有样本"的 A/B，
    而声音样本通道已整体删除（见 retrieval.py 顶部说明）——没有可关的东西了。
    现在传 ab_mode=True 会直接报错退出，而不是给出一份假的对比。
    """
    global DANMAKU
    danmaku = _load_danmaku(danmaku_file)
    out = Path(out_dir) if out_dir else Path("outputs/eval/regression")
    out.mkdir(parents=True, exist_ok=True)

    core = _init()

    if ab_mode:
        raise SystemExit(
            "regression --ab 已失效：它对比的是'有/无声音样本'，"
            "而 voice_samples 通道已于 2026-10-04 删除。请直接跑 --check。")

    v1 = asyncio.run(run_batch(core, "test_user", danmaku))

    lines = ["========== 当前版本（骨架 + 情景样本 + 融合检索）=========="]
    for msg, reply in v1.items():
        lines.append(f"\n💬 {msg}\n↪ {reply}")
    text = "\n".join(lines)
    print(text)
    (out / "full.txt").write_text(text, encoding="utf-8")
    print(f"\n✅ 输出已保存: {out / 'full.txt'}")
    return


def main():
    if "--check" in sys.argv:
        case_id = None
        if "--case" in sys.argv:
            i = sys.argv.index("--case")
            if i + 1 < len(sys.argv):
                case_id = sys.argv[i + 1]
        sys.exit(1 if check(case_id=case_id) else 0)
    run(ab_mode="--ab" in sys.argv)


if __name__ == "__main__":
    main()
