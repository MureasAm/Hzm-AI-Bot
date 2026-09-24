"""灰泽满回复回归：虚构弹幕浏览 + **真实翻车案例的自动判据**。

用法：
    python scripts/regression_test.py                 # 跑一遍内置弹幕，打印回复（人工看）
    python scripts/regression_test.py --ab            # A/B：V0(无样本) vs V1(有样本) 对比
    python scripts/regression_test.py --check         # ★判据回归：跑 scripts/regression_cases.json
    python scripts/regression_test.py --check --case no_old_fact_as_current_reason

`--check` 是这套工具的重点：它把**真实翻过的车**（记录.txt / 待办清单.md 里归档的）
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
    # 禁用长期记忆提取副作用（只测回复质量）
    core.update_memory_task = lambda *a, **k: asyncio.sleep(0)

    # 把记忆文件指向临时目录，避免污染线上数据
    tmp = tempfile.mkdtemp(prefix="hzm_test_")
    import memory_manager as mm
    mm.MEMORY_FILE = Path(tmp) / "long_term.json"
    import src.plugins.chatbot.memory as mem
    mem.MEMORY_FILE = Path(tmp) / "short_term.json"
    return core


async def run_batch(core, user_id: str, danmaku: list = None) -> dict:
    results = {}
    for msg in (danmaku if danmaku else DANMAKU):
        reply = await core.handle_chat(user_id, msg)
        results[msg] = reply
    return results


def _reset_sample_cache():
    """清空检索模块的样本缓存，让环境变量开关生效。"""
    import src.plugins.chatbot.retrieval as ret
    ret._sample_vectors = None


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


def _load_cases(path=None) -> list:
    p = Path(path) if path else CASES_FILE
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("cases", data) if isinstance(data, dict) else data


def _run_case(core, case: dict) -> dict:
    """跑一个 case N 次，每次换一个 user_id（空短期记忆），返回回复与指标。"""
    cid = case["id"]
    runs = int(case.get("runs", 3))
    replies = [asyncio.run(core.handle_chat(f"regr_{cid}_{i}", case["user"]))
               for i in range(runs)]
    openers = Counter(_opener(r) for r in replies)
    top_opener, top_n = openers.most_common(1)[0]
    return {
        "replies": replies,
        "opener": top_opener,
        "homogeneity": top_n / len(replies),
        "verbatim": max(Counter(replies).values()),
    }


def _judge(case: dict, res: dict) -> list:
    """返回失败原因列表（空 = 通过）。"""
    fails = []
    for word in case.get("forbid", []):
        hit = [r for r in res["replies"] if word in r]
        if hit:
            fails.append(f"出现禁词「{word}」×{len(hit)}（例：{hit[0][:40]}）")

    req = case.get("require_any", [])
    if req:
        miss = [r for r in res["replies"] if not any(w in r for w in req)]
        if miss:
            fails.append(f"{len(miss)} 条回复一个都没含 {req}（例：{miss[0][:40]}）")

    cap = case.get("opener_homogeneity_max")
    if cap is not None and res["homogeneity"] > cap:
        fails.append(f"开头同质 {res['homogeneity']:.0%} > 上限 {cap:.0%}"
                     f"（{res['homogeneity'] * len(res['replies']):.0f}/{len(res['replies'])} "
                     f"条以「{res['opener']}」开头）")

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

    lines, failed = [], 0
    for case in cases:
        res = _run_case(core, case)
        fails = _judge(case, res)
        failed += len(fails)
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
    print(f"case 通过率: {len(cases) - int(bool(failed))}/{len(cases)}"
          f"   失败判据数: {failed}")
    (out / "check.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ 明细已保存: {out / 'check.txt'}")
    return failed


def run(ab_mode=False, danmaku_file=None, out_dir=None):
    """参数化入口（供 run_tool 调用）。"""
    global DANMAKU
    danmaku = _load_danmaku(danmaku_file)
    out = Path(out_dir) if out_dir else Path("outputs/eval/regression")
    out.mkdir(parents=True, exist_ok=True)

    core = _init()

    # 先跑"有样本"（当前状态）
    v1 = asyncio.run(run_batch(core, "test_user", danmaku))

    if not ab_mode:
        lines = ["========== 当前版本（有声音样本 + 融合检索）=========="]
        for msg, reply in v1.items():
            lines.append(f"\n💬 {msg}\n↪ {reply}")
        text = "\n".join(lines)
        print(text)
        (out / "full.txt").write_text(text, encoding="utf-8")
        print(f"\n✅ 输出已保存: {out / 'full.txt'}")
        return

    # A/B：V0 关闭样本（环境变量 + 清缓存）
    os.environ["VOICE_SAMPLES"] = "0"
    _reset_sample_cache()
    v0 = asyncio.run(run_batch(core, "test_user", danmaku))

    lines = ["========== A/B 对比（V0 无样本 vs 当前有样本）=========="]
    for msg in danmaku:
        lines.append(f"\n💬 {msg}")
        lines.append(f"  V0（无样本）: {v0[msg]}")
        lines.append(f"  当前（有样本）: {v1[msg]}")
    text = "\n".join(lines)
    print(text)
    (out / "ab.txt").write_text(text, encoding="utf-8")
    print(f"\n✅ 输出已保存: {out / 'ab.txt'}")


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
