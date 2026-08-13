"""灰泽满回复"灵性"回归测试（虚构弹幕 A/B 对比）。

用法：
    python scripts/regression_test.py            # 只跑 V1（有声音样本）
    python scripts/regression_test.py --ab       # A/B：V0(无样本) vs V1(有样本) 对比

说明：
- 弹幕为虚构，覆盖被夸/被催播/被越界/表达依赖/日常等典型场景
- 使用临时记忆文件，不污染线上 user_memory/short_term.json 与 long_term.json
- 长期记忆提取被禁用（只测回复质量，不测记忆副作用）
"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

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
    ab_mode = "--ab" in sys.argv
    run(ab_mode=ab_mode)


if __name__ == "__main__":
    main()
