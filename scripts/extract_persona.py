#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""专属 behaviors 挖掘工具：从 convert-to-chat 产物提取新格式行为规则。

新格式（persona_behaviors.json）：
  {name, trigger, response, samples[], evidence[]}
  - name       行为名（如"被夸时嘴硬否认"）
  - trigger    触发情境（向量匹配用，要具体）
  - response   她的典型反应（给模型的行为指令，一句话）
  - samples    真人原话示范（few-shot，模型照这个学腔调）
  - evidence   溯源（素材里的原句佐证）

铁律（写死在代码里）：
- samples 必须【逐字】来自 convert-to-chat 产物，禁止模型改写/润色/代拟
- validation 用"reply 逐字匹配产物"校验，防模型自创
- 输入必须是 convert-to-chat 产物（已分离读弹幕 vs 回答），不是 cleaned

场景聚合：提取时模型先标 scene（归到现有 8 大行为场景），
相近 trigger 会自动聚到同一场景，避免 behaviors 拆太细。

用法：
    python scripts/convert_to_chat.py -i cleaned_0807.json -o outputs/transcribe/converted_0807.json
    python scripts/extract_persona.py -i converted_0807.json converted_0726.json ...
             [--out outputs/persona_extract/behaviors_candidates.json]

输出候选 → 人工审批 → 并入 persona/persona_behaviors.json → precompute triggers
"""
import json
import sys
import asyncio
from pathlib import Path
from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env.prod"
OUT_DIR = PROJECT_ROOT / "outputs" / "persona_extract"
DEFAULT_OUT = OUT_DIR / "behaviors_candidates.json"

MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com/v1"
THINKING_DISABLED = {"extra_body": {"thinking": {"type": "disabled"}}}
BATCH_SIZE = 50

# 现有 behaviors 的 8 大场景（对齐 persona_behaviors.json 的 name 归类）
SCENE_GROUPS = {
    "被夸": "被夸时嘴硬否认",
    "被质疑": "被质疑时心虚辩解",
    "被戳穿": "被质疑时心虚辩解",
    "被越界": "被越界时冷静推开",
    "被调戏": "被越界时冷静推开",
    "冷场": "冷场时主动自爆填补",
    "立Flag": "立Flag后秒打脸",
    "承诺": "立Flag后秒打脸",
    "抛梗": "主动抛梗与预判调侃",
    "推进": "主动抛梗与预判调侃",
    "感性": "感性流露后迅速缩回",
    "脆弱": "感性流露后迅速缩回",
    "失约": "失约被抓包时滑跪",
    "催播": "失约被抓包时滑跪",
    "日常": "日常闲聊（无固定行为模式）",
}

# ==================== 数据加载 ====================

def get_deepseek_key():
    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY"):
                    return line.split("=", 1)[1].replace('"', '').strip()
    return None


def load_converted_pairs(input_paths) -> list:
    """加载 convert-to-chat 产物，归一化为 [{"user","reply"}] 列表。"""
    pairs = []
    for p in input_paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for it in data:
            if not isinstance(it, dict):
                continue
            user = (it.get("user_situation") or "").strip()
            reply = (it.get("reply") or "").strip()
            if user and reply:
                pairs.append({"user": user, "reply": reply})
    return pairs


# ==================== 提取提示词 ====================

EXTRACT_PROMPT = """你是灰泽满直播语料的人格分析师。我给你一批【已分离的对话对】（粉丝说了什么 user + 灰泽满怎么答 reply，全部来自她真实直播素材）。

任务：从这些对话对里提取【稳定的触发-响应行为模式】——她在**某类情境下反复出现**的反应，输出**新格式**行为规则。

# 新格式（每条行为规则）
{{
  "scene": "行为场景（从以下选）：被夸 / 被质疑 / 被戳穿 / 被越界 / 被调戏 / 冷场 / 立Flag / 承诺 / 抛梗 / 推进 / 感性 / 脆弱 / 失约 / 催播",
  "name": "行为名（4-10字，如'被夸时嘴硬否认'）",
  "trigger": "触发情境（向量匹配用，要概括到"这类情境"，如'被粉丝夸时'，不要细到'被夸声音/被夸厉害'）",
  "response": "她的典型反应流程（一句话描述行为，如'先嘴硬否认再用自嘲带过'）",
  "samples": [
    {{"user": "粉丝原话（照抄下面素材）", "reply": "灰泽满原话（照抄下面素材）"}}
  ]
}}

# 铁律（务必遵守）
1. **宁缺毋滥，只提取稳定模式**：只有某类情境在素材里**反复出现、且灰泽满的反应一致**，才值得提取为一条行为规则。闲聊、分享话题、回答具体问题（如"你吃什么""你住哪"）**不是行为模式**，不要提取。
2. **触发要概括、行为要聚合**：被夸声音/被夸厉害/被夸可爱 → 都归"被夸"一条，合并 samples。**严禁**把同一情境拆成多条（那是碎片化，会稀释检索）。
3. **行为 > 标签**：写"被夸时她具体怎么说"，不写"她是嘴硬的人"。
4. **samples 必须逐字来自下面素材**：直接【原封不动挑选】对话对（user 和 reply 都照抄），一个字都不能改、不能润色、不能代拟。挑最有代表性的 2-4 个。
5. 只提取素材里【真实出现】的反应，不推断不脑补。
6. **宁可少**：一批素材里有多少条**真正的稳定行为模式**就输出多少条，通常 2-5 条。凑不出来就少输出，不要硬编。

# 输出 JSON
{{
  "behaviors": [
    {{
      "scene": "被夸",
      "name": "被夸时嘴硬否认",
      "trigger": "被粉丝夸时",
      "response": "先嘴硬否认再用自嘲带过",
      "samples": [{{"user": "粉丝说：你这声音好好听", "reply": "哎呦天啊，真有点好听吐了"}}]
    }}
  ]
}}
只输出 JSON，不要多余内容。

【素材对话对】
{text}"""


# ==================== 调用 ====================

async def extract_batch(client, batch_text: str) -> list:
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": EXTRACT_PROMPT.format(text=batch_text)}],
            temperature=0.3,
            max_tokens=8000,
            **THINKING_DISABLED,
        )
        content = resp.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        result = json.loads(content)
        behaviors = result.get("behaviors", [])
        return [b for b in behaviors if isinstance(b, dict) and b.get("trigger")]
    except Exception as e:
        print(f"⚠️ 提取批次失败: {e}")
        print(f"   原始输出前300字符: {content[:300]}")
        return []


def _normalize(text: str) -> str:
    """归一化用于逐字匹配：去掉空白。"""
    return "".join(text.split())


def validate_samples(behaviors: list, all_pairs: list) -> list:
    """校验 samples 是否逐字来自 convert-to-chat 产物。剔除模型自创/改写的样本。"""
    reply_set = {_normalize(p.get("reply", "")) for p in all_pairs if p.get("reply")}
    user_set = {_normalize(p.get("user", "")) for p in all_pairs if p.get("user")}
    validated = []
    dropped = 0
    for b in behaviors:
        good_samples = []
        for s in b.get("samples", []):
            reply = (s.get("reply", "") or "").strip()
            user = (s.get("user", "") or "").strip()
            if reply and _normalize(reply) in reply_set and user and _normalize(user) in user_set:
                good_samples.append(s)
            else:
                dropped += 1
                print(f"  ⚠️ 剔除非原文样本: U:{user[:18]}… / R:{reply[:18]}…")
        if good_samples:
            b["samples"] = good_samples
            validated.append(b)
        else:
            print(f"  ⚠️ 剔除整条 behavior（无原文样本）: {b.get('name', b.get('trigger',''))[:30]}")
    print(f"  ✅ 校验完成：保留 {len(validated)} 条，剔除 {dropped} 个非原文样本")
    return validated


MAX_SAMPLES_PER_SCENE = 6  # 每个行为场景最多保留的 samples 数（宁缺毋滥）

def merge_behaviors(behaviors: list) -> list:
    """按 scene 聚合：同一场景的多条行为合并成一条，保留多个 samples。

    不同批次的模型可能给同一行为起不同 name（如"被夸时嘴硬否认"vs"被夸时自嘲"），
    但 scene 一致 → 合并成一条。samples 去重累积，每 scene 上限 MAX_SAMPLES_PER_SCENE。
    """
    merged = {}
    for b in behaviors:
        scene = (b.get("scene", "") or "").strip()
        if not scene or scene == "日常":
            continue  # 日常闲聊不是行为模式，丢弃
        name = (b.get("name", "") or "").strip()
        trigger = (b.get("trigger", "") or "").strip()
        response = (b.get("response", "") or "").strip()
        if scene not in merged:
            merged[scene] = {
                "scene": scene,
                "name": name or scene,
                "trigger": trigger,
                "response": response,
                "samples": [],
            }
        else:
            # trigger 拼接去重：只保留最概括的那个（最短的），避免长串变体
            if trigger and trigger not in merged[scene]["trigger"]:
                if len(merged[scene]["trigger"]) > len(trigger) or not merged[scene]["trigger"]:
                    merged[scene]["trigger"] = trigger
            if response and not merged[scene]["response"]:
                merged[scene]["response"] = response
        for s in b.get("samples", []):
            if s not in merged[scene]["samples"]:
                merged[scene]["samples"].append(s)
        # 截断 samples
        merged[scene]["samples"] = merged[scene]["samples"][:MAX_SAMPLES_PER_SCENE]
    return list(merged.values())


# ==================== 主流程 ====================

async def run(input_paths, out_file=None):
    key = get_deepseek_key()
    if not key:
        print("❌ 未找到 OPENAI_API_KEY")
        return
    pairs = load_converted_pairs(input_paths)
    if not pairs:
        print("❌ 没有加载到对话对（请先跑 convert-to-chat 生成 converted_*.json）")
        return

    out = Path(out_file) if out_file else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"📄 加载 {len(pairs)} 个对话对，分批提取行为规则（每批 {BATCH_SIZE}）...")
    client = AsyncOpenAI(api_key=key, base_url=BASE_URL)

    all_behaviors = []
    n_batches = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i:i + BATCH_SIZE]
        text = "\n".join(f"- 粉丝说：{p['user']}\n  灰泽满：{p['reply']}" for p in batch)
        behaviors = await extract_batch(client, text)
        all_behaviors.extend(behaviors)
        print(f"  [批次 {i // BATCH_SIZE + 1}/{n_batches}] 提取 {len(behaviors)} 条")

    merged = merge_behaviors(all_behaviors)
    print(f"🔍 场景聚合后 {len(merged)} 条行为规则")

    # 校验 samples 逐字来自产物（防模型自创措辞）★ 铁律
    print("🔍 校验 samples 是否逐字来自 convert-to-chat 产物...")
    merged = validate_samples(merged, pairs)

    candidates = {"source_files": [str(p) for p in input_paths], "behaviors": merged}
    out.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ 行为候选生成：{out}")
    print(f"   共 {len(merged)} 条（待人工审批）")
    # 按场景分组展示
    from collections import Counter
    scenes = Counter(b.get("scene", "日常") for b in merged)
    print(f"   场景分布: {dict(scenes)}")
    for b in merged:
        print(f"   [{b.get('scene','')}] {b['name']} | trigger={b['trigger'][:30]} | {len(b.get('samples',[]))} 示范")
    print("\n💡 下一步：人工审批后并入 persona/persona_behaviors.json，再跑 precompute triggers")


async def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) < 2:
        print("用法: python scripts/extract_persona.py -i converted_0807.json converted_0726.json ... [--out 输出.json]")
        return
    args = sys.argv[1:]
    out_file = None
    if "--out" in args:
        out_file = args[args.index("--out") + 1]
        # 去掉 --out 及其值，剩余才是输入
        args = args[:args.index("--out")] + args[args.index("--out") + 2:]
    inputs = [a for a in args if not a.startswith("--")]
    await run(inputs, out_file)


if __name__ == "__main__":
    asyncio.run(main())
