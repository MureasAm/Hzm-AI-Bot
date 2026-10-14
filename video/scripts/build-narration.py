#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EP01 旁白源数据 + 生成器。

## 核心约定（用户反馈后改的，别再改回去）

**一个 LINES 条目 = 一个完整句子 = 一次 TTS 调用 = 时间轴上的一个单元。**

第一版把"每行 ≤28 字"当成了**朗读单位**，于是把完整句子切成两三个片段分别送 TTS。
后果（用户实测反馈）：
  · 句子被切成两段念 → 语调、停顿都断，**听起来不连贯**；
  · 片段之间的空隙变成"无声阶段"，配着画面就像卡住了；
  · 为了让总长凑够 10 分钟，还得把停顿拉到 700ms，进一步放大不自然。
根因是把**字幕长度**（22 字的排版约束）当成了**朗读长度**。

所以现在：
  · LINES 里只写**整句**（尽量 15~40 字，让 GPT-SoVITS 一次念完一句话）；
  · 字幕折行交给渲染端（`wrapCaption`）与导出的 SRT，**不影响朗读**；
  · 时间轴 = 真实音频长度 + 小停顿，**不再为了凑时长而拉伸**（总长随内容自然浮动）。

## 时间轴

`redistribute-narration.mjs` 先给一个基于时长模型的初版（音频还没生成时用），
音频生成后由 `fit-narration-to-audio.mjs` 按**实测时长**钉准。

    python video/scripts/build-narration.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "video" / "src" / "episodes" / "ep01" / "ep01-narration.json"

TITLE = "第一集：人格核心与完整注入链路"

README = [
    "旁白视角（规格 §2.1）：角色感与转场用灰泽满第一人称；架构事实用项目作者/工程视角。",
    "音色始终是灰泽满。每 20~30 秒最多一次轻吐槽（规格 §2.1）。",
    "**一行 = 一个完整句子 = 一次 TTS 调用**（不要为了字数把句子切开，那样念出来不连贯）。",
    "字幕折行由渲染端负责；总时长随内容自然浮动，不为了凑 10 分钟而拉长停顿。",
    "subtitleHighlight：只对核心词做短高亮（规格 §13.3）。",
]

# (id, 章节, 标题, 副标题, visual)
SCENES = [
    ("cold-open", "01 · 三板真实回复", "同一个模型，三种上下文",
     "输入完全相同，改变的只有它当时看到的东西", "cold-open"),
    ("system-prompt", "02 · 稳定骨架", "system_prompt.txt",
     "决定她始终是谁，以及表达的边界", "system-prompt"),
    ("behavior-core", "03 · 情景行为", "behaviors.json",
     "情景 → 反应 → 例句", "behavior-core"),
    ("files-to-messages", "04 · 关键转折", "文件不是模型输入",
     "文件是人维护内容的单位，messages 才是模型接收的单位", "files-to-messages"),
    ("three-domains", "05 · 三类内容", "人格、事实、现场",
     "她会怎样反应 · 她知道什么 · 此刻发生了什么", "three-domains"),
    ("four-gates", "06 · 四种判别", "四个问题，四种判别",
     "每轮都需要吗 · 提到了吗 · 意思相近吗 · 是哪种人际情景", "four-gates"),
    ("content-forms", "07 · 文件与内容形式", "什么内容，用什么形态进去",
     "同一份材料，判别方式与注入段都不一样", "content-forms"),
    ("full-trace", "08 · 完整链路", "三条消息穿过真实链路",
     "固定 fixture · 真实代码产生的 trace", "full-trace"),
    ("reply-output", "09 · 输出侧", "生成之后",
     "清洗、语音或文字二选一、回写", "reply-output"),
    ("closing", "10 · 收束", "文件名模型看不见",
     "它读到的是内容的文本形态、role、注入位置和当前对话", "closing"),
]

# 章节时长的**相对权重**（只用来在内容时长之上分配一点呼吸，不是硬时长）。
SCENE_WEIGHT = {
    "cold-open": 45,
    "system-prompt": 50,
    "behavior-core": 65,
    "files-to-messages": 35,
    "three-domains": 50,
    "four-gates": 80,
    "content-forms": 135,
    "full-trace": 90,
    "reply-output": 32,
    "closing": 23,
}

# ─────────────────────────────────────────────────────────────────────
# 旁白：(sceneId, 完整句子, [字幕高亮词])
# 每一条都是**一次 TTS 调用的单位**。写完跑 build 会检查是否过长。
# ─────────────────────────────────────────────────────────────────────
LINES = [
    # ── 01 三板真实回复（§4）────────────────────────────────────────
    ("cold-open", "同一句话，同一个模型，我把它发三遍。", ["同一个模型"]),
    ("cold-open", "三遍的结果完全不同，而变的不是模型。", []),
    ("cold-open", "用户消息是同一句：你今天唱得特别好，我真的很喜欢。", []),
    ("cold-open", "左边这栏，只给模型这一句，什么背景都没有。", []),
    ("cold-open", "它礼貌地道谢，然后开始介绍自己是人工智能。", ["礼貌回应"]),
    ("cold-open", "中间这栏，多了一份基础提示词，她会压评价了，但还是有点飘。", ["稳定骨架"]),
    ("cold-open", "右边这栏，再多一条 behavior：被夸时嘴硬否认。", ["behavior"]),
    ("cold-open", "先挡回去，再别扭一下，这个动作就是灰泽满。", ["完整反应动作"]),

    # ── 02 稳定骨架（§5.1）──────────────────────────────────────────
    ("system-prompt", "先别急着看那十几段上下文。", []),
    ("system-prompt", "最核心的其实就两件事：我是谁，还有这种时候我会怎么做。", ["人格"]),
    ("system-prompt", "第一份是 system_prompt，稳定骨架，每一轮都在场。", ["常驻"]),
    ("system-prompt", "它规定的是不应该随话题变化的东西：身份、说话节奏、自我称呼。", []),
    ("system-prompt", "它擅长保证她始终是同一个人。", []),
    ("system-prompt", "它不适合穷举每一种具体的人际情景，也不是越长越好。", []),
    ("system-prompt", "所以不管用户说什么，这张骨架都不动，它保证的是不变。", []),
    ("system-prompt", "至于变化，交给第二份文件。", []),

    # ── 03 情景行为（§5.2、§5.3）────────────────────────────────────
    ("behavior-core", "第二份是 behaviors.json，我把它拆成三段：情景、反应、例句。", ["behavior"]),
    ("behavior-core", "情景，说的是用户创造了什么处境；这一条是用户真诚夸她唱得好。", []),
    ("behavior-core", "反应，说的是她往哪个方向走：先把评价压低或者挡回去，被夸真了再别扭一下，最后漏一点软话。", []),
    ("behavior-core", "例句，就是她真实这么说过的几句，挂在这个情景底下。", []),
    ("behavior-core", "情景负责选对抽屉，反应负责告诉模型往哪走，", []),
    ("behavior-core", "例句负责告诉它，这个动作由灰泽满做出来是什么质感。", []),
    ("behavior-core", "这些例句就是 few-shot，它不是让模型复制句子。", []),
    ("behavior-core", "它给的是示范，示范必须绑在正确的情景上，而且行为不是固定回复表。", []),
    ("behavior-core", "例句影响的是节奏和动作形态，最终那句话还是根据用户当前消息生成。", []),
    ("behavior-core", "以前这些职责拆在更多文件里，现在稳定内容进骨架，示范按情景并进 behavior。", ["behavior"]),

    # ── 04 关键转折（§6）────────────────────────────────────────────
    ("files-to-messages", "到这里要拐一个弯，也是整集最关键的一句。", []),
    ("files-to-messages", "文件名，模型是看不见的。", ["messages"]),
    ("files-to-messages", "它不知道有 system_prompt 这个文件，也不知道 behaviors 在哪一层目录。", []),
    ("files-to-messages", "文件是我们维护内容的单位，messages 才是模型真正接收的单位。", ["messages"]),
    ("files-to-messages", "中间隔着三个动作：读取、判断、格式化。", []),

    # ── 05 三类内容（§7）────────────────────────────────────────────
    ("three-domains", "接下来把这些内容分成三类，全片颜色都按这个来。", []),
    ("three-domains", "第一类，人格，说的是她会怎样反应，就是刚才那两份。", ["人格"]),
    ("three-domains", "第二类，事实，说的是她知道什么。", ["事实"]),
    ("three-domains", "词是什么意思、她一贯的偏好、最深的经历、过去直播里发生过的事、这周的排班表、固定的梗。", []),
    ("three-domains", "第三类，现场，说的是此刻发生了什么。", ["现场"]),
    ("three-domains", "用户记忆、会话主题、最近几轮原文、时间、天气、直播状态、群聊、图片。", []),
    ("three-domains", "关键是其他层补的是事实、记忆和现场，不是另造一套人格。", []),

    # ── 06 四种判别（§8）────────────────────────────────────────────
    ("four-gates", "内容有了，下一个问题是怎么判断这一轮需不需要它。", []),
    ("four-gates", "这块有四种判别，每种前面都是一个很朴素的问题。", []),
    ("four-gates", "第一种，问每轮都需要吗；需要就常驻，比如基础提示词、周表、当前时间。", ["常驻"]),
    ("four-gates", "第二种，问消息里明确提到了吗；靠关键词和正则，比如名词库、偏好、语料钩子、梗库的粗筛。", ["关键词"]),
    ("four-gates", "第三种，问意思相近吗；用向量召回，比如过去说过的那些事、还有最深刻的经历。", ["向量"]),
    ("four-gates", "第四种，问这是哪种人际情景、候选是不是真的相关；这个交给模型判断。", ["LLM"]),
    ("four-gates", "命中就点亮、进入待注入，没命中就保持灰色，放进本轮不注入。", []),
    ("four-gates", "但别理解成每个文件只用一种判别，有些路径是组合的。", []),
    ("four-gates", "语料会先有语义候选，再用词面补位，最后过一遍模型确认。", []),
    ("four-gates", "梗库是关键词先粗筛，再做语境确认；行为以情景分类为主，另有确定性兜底。", []),

    # ── 07 文件与内容形式（§9，按用户要求：列全、不讲记忆）──────────
    ("content-forms", "接下来这一段不逐个念文件，只回答一件事：什么内容用什么形态进去。", []),
    ("content-forms", "system_prompt：常驻，内容是稳定说明，注入成第一条 system message。", ["常驻"]),
    ("content-forms", "behaviors：看当前消息的人际情景，内容是情景反应例句，注入成一段行为指令。", ["behavior"]),
    ("content-forms", "语料：向量候选加词面补位，再过模型确认；它装的是过去发生过的事实陈述。", []),
    ("content-forms", "它不是一个可以直接复用的当前回答，注入成她经历过的相关背景。", []),
    ("content-forms", "作用是让她记得经历，但不能拿旧事去编今天的原因。", []),
    ("content-forms", "名词库、偏好、核心经历、周表，这些事实层各有各的判别方式。", []),
    ("content-forms", "名词库让她懂词，偏好是她一贯的样子，核心经历是最深的那几条。", []),
    ("content-forms", "周表是当前排班，梗库是固定该怎么答，命中就直接发出去。", []),
    ("content-forms", "还有一类只在最终 messages 里露面：时间、群聊现场、图片描述、批量归纳、短消息补全、连续索要次数。", ["messages"]),
    ("content-forms", "这些只标来源和注入段名，不再展开内部实现。", []),
    ("content-forms", "所以真的不是把所有文件都塞进去。", []),
    ("content-forms", "每一层都有机会判断，而不相关的内容能够保持沉默。", []),

    # ── 08 完整链路演示（§10）───────────────────────────────────────
    ("full-trace", "现在把整条链完整跑一遍。", []),
    ("full-trace", "人物和数据都是虚构的，但走的是真实代码。", []),
    ("full-trace", "周四晚上六点四十，广州小雨，用户阿澈刚下班，连着发了三条消息，读秒窗口让它们合成一批。", []),
    ("full-trace", "基础提示词常驻；周表命中，周四十九点开播；名词库当场命中，满神和绿冻都在里面。", ["常驻"]),
    ("full-trace", "核心经历向量命中了作家梦，也就是乌色月这一条；偏好命中了下雨天。", ["向量"]),
    ("full-trace", "行为分类把情景判成被推着要求时打太极，于是行为指令整段亮起。", ["LLM"]),
    ("full-trace", "语料召回了九条候选，模型确认后只留下一条，其余八条安静退场。", []),
    ("full-trace", "长期记忆给名字和城市，会话记忆给正在磨她念书这件事。", []),
    ("full-trace", "短期记忆带上她前两次都没答应，时间天气带上周四和广州小雨。", []),
    ("full-trace", "连续索要判定第三次，批量归纳说明这三条属于同一批。", []),
    ("full-trace", "最后拼出十五段 messages。", ["messages"]),

    # ── 09 输出侧（§11）─────────────────────────────────────────────
    ("reply-output", "模型生成之后，还没结束。", []),
    ("reply-output", "先做防复读检测，必要时重新生成，然后清洗掉多余括号和换行。", []),
    ("reply-output", "再决定语音还是文字，语音成功就不再发拆分文字。", []),
    ("reply-output", "然后写回短期记忆和长期记忆，这一集到这里就够了。", []),

    # ── 10 收束（§1.1、§18；预告指向第二集＝记忆专题）────────────────
    ("closing", "基础提示词负责让她始终是同一个人；", ["人格"]),
    ("closing", "behavior 负责让这个人在不同情景下做出不同反应。", ["人格"]),
    ("closing", "文件名模型看不见，它读到的是内容的文本形态、role 和注入位置。", ["messages"]),
    ("closing", "下一集讲记忆，她为什么还记得你之前说过什么。", ["fact"]),
]


def estimate_ms(text: str, speed: float = 1.2) -> int:
    """按**实测拟合**的时长模型估算朗读时长（毫秒）。

    模型来自真实 HZM-SPEAK 合成结果的最小二乘拟合：
        audio_ms(1.0) ≈ 332 + 200×汉字 + 355×标点 + 71×英文字母
    提速时只有"发音段"会变短，**固定开销与标点停顿不变**：

        audio_ms(speed) ≈ (332 + 355×标点) + (200×汉字 + 71×字母) / speed
    """
    han = len(re.findall(r"[\u4e00-\u9fa5]", text))
    punc = len(re.findall(r"[，。、；：？！…]", text))
    lat = sum(len(m) for m in re.findall(r"[A-Za-z0-9_]+", text))
    return int((332 + 355 * punc) + (200 * han + 71 * lat) / max(0.5, speed))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # 场景边界先占位（真实时间轴由 redistribute / fit-voice 算）
    total_weight = sum(SCENE_WEIGHT[s[0]] for s in SCENES)
    want_total = sum(estimate_ms(t) for _, t, _ in LINES) + 1200 * len(LINES)
    scenes = []
    cursor = 0
    for sid, kicker, title, subtitle, visual in SCENES:
        dur = int(want_total * (SCENE_WEIGHT[sid] / total_weight))
        scenes.append({
            "id": sid, "startMs": cursor, "endMs": cursor + dur,
            "kicker": kicker, "title": title, "subtitle": subtitle, "visual": visual,
        })
        cursor += dur

    counters: dict[str, int] = {}
    lines = []
    for scene_id, text, highlight in LINES:
        counters[scene_id] = counters.get(scene_id, 0) + 1
        lines.append({
            "id": f"{scene_id}-{counters[scene_id]}",
            "sceneId": scene_id,
            "startMs": 0,
            "endMs": 1000,
            "text": text,
            "subtitleHighlight": highlight,
            "estMs": estimate_ms(text),
        })

    payload = {
        "title": TITLE,
        "episode": "ep01",
        "fps": 30,
        "width": 1920,
        "height": 1080,
        "durationMs": cursor,
        "narrationVoice": "灰泽满 GPT-SoVITS（HZM-SPEAK），全程不切换中性配音",
        "readme": README,
        "scenes": scenes,
        "lines": lines,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 报告：整句是否过长（太长就再拆一句，而不是拆朗读片段）
    ests = [estimate_ms(l["text"]) for l in lines]
    longest = sorted(lines, key=lambda l: -estimate_ms(l["text"]))[:8]
    too_long = [l for l in lines if estimate_ms(l["text"]) > 9000]
    print(f"场景 {len(scenes)}  句 {len(lines)}  （每句 = 一次 TTS 调用）")
    print(f"预计旁白纯时长 {sum(ests) / 1000:.1f}s，最长句 {max(ests)}ms，最短句 {min(ests)}ms")
    print("\n最长的 8 句：")
    for l in longest:
        print(f"   {l['id']:<22} {estimate_ms(l['text']):>5}ms  {l['text']}")
    if too_long:
        print(f"\n⚠️ {len(too_long)} 句超过 9 秒，建议再拆成两句（拆句 = 拆朗读单位，要谨慎）：")
        for l in too_long:
            print(f"   {l['id']:<22} {estimate_ms(l['text']):>5}ms  {l['text']}")
    print(f"\n✅ {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
