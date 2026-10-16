#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为每条 behavior 从 handoff 问答对里挑例句。

两步：
  ① 确定性筛选（正则打弹幕/打她的回复）——把 3605 对收窄到几十对
  ② 让模型**从候选里挑 + 按私聊语境最小改写**——它不许自己编

## 例句的硬标准（用户 2026-10-16 定）

**必须自足**：把这两句单独拿出来、不给任何上下文，读的人要能明白在说什么。
最要紧的一条：**reply 里提到的具体人物／事件／物品，必须在 user 那句话里出现过**。
反例（现在文件里就有）：`[满神你好厉害] → [别抬举灰泽满了，三毛还是挺厉害的]`
—— 原语境是在聊三毛这个人，泛化之后三毛凭空冒出来，实测模型会原样照搬。

用法：
    python scripts/curate_behavior_samples.py                 # 全部
    python scripts/curate_behavior_samples.py --only 夸        # 只跑名字含某词的
    python scripts/curate_behavior_samples.py --out out.json   # 落盘（缺省只打印）
"""
import argparse
import asyncio
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

from src.plugins.chatbot import memory_v2  # noqa: E402,F401  (拉起插件)
from src.plugins.chatbot.config import _get_clients, _get_model_name  # noqa: E402
from src.plugins.chatbot.constants import THINKING_DISABLED  # noqa: E402

SFT = ROOT / "assets/handoff/handoff-data/a2-sft-round3.jsonl"
BEH = ROOT / "persona" / "injection" / "10_behaviors" / "behaviors.json"

# 每条 behavior 的候选筛选：(打弹幕的) 、(打她回复的)。两个都为真才算候选。
# 只打弹幕太粗——会捞进一堆"弹幕里提了这个词、但她答的是别的事"的对。
# ⚠️ 打回复那一条是**她这个姿态的特征词**，宽一点没关系，宁可多给模型几条。
FILTERS = {
    "被夸时把夸奖调小": (r"好听|好厉害|好棒|好会|好强|太强|牛啊|牛蛙|厉害|好甜|可爱|好美|好萌|优秀",
                   r"也就|别夸|哪里|有点夸张|算不上|不至于|别抬|受宠若惊|是在夸|哪有|算什么|真的吗"),
    "被质疑戳穿时心虚辩解": (r"敷衍|骗|撒谎|假的|你不是说|你又|真正的|是不是没|你不|你怎么",
                   r"确实|好吧|也没有|不承认|真的没有|不是这样|忘了|当时|其实"),
    "被说低俗或擦边时冷脸挡回": (r"黄桃|黄腔|低俗|擦边|涩|搞颜色",
                   r"别|不要|正经|高雅|绿色|好女孩|不可以|不聊"),
                   r"没有|不亲|嫁不了|还小|不行|不能|不给|别|算了"),
    "立Flag后立刻给自己留退路": (r"早睡|不鸽|打卡|说好|坚持|明天一定|能做到|准时",
                   r"争取|尽量|看情况|不一定|再说|困难|说不准|可能|应该|哪能"),
    "被推着要求时打太极": (r"你能|能不能|帮我|教我|发段|说一句|会来|可以吗|试试",
                   r"有机会|看看|再说|不一定|不会|哪会|做不到|不知道|不太"),
    "被真心对待时别扭缩回": (r"谢谢你|一直在|不开心|孤单|孤独|在意|陪着我|谢谢",
                    r"干嘛|别|肉麻|不知道说|突然|不用|这种话|怪"),
    "失约被催时认栽滑跪": (r"鸽|迟到|说好的|怎么又|等了好久|上次说|没开播|又不|准时",
                   r"迟到|不好意思|对不起|忘了|认|错了|下次|尽量"),
    "能嘴熟络同期是亲近": (r"同期|前辈|十六|莉蔻|克罗雅|联动|好姐妹",
                   r"同期|前辈|好姐妹|熟|关系|圈子|一起"),
    "被套话点名排名他人时端水": (r"最喜欢|最讨厌|排个名|排名|选一个|谁更好|评价一下|爆料|谁最",
                    r"不能说|都很好|不评价|不能说这种话|要闹|不合适|别问|不能给"),
    "被追问私事细节时含糊带过": (r"你为什么|你是不是|你什么时候|几点睡|有没有|你昨晚|你那个|你真的",
                    r"没有|不告诉你|你管|别问|翻篇|不重要|说不清|干嘛|也没有"),
    "对方递话时顺着讲自己的事": (r"那你呢|你呢|你会吗|你也|你当时|你平时|你怎么想",
                    r"灰泽满|我"),
    "被起哄编故事加码时半推半就": (r"怀孕|肚子|在外面|造谣|离谱|你是不|真的假的",
                    r"什么意思|翻篇|编|脑洞|什么时候成|谁说的|别"),
    "对方问日常时答一句自己的事": (r"吃|睡|喜好|喜欢|作息|在干嘛|忙|累|几点|平时|爱好|习惯",
                    r"灰泽满"),
    "对方难过时共情一句就转开": (r"难受|好累|累死|想哭|安慰|撑不住|分手|委屈|被骂|郁闷",
                    r"辛苦了|懂你|没事|别难过|抱|陪你|行|唉"),
                    r"不是不是|不对|那个|就是说|怎么|其实"),
                    r"不说了|算了|翻篇|下一个|不说这个|怪肉麻|说了干嘛"),
    # 2026-10-16 新加：用户要的这一格。只收「阴阳怪气／抬杠／调侃」，
    # **直接骂她本人的不收**（数据里那类几乎全是粉丝在玩"区"这个梗，不是真攻击）。
    "被阴阳或调侃时": (r"不是最|怎么保证|真以为|你也就|好意思|难道|是不是怕|凭什么|就这|"
                  r"也不过|还以为|怎么好意思|管得着|你配|太区了|好丢人|质量高在哪|"
                  r"有区别吗|有什么用|图什么|真有人|居然",
                  r"绿冻|谢谢|对不起|可能真的|是是是|好吧|确实|哪有|你别|"
                  r"那就|也是|不是啊|怎么|反正"),
}

PROMPT = """你在给一个角色扮演的「行为示范库」挑例句。这个角色叫灰泽满，虚拟主播。

【这一格是什么】
  情境（trigger）：{trigger}
  该怎么做（response）：{response}

【候选例句】下面每一对都是她直播时的**真实**弹幕和回复（语音转写，有错字、有口语病）：
{cands}

【挑选标准】—— **只有一条硬标准，但它最重要**：

> **自足。** 把挑出来的这两句单独拿出来、不给任何上下文，读的人要能明白在说什么。
> 具体判据：**回复里提到的具体人物／事件／物品，必须在用户那句话里出现过。**
> ❌ 反例：`用户：满神你好厉害 ／ 她：别抬举灰泽满了，三毛还是挺厉害的`
>    —— 原语境是在聊"三毛"这个人，挑出来之后三毛凭空冒出来，模型会原样照搬，让人莫名其妙。

另外：
- **只从上面候选里挑，内容不许自己编**。但**表面要改彻底**——下面两条是硬要求：
- ⛔ **改一半比不改还糟。** 候选里带直播语境的词（弹幕／切片／下播／舰长／上舰／刷／直播间／免费的好听）
  或语音转写的残片（「好听的钥匙」这种在别处没意义的词），要么**整句改写成私聊里说得通的**，
  要么**就别挑它**。⚠️ 现在的文件里就有改一半留下的：一句「好听的钥匙」原样搬进去了，
  用户看了不知道那是什么。
- ⛔ **回复里凡是用户那句话里没有的具体东西（人名／作者名／作品名／某件东西），
  要么删掉，要么把那个东西补进用户那句话里**——绝不能让它在用户那句话里凭空出现。
  ❌ 现在文件里的反面教材：`用户：满神你好厉害 ／ 她：别抬举灰泽满了，三毛还是挺厉害的`
  （三毛是"满妈喜欢的作者"，跟"被夸"没关系，纯粹是拼接时串进来的）。
  ✅ 改成 `用户：满神你好厉害 ／ 她：别抬举灰泽满了，灰泽满也就那样。`——删掉那个凭空的东西。
- ⛔ **"照念"的不要挑**：她只是把用户那句话重复一遍、或换个问号再抛回去的，那不是回答。
  （直播里她的动作本来就是"念弹幕 + 接话"，所以这种在素材里特别多。已经在筛的时候排掉一批，
  剩下漏网的你自己认——判据是"这句回复里有没有**用户那句话之外的新内容**"。）
- 顺带去掉语音转写的口头填充（"呃""嗯嗯""就是说""那个那个"）和重复。
- 用户那句话要写成**一个人真会对她说的私聊消息**，别用"（某某情境）"这种占位符。
- 挑 **3~5 条**，**话题要互相不同**（别五条都在说同一件事）。
- **候选要都是脏的，就返回空数组**——宁可不给，也不要凑数。这个我宁可自己去补。

只输出 JSON：
{{"samples": [{{"user": "用户说的那句话", "reply": "她的回复", "from": "原候选里的第几条（序号）"}}]}}
"""


def load_pairs():
    out = []
    for line in SFT.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        m = json.loads(line)["messages"]
        dm = [x for x in m if x["role"] == "user"][0]["content"].strip()
        rp = [x for x in m if x["role"] == "assistant"][0]["content"].strip()
        out.append((dm, rp))
    return out


def _bg(s: str) -> set:
    s = re.sub(r"[^一-鿿A-Za-z0-9]", "", s or "")
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else ({s} if s else set())


def is_echo(dm: str, reply: str) -> bool:
    """她的回复是不是**基本在照念**用户那句话（换个问号/加个语气词不算新东西）。

    vup.db 里那张复判表把这一类单独判成 `echo`（理由写「机械念读『X』无新增」），
    但它只覆盖 3 万条；而且 SFT 那 3,605 条虽然全是 premium，抽查仍有漏网的
    （`[就这么富裕啊！] → [就这么富裕啊？]` 被判成 premium/high）。所以自己再拦一道。
    判据：**回复的二字组有 50% 以上能在用户那句话里找到** → 当作照念。

    阈值定得**偏低（宁可错杀）**：候选一格里几十条，误杀一条真例句代价很小；
    而放进去一条照念的，模型会跟着学"把用户的话重复一遍"，代价大。
    实测（拿已知的 7 条验）：三条照念的当场拦住，三条正常回答都只有 0~10% 重合，不受影响。
    """
    gr, gd = _bg(reply), _bg(dm)
    if not gr or not gd:
        return False
    return len(gr & gd) / len(gr) >= 0.50


def candidates(pat, rpat, pairs, limit=45):
    rx, rrx = re.compile(pat), re.compile(rpat)
    seen, out = set(), []
    for d, r in pairs:
        if not (rx.search(d) and rrx.search(r)):
            continue
        if d in seen or len(d) > 46 or len(r) > 110 or len(r) < 4:
            continue
        if is_echo(d, r):                 # 照念的不要
            continue
        seen.add(d)
        out.append((d, r))
        if len(out) >= limit:
            break
    return out


async def curate(client, model, beh, cands):
    listing = "\n".join(f"{i + 1}. 用户：{d}\n   她：{r}" for i, (d, r) in enumerate(cands))
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT.format(
            trigger=beh["trigger"], response=beh["response"], cands=listing)}],
        temperature=0.2, max_tokens=1600, **THINKING_DISABLED)
    txt = (resp.choices[0].message.content or "").strip()
    txt = re.sub(r"```json|```", "", txt).strip()
    if txt[:4].lower() == "json":
        txt = txt[4:].strip()
    try:
        return json.loads(txt).get("samples") or []
    except json.JSONDecodeError:
        return []


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    data = json.loads(BEH.read_text(encoding="utf-8"))
    pairs = load_pairs()
    client, _ = _get_clients()
    model = _get_model_name()

    # 遍历现有 behaviors 文件里的条目；**外加** FILTERS 里那些还没进文件的（新格）
    grid = [(b["name"], b["trigger"], b["response"]) for b in data["behaviors"]]
    known = {n for n, _, _ in grid}
    for n in FILTERS:
        if n not in known:
            grid.append((n, "（**新格，还没进文件**——按下面的例句判它该是什么 trigger）", "（还没定）"))

    result = {}
    for name, trigger, response in grid:
        if args.only and args.only not in name:
            continue
        pat, rpat = FILTERS.get(name, ("", ""))
        cands = candidates(pat, rpat, pairs)
        print(f"\n{'=' * 78}\n【{name}】候选 {len(cands)} 条")
        if not cands:
            print("   ⚠️ 没筛出候选")
            continue
        picked = await curate(client, model, {"trigger": trigger, "response": response}, cands)
        result[name] = picked
        for s in picked:
            print(f"   [{s.get('user')}]")
            print(f"      → {s.get('reply')}")
        if not picked:
            print("   ⚠️ 模型没挑出可用的")

    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n[OK] → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))