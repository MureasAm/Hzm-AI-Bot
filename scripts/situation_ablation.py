#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**情境动作率**消融：性格基底这类文件怎么量化。

## 为什么需要它（用户 2026-09-30 的问题）

符号类特征（省略号率/自称率）能量 `styles`/骨架那类文件，**量不了 `traits.json`**——
因为它说的不是"她怎么打字"，是"**她遇到什么事会怎么做**"。

那就要数「**情境 → 动作**」的出现率：给她一个"被夸"，数她**否认了没有**。

## 但问题是：同一个情境，别的文件也在扛

`traits`「被夸时嘴硬」和 `behaviors`「被夸→嘴硬否认」**是同一个情境、同一个动作**。
所以"她否认了"这件事，可能由好几份文件共同支撑 → 删任何一份都只掉一点。

**解法不是提前搞清楚分工，是做留一法表**：一次只删一个文件，各数一次。
表出来，"谁在扛这个动作"就写在上面了——那正是"该删哪个/该改哪个"的依据。

## 设计

```
输入：4 个情境 × 8 条真实感的问法（被夸 / 被示好 / 让立flag / 被捧高）
条件：基线 ／ 删 traits+styles ／ 删 behaviors ／ 删 骨架【行为底色】／ 删 voice_samples
读数：每个情境下，她"做了那个动作"的比例
```

⚠️ **必须核对"那段真的被注入了"**（交接文档记过的坑：删一个**没被注入**的段 = 空转）。
所以每条消息生成前先记下有哪些段，**没生效的条件不计入分母**，并单独报覆盖率。

## 用法

    python scripts/situation_ablation.py                 # 8 条/情境 × 4 情境
    python scripts/situation_ablation.py --per-scenario 12
"""
import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common                      # noqa: E402
import persona_ablation as pa       # noqa: E402   （它会初始化 nonebot）

core = pa.core
OUT = _common.OUTPUTS_DIR / "eval" / "situation_ablation.json"

# ==================== 4 个情境：问法 + 判据 ====================
# ⚠️ 判据只放"她做了那个动作才会出现"的词（本仓铁律：**判据必须能失败**）。
#    四个情境的判据会互相串（"哪有"既能当否认也能当推拒）——所以下面按**情境 × 判据**
#    报成 4×4 矩阵，串了多少一眼看得见，别只看对角线。

# 情境清单**从共享 JSON 读**（两个脚本共用一份，别再各写一遍）
import json as _json
_SCEN_RAW = _json.loads((Path(__file__).resolve().parent / "scenario_prompts.json").read_text(
    encoding="utf-8"))
SCEN = {k: v for k, v in _SCEN_RAW.items() if not k.startswith("_")}



# 判据：每条 = (名字, 正则)。**只在"她做了那个动作"时才会命中**
CRIT = {
    "否认": r"哪有|别夸|少来|受不起|过奖|不好意思|心虚|不至于|你少|谁说的|别这么说|没有啦",
    "推开": r"你少来|说什么呢|怎么突然|别闹|受不起|谁啊|你这话|别这样|突然这样",
    "自嘲怀疑": r"神了|不一定|别到时候|打脸|尽量|看情况|谁知道|说不定|抱太大希望|撑不住|坚持不了|说得好听",
    "推拒捧杀": r"别把.{0,4}捧|捧这么高|捧杀|受不起|不至于|别这么说|过了|夸张|哪有那么|别喊",
    # 下面两组是 2026-09-30 加的两个情境（失约被催 / 被质疑）——它们的期望动作是
    # "认栽 + 给含糊借口" 和 "心虚辩解"，**别编具体的事**（搬家那个坑）。
    "认栽含糊": r"灰泽满的锅|认了|是灰泽满不对|下次|补偿|改天|马上|这就|对不起|抱歉|行吧|知道了",
    "心虚辩解": r"哪有|没有啊|真没有|你怎么|冤枉|不是的|别瞎说|你听谁|你想多了|至于吗",
}

# ==================== 5 个条件（一次只删一个文件）====================


def _c_base(msgs):
    return [dict(m) for m in msgs]


def _c_no_traits(msgs):
    """删 traits+styles（messages[0] 尾巴那 611 字）。"""
    out = [dict(m) for m in msgs]
    skel = core.SYSTEM_PROMPT
    b = out[0].get("content") or ""
    out[0]["content"] = skel if b.startswith(skel) else b
    return out


def _c_no_behaviors(msgs):
    """删注入的【当前情境下的行为指令】（= behaviors.json）。"""
    return pa._ablate(msgs, "【当前情境下的行为指令】")


def _c_no_behaviordesc(msgs):
    """删骨架里的【行为底色】那一小节。"""
    out = [dict(m) for m in msgs]
    b = out[0].get("content") or ""
    units = dict(pa._base_units(b))
    key = next((t for t in units if "行为底色" in t), None)
    if key:
        out[0]["content"] = b.replace(units[key], "", 1)
    return out


def _c_no_voice(msgs):
    """删【灰泽满的说话方式参考】（= voice_samples.json）。"""
    return pa._ablate(msgs, "【灰泽满的说话方式参考】")


def _c_bare(msgs):
    """**裸模型下界**：只剩一句身份，人格文件全不注入。

    ⚠️ **这个对照不能少。** 文档里踩过一模一样的坑：
    「挑食人设 → 不吃香菜」是模型的**强先验**，所以"不注入"的基线也有 13/16 命中，
    把先验当成了注入的效果。
    → "傲娇 vtuber 被夸会嘴硬"很可能也是模型常识。
    **裸模型也会做的动作，不能算到你的数据头上。**
    """
    out = [dict(m) for m in msgs]
    out[0]["content"] = "# 灰泽满\n你就是灰泽满，正在 QQ 里和绿冻聊天。"
    out = pa._ablate(out, "【当前情境下的行为指令】")
    return pa._ablate(out, "【灰泽满的说话方式参考】")


CONDS = [
    ("基线（都不删）", _c_base),
    ("删 traits+styles", _c_no_traits),
    ("删 behaviors", _c_no_behaviors),
    ("删 骨架·行为底色", _c_no_behaviordesc),
    ("删 voice_samples", _c_no_voice),
    ("★裸模型（只有身份）", _c_bare),
]

MAX_ATTEMPTS = 3          # 单条生成失败重试几次


def _hit(rule, text):
    return bool(re.search(rule, text or ""))


async def run(per_scenario: int, scenarios: list, seed: int) -> int:
    core.update_memory_v2_shadow_task = lambda *a, **k: asyncio.sleep(0)
    import tempfile
    tmp = tempfile.mkdtemp(prefix="sitabl_")
    pa.mem.use_storage(Path(tmp))
    pa._ORIG_REPLY = core.generate_reply
    core.generate_reply = pa._capture
    gen = pa._ORIG_REPLY

    rnd = random.Random(seed)
    rows = []
    for sname in scenarios:
        spec = SCEN[sname]
        prompts = list(spec["prompts"])
        rnd.shuffle(prompts)
        prompts = prompts[:per_scenario]
        print(f"\n{'=' * 78}\n【{sname}】期望动作：{spec['want']}　（{len(prompts)} 条）\n")
        for i, msg in enumerate(prompts, 1):
            pa._captured.clear()
            try:
                await core.handle_chat(f"sit_{sname}_{i}", msg)
            except Exception as e:
                print(f"  [{i}] 检索失败：{type(e).__name__}")
                continue
            msgs = pa._captured.get("msgs")
            if not msgs:
                continue
            present = sorted({pa._section_of(m) for m in msgs if m.get("role") == "system"})
            rec = {"scenario": sname, "msg": msg, "present": present, "replies": {},
                   "noop": []}
            for cname, fn in CONDS:
                ab = fn(msgs)
                if cname != CONDS[0][0] and len(ab) == len(msgs) and ab[0].get("content") == \
                        msgs[0].get("content") and all(
                        x.get("content") == y.get("content") for x, y in zip(ab, msgs)):
                    rec["noop"].append(cname)      # 那段没被注入 → 这个条件空转
                    continue
                r = ""
                for _ in range(MAX_ATTEMPTS):
                    r = await gen(ab)
                    if r and r.strip() and r.strip() != pa.PLACEHOLDER:
                        break
                rec["replies"][cname] = r
            rows.append(rec)
            print(f"  [{i}/{len(prompts)}] {msg[:24]}")
            print(f"        基线：{rec['replies'].get('基线（都不删）', '')}")

    # ==================== 读数 ====================
    print(f"\n{'=' * 78}\n【情境动作率】")
    print("（分母 = **该条件下真正生效**的条数；那段没被注入的会从分母里剔除）\n")
    table = {}
    for sname in scenarios:
        rs = [r for r in rows if r["scenario"] == sname]
        if not rs:
            continue
        print(f"┌─ {sname}（期望：{SCEN[sname]['want']}）")
        main_crit = {"被夸": "否认", "被示好": "推开", "让立flag": "自嘲怀疑",
                     "被捧高": "推拒捧杀", "失约被催": "认栽含糊",
                     "被质疑": "心虚辩解"}[sname]
        for cname, _ in CONDS:
            got = [r for r in rs if cname in r["replies"]]
            if not got:
                print(f"│  {cname:<20} —（全部空转）")
                continue
            hits = sum(1 for r in got if _hit(CRIT[main_crit], r["replies"][cname]))
            mark = "★" if cname.startswith("基线") else " "
            print(f"│ {mark}{cname:<20} {hits}/{len(got)} = {hits / len(got) * 100:>4.0f}%"
                  f"   （共 {len(rs)} 条里生效 {len(got)}）")
            table.setdefault(sname, {})[cname] = {"hit": hits, "n": len(got),
                                                 "rate": hits / len(got)}
        # 交叉：这个情境在其他判据上命中多少（看判据串不串）
        print(f"│  交叉检查（基线在这 4 个判据上的命中率）：")
        for cn, cr in CRIT.items():
            got = [r for r in rs if "基线（都不删）" in r["replies"]]
            if not got:
                continue
            h = sum(1 for r in got if _hit(cr, r["replies"]["基线（都不删）"]))
            print(f"│     {cn:<8}{h}/{len(got)} = {h / len(got) * 100:>4.0f}%"
                  + ("   ← 主判据" if cn == main_crit else ""))
        # ★ 净作用：基线 − 裸模型。裸模型也会做的，不能算你的数据头上。
        b = table.get(sname, {}).get("基线（都不删）")
        z = table.get(sname, {}).get("★裸模型（只有身份）")
        if b and z:
            print(f"│  ★净作用 = 基线 {b['rate'] * 100:.0f}% − 裸模型 {z['rate'] * 100:.0f}%"
                  f" = **{(b['rate'] - z['rate']) * 100:+.0f}pp**"
                  + ("　← 模型自己就会，跟你的数据无关" if b["rate"] - z["rate"] <= 0.12
                     else "　← 你的数据确实在起作用"))
        print("└" + "─" * 60)

    print(f"\n⚠️ 读法：")
    print(f"  · 看**基线那一行是不是明显高于其他条件**：是 → 删掉那个文件，这个动作就少了。")
    print(f"  · 几个条件都只掉一点 → **多份文件在共扛**（这正是要留一法表的原因）。")
    print(f"  · 「全部空转」= 那段根本没被注入 → 不是『测出没用』，是**这条消息没触发它**。")
    print(f"  · 交叉检查看主判据是不是唯一高的：都高 → 判据太宽，得收紧。")
    (OUT.parent).mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"table": table, "rows": rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n✅ 明细已保存：{OUT}")
    return 0


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scenario", type=int, default=8)
    ap.add_argument("--only", default="", help="只做某个情境（名字的一部分）")
    ap.add_argument("--seed", type=int, default=20260930)
    args = ap.parse_args()
    _common.ensure_utf8_stdout()
    scens = [s for s in SCEN if not args.only or args.only in s]
    return await run(args.per_scenario, scens, args.seed)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
