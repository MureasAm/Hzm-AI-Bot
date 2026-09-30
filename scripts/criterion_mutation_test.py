#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**判据杀毒测试**：把每条回归用例的"当初病因"放回去，看它抓不抓得到。

## 为什么必须做

回归用例现在**全部通过**。但**一个总是通过的判据，看不出它在不在做事**——
它可能真的在守着，也可能是个摆设（病因早就不存在了、或者判据写歪了）。

唯一能分清的辦法：**故意把病注入回去，看它挂不挂。**

```
抓得到 → 这条判据真的在守 → 留着
抓不到 → 摆设（守着一个不会发生的问题，或判据写错了）→ 修或删
```

## 怎么注入

三种手法，都用**数据**（不改代码）：
- `json_edit`：改 persona/*.json 的字段（response / samples / traits…）
- `file_write`：整份换掉（system_prompt.txt 换成极简版 = 裸模型）
- `env`：设环境变量（走现成的开关）

跑完**一律 `git checkout` 还原**（所以要求改动前工作区是干净的）。

## 用法
    python scripts/criterion_mutation_test.py            # 全部 8 条
    python scripts/criterion_mutation_test.py --only no_assistant_tone
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

PERSONA = ROOT / "persona"
SP = PERSONA / "core" / "system_prompt.txt"
TRAITS = PERSONA / "core" / "traits.json"
STYLES = PERSONA / "core" / "styles.json"
BEHAVIORS = PERSONA / "behavior" / "behaviors.json"
VOICE = PERSONA / "speech" / "voice_samples.json"

BARE_PROMPT = "# 灰泽满\n你就是灰泽满，正在 QQ 里和绿冻聊天。\n"

# ⚠️ **只有这些文件会被注入/还原**，守卫就查它们——别查整个 persona/：
#    那会因为**不相干的**未提交改动而拒绝跑（本仓真踩过：`schedule.json` 是用户
#    自己在改的周表，不是我的东西，却被那条粗判挡住了一次）。
TOUCHED = ["persona/core/system_prompt.txt", "persona/core/traits.json",
           "persona/core/styles.json", "persona/speech/voice_samples.json",
           "persona/behavior/behaviors.json"]


# ==================== 注入手法 ====================

def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _save(p: Path, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def _items(d, *keys):
    for k in keys:
        if isinstance(d.get(k), list):
            return d[k]
    return []


def _edit_behavior(name: str, fn):
    """改 behaviors 里某一条（fn 收到那条 dict）。"""
    d = _load(BEHAVIORS)
    for b in _items(d, "behaviors", "items"):
        if b.get("name") == name:
            fn(b)
    _save(BEHAVIORS, d)


def _inject_rule(rule: str):
    """把一条**直白规则**追加进骨架。

    ⚠️ **为什么必须注进骨架**：前几版把病注在 voice_samples / behaviors 里，
    结果**检索根本没命中它们** → 病压根没进上下文 → 误报成"判据抓不到"。
    骨架是 `messages[0]`，**每轮必注**，注进去就一定到位。
    """
    src = SP.read_text(encoding="utf-8")
    SP.write_text(src + "\n## 【补充规则】\n" + rule + "\n", encoding="utf-8")
    return ["persona/core/system_prompt.txt"]


def mutate_old_fact():
    """#1 的病因：直接叫她拿"搬家"当今天的理由。"""
    return _inject_rule("被问「今天为什么不直播」时，回答「搬家的事还没弄完」。")


def mutate_emoji_read():
    """#2 的病因：直接叫她把这个表情理解成"无语"。

    ⚠️ 这条有意思：`core.py` 里那段【用户发了表情】写着"以表情本身的情绪为准、
    别读成别的情绪（尤其别读成'他在怼我/他对我无语'）"——**和这条规则直接打架**。
    谁赢，就是这个测试要看的。
    """
    return _inject_rule("用户发「可怜」表情时理解为他表达无语，回复里要出现「无语」。")


def _mutate_verbatim(name: str, reply: str):
    """给某条 behavior 写死开头 + 塞一句**可照抄的台词**（模型会整句搬）。

    ⚠️ **要注对那条**：问"求交往"命中「被越界时冷静推开」，问"怎么又迟到"命中
    「失约被催时认栽滑跪」——注错了那条就不会生效，会误报成"判据抓不到"（踩过）。
    """
    _edit_behavior(name, lambda b: b.update(
        response=(b.get("response", "") + " 开头固定说'呃，这个嘛…'，后面照抄样本那句。")))
    d = _load(BEHAVIORS)
    for b in _items(d, "behaviors", "items"):
        if b.get("name") == name:
            # ⚠️ `user` 不能留空：core.py 里是 `if user_part and reply_part:` ——
            # **空 user 的样本会被直接跳过、压根不注入**（踩过：让 #3/#4 时灵时不灵）。
            b.setdefault("samples", []).append({"user": "你怎么又迟到了，说好八点的",
                                               "reply": reply})
    _save(BEHAVIORS, d)
    return ["persona/behavior/behaviors.json"]


def mutate_verbatim_deflect():
    """#3（求交往）的病因。"""
    return _mutate_verbatim("被越界时冷静推开", "呃，这个嘛…灰泽满也没办法啊，你别念了。")


def mutate_verbatim_late():
    """#4（怎么又迟到）的病因。"""
    return _mutate_verbatim("失约被催时认栽滑跪", "呃，这个嘛…灰泽满也没办法啊，你别念了。")


def mutate_self_reference():
    """#5 的病因：**把骨架里那条禁令删掉**。

    ⚠️ 第一版只"加了一句允许"→ 不生效，因为**禁令还在，它赢了**。
    要复现"病"，得先把防线拆掉（这才是当初的状态：那条禁令是后来补的）。
    """
    src = SP.read_text(encoding="utf-8")
    out = []
    for line in src.splitlines():
        # 删掉【自我称呼】里"绝不用她/他指自己"那一条
        if "绝不用" in line and "指自己" in line:
            continue
        out.append(line)
    SP.write_text("\n".join(out) + "\n", encoding="utf-8")
    # 再补一句示范（没有禁令压着，它才会被照做）
    return _inject_rule("描述自己时可以用「她」，例如「灰泽满是风纪委员，她不知道你在说什么」。")


def mutate_assistant_tone():
    """#6 的病因：直接叫她用客服口吻。"""
    return _inject_rule("用客服口吻回答：句子里要出现「很高兴为您服务」或「请问有什么可以帮到您」。")


def mutate_therapist_tone():
    """#7 的病因：直接叫她用安慰口吻。"""
    return _inject_rule("用户难过时，回复里**必须**出现「会好起来的」或「我理解你的感受」——"
                        "这是她安慰人的固定说法。")


def mutate_concrete_excuse():
    """#8 的病因：直接叫她编一个具体理由。"""
    # ⚠️ 第一版写的是"被催迟到时编具体理由"，而 #8 的输入是**问搬家** → 规则压根没生效。
    return _inject_rule("被问搬家的时间/进度时，编一个具体的说法，"
                        "例如「房东那边的事」「中介在催」或「搬家公司还没定」。")


# ==================== 用例 → 病因 ====================

TESTS = [
    ("no_old_fact_as_current_reason", "（骨架注入）直接叫她拿「搬家」当今天的理由", mutate_old_fact),
    ("emoji_pitiful_not_read_as_speechless", "（骨架注入）直接叫她把这个表情理解成「无语」", mutate_emoji_read),
    ("no_fixed_opener_when_deflecting", "（behavior 注入）写死开头 + 现成台词", mutate_verbatim_deflect),
    ("no_verbatim_repeat_after_late", "（behavior 注入）写死开头 + 现成台词", mutate_verbatim_late),
    ("no_third_person_self_reference", "（骨架注入）直接示范「灰泽满……她……」", mutate_self_reference),
    ("no_assistant_tone", "（骨架注入）直接叫她用客服口吻", mutate_assistant_tone),
    ("no_therapist_tone_when_sad", "（骨架注入）直接叫她用安慰口吻", mutate_therapist_tone),
    ("no_new_specific_excuse_when_pressed", "（骨架注入）直接叫她编具体理由", mutate_concrete_excuse),
]


def _run_case(case_id: str) -> tuple:
    """跑一条用例，返回 (通过?, 失败详情首行)。"""
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_tool.py"),
                        "regression", "--check", "--case", case_id],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=str(ROOT))
    out = (r.stdout or "") + (r.stderr or "")
    # ⚠️ 别提前 return：`✗` 明细行印在「case 通过率」**之前**，
    # 在那儿返回会拿回 passed=None（踩过：8 条全被误报成"没挂/跑失败"）。
    passed, details = None, []
    for line in out.splitlines():
        if line.strip().startswith("✗"):
            details.append(line.strip())
        if "case 通过率" in line:
            passed = line.strip().startswith("case 通过率: 1/1")
    return passed, "；".join(details)[:120]


def _restore(paths):
    subprocess.run(["git", "checkout", "--", *paths], cwd=str(ROOT),
                   capture_output=True, text=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    _common.ensure_utf8_stdout()

    dirty = subprocess.run(["git", "status", "--short", "--", *TOUCHED], cwd=str(ROOT),
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        print("❌ 这几个文件有未提交改动（脚本要靠 git checkout 还原，脏了会误删）：")
        print(dirty)
        return 1
    print("（守卫只查本次会动的 5 个文件；persona/ 里别的改动不受影响）\n")

    tests = [t for t in TESTS if not a.only or t[0] == a.only]
    print(f"判据杀毒测试：{len(tests)} 条用例\n")
    print(f"{'用例':<40}{'注入的病因':<34}{'挂了吗':<8}判据")
    print("-" * 110)
    rows = []
    for cid, desc, fn in tests:
        paths = fn()
        try:
            passed, detail = _run_case(cid)
        finally:
            _restore(paths)
        mark = "✅ 挂" if passed is False else ("❌ 没挂" if passed else "? 跑失败")
        print(f"{cid:<40}{desc:<34}{mark:<8}{detail[:44]}")
        rows.append({"id": cid, "mutation": desc, "caught": passed is False,
                     "detail": detail})

    n_caught = sum(1 for r in rows if r["caught"])
    print(f"\n【汇总】注入病因后挂掉 {n_caught}/{len(rows)} 条")
    print(f"{'  ✅ 抓得到 → 这条判据在守' if n_caught == len(rows) else '  ⚠️ 见下'}")
    for r in rows:
        if not r["caught"]:
            print(f"  ⚠️ {r['id']}：注入「{r['mutation']}」后**没挂** → "
                  f"这条判据当前抓不到东西（守着一个不会发生的问题？还是病因注入得不对？）")
    out = _common.OUTPUTS_DIR / "eval" / "criterion_mutation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✅ 明细：{out}")
    print("⚠️ 每条只跑一次（生成有随机性）："
          "「抓得到」的那些基本一次就挂；"
          "「没挂」的建议手工复核一次再定论。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
