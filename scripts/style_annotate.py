#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""风格层标注集：**把"像不像她"固化成数据**，让风格类消融实验有尺子。

## 为什么要有它（一句话）

检索层有 `retrieval_eval_cases.json`、生成层有 `regression_cases.json`，
**风格层什么都没有** —— 所以这一轮所有风格类结论都不可靠：
三把尺子（长度/括号指标 → 指纹距离 → LLM 成对比较）全废了，
废因见 `docs/文件体检结论.md`。

**"像不像她"的标准只能来自懂她的人**（用户），而这个标准必须**固化成数据**，
不能每次临场判。这份就是这个数据。

## 三种题（各有各的用处，**v3 起整组题是主力**）

| 形态 | 长什么样 | 回答什么 | 为什么这么设计 |
|---|---|---|---|
| **`set`** ★ | 两组各 8 条（有那文件 / 没有），组内打乱、**组间不标** | **哪一组更像她** | **人判风格本来就是看一批，不是看一对**（见下） |
| `pair` | 同一句话的两版（有/删），左右打乱 | ①差在内容还是说话方式 ②哪版更像她 | 能给出 per-item 的 n 和双侧 p |
| `single` | 一句回复（她本人真实 / 生成，盲） | **这句像她吗** | 建"正例锚"；也能量"生成离她本人有多远" |

### ⚠️ 为什么 v3 换成"整组"（2026-09-30 实测，别改回去）

成对题那一版，用户标了 30 条：**23 道成对题里 20 道答「差不多」（87%）**。
原因不是文件没用，是**问法测错了轴**——两版是同一个模型 + 同一套 base 骨架生成的，
风格天然一样，差的只是**信息量**；而"这两句哪句更像她"在人看来经常是"都对"。

所以 v3 两处改动：
1. **整组题**（8 条 vs 8 条，只答一次）：把判断粒度抬到"一批的总体风格"。
   ⚠️ 代价要认：每组只有 1 个判断 → 统计功效低，靠**多组重复**换可信度（默认每个段 3 组）。
2. **成对题改问两问**：①差在**内容**还是**说话方式** ②风格上哪版更像她。
   第①问本身就是读数：答"只有内容不同"多 → 那段管的是内容，该改用**任务正确性**判据。

### ⚠️ 先验尺子，再下结论（`--validate`）

**不许跳过这一步**。两道阳性对照：

- **硬对照**：她本人 vs 客服腔 → 分不出来就说明**这把尺子废**，别拿它测任何文件
- **软对照**：她本人 vs 生成 → 量"生成离她本人有多远"，同时给出**灵敏度地板**：
  比这个差还小的文件效应，这把尺子根本测不出来

读数的**第一节永远是"仪器分辨力"**（多少组答"分不出"），不是那些占比——
87% 分不出的时候，后面的占比分母只剩个位数，报出来就是骗人。

⚠️ **每条都带「用户说的那句话」**：本仓踩过的坑——脱离上下文的对比，
比的是"两个都不像她的东西"（见 `docs/交接文档.md` 负一节坑 5）。

## 关键纪律（都写进代码了）

- **左右打乱 + 内容哈希定序**：`cond`（真实条件）只写在用例库里，`风格标注.md` 里看不到
- **单条必须蒙住来源**：`real`（她本人）还是 `gen`（生成）**只在 JSON 里**，md 里绝不显示
  ——否则"像不像她"变成"猜这题出得对不对"
- **混着出题**：一批里混多个文件/小节 + 单条，避免"这批全在测同一个东西"把你判得越来越宽
- **只增不改**：已有 `label` 不会被覆盖（要重标用 `--relabel`）
- **幂等去重**：`src` 是内容哈希，重跑同一批不会重复出题
- **没标的不丢**：`--batch` 每次都把"还没标的"重新摆出来，只标一半也不会丢掉其余
- **报 p 值**：只报选中率不报显著性 = 骗人
  （本仓踩过：只算下尾再乘 2 → 100% 却 p=1.0，靠"结论自相矛盾"才发现）

## 用法

    python scripts/run_tool.py style-annotate                  # 出一批题（默认 20）→ 写 风格标注.md
    python scripts/run_tool.py style-annotate --ingest         # 你标完后：回读 → 落进用例库
    python scripts/run_tool.py style-annotate --report         # 出消融读数（按文件/小节 + p 值）
    python scripts/run_tool.py style-annotate --status         # 进度
    python scripts/run_tool.py style-annotate --run "【说话方式参考】"
        # ↑ 给还没测过的段**现生成**一批对照（要跑 LLM，慢、花钱）

## 路线图（别跳步）

    ① 你标一批          ← 现在这里。**你的判断本身就是读数**，不需要任何模型当中间人
    ② 验裁判：拿你标好的当标准答案，看某个自动判能不能复现你的判断
    ③ 只有 ② 过关了，才谈"自动判新对照" —— 否则又是"拿不懂她的模型当裁判"那个坑

⚠️ **别用向量相似度当"更接近"**：风格不是语义，"意思一样但不像她"的句子余弦可能更高。
"""
import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

SCRIPTS = _common.SCRIPTS_DIR
ROOT = _common.PROJECT_ROOT
CASES_FILE = SCRIPTS / "style_eval_cases.json"
MD_FILE = ROOT / "风格标注.md"          # 你手上唯一要碰的文件（不入库，和 问题记录.md 一样）
EVALDIR = ROOT / "outputs" / "eval"
CHAT_LOG = ROOT / "data" / "chat_log" / "chat.jsonl"

SEED = 20260930
# 「有它」的那些条件名（其余 = 「删掉它」那侧）
WITH_CONDS = {"with", "base"}

# 默认**先做哪些段**：按"预算占比大 × 从没被验过"排（依据是本仓自己的体检结论）。
# 想让某个段插队就把它写前面；想只做某一个段用 `--only 段名的一部分` 覆盖。
#   · 【性格基底】+【语言风格】 611 字 ≈ base 的 17%，只有"用户盲判 76%"这一条弱证据
#   · 骨架各小节 2974 字 ≈ base 的 83%，是**唯一从没测过的大块**
#   · 【世界】terms 曝光面最大（7% 的轮次用得到）／【行为指令】4%
PRIORITY = ("性格基底", "核心人格", "身份", "说话节奏", "自我称呼",
            "括号与省略号", "行为底色", "世界", "行为指令")


def _prio(section: str) -> int:
    for i, key in enumerate(PRIORITY):
        if key in section:
            return i
    return len(PRIORITY)


# 她自己的称呼 —— 用来控"名字带偏"那个混淆变量（见 _name_sym）
SELF = re.compile(r"(灰泽满|hzm|小满|满姐)")


# ⚠️ v1 只问一个问题（"哪句更像她"），实测**测错了轴**：
#   「大多数语句在风格上是一致的，只不过内容上更具体」（用户原话，2026-09-30）
#   实测印证：43% 的对**四个形式特征完全一样**（问号/省略号/括号/叹号都没差），
#   另有 22% 长度悬殊（长的就是"更具体"那版）→ 那一问量到的是**内容丰不丰富**，不是风格。
# 所以 v2 拆成两问：**先分轴，再比方向**。第一问本身就是读数（那段动不动风格）。
DIFFS = [("只有内容不同", "content"), ("只有说话方式不同", "style"),
         ("两样都不同", "both"), ("基本没区别", "none")]
STYLE_DIFFS = {"style", "both"}          # 这两类才说明"那段动了风格"
ANSWERS = {
    # ⚠️ `差不多` 是 v1 的选项名，v2 改成 `一样`。**两个都认**——
    # 老文件里已经打好的勾不能因为改版就作废（2026-09-30 真丢过一次标注，别再犯）。
    "pair": [("甲", "left"), ("乙", "right"), ("一样", "tie"), ("差不多", "tie")],
    "set": [("甲组", "left"), ("乙组", "right"), ("分不出", "tie")],
    "single": [("不像她", "unlike"), ("像她", "like"), ("说不准", "unsure")],
}
PICK_RE = {k: [(re.compile(r"\[[xX✓]\]\s*" + re.escape(t)), lab) for t, lab in v]
            for k, v in ANSWERS.items()}
DIFF_RE = [(re.compile(r"\[[xX✓]\]\s*" + re.escape(t)), lab) for t, lab in DIFFS]


def _h(s: str) -> str:
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()


def _clean(x) -> str:
    return (x or "").strip()


def _norm_section(s: str) -> str:
    """把段名收拾成能看的：去掉 `## `，补回被截断的 `】`。

    ⚠️ 骨架小节的标题在 `persona_ablation.py::_base_units` 里被截到 16 字
    （`skel[st: st + 16]`），所以库里存的是「## 【括号与省略号：例外不是常」这种半截名。
    库里当 key 用没问题，**摆给人看必须收拾干净**。
    """
    s = re.sub(r"^#+\s*", "", (s or "").strip())
    if s.startswith("【") and not s.endswith("】"):
        s += "】"
    return s


def _overlap(a: str, b: str) -> float:
    """a 的字符 bigram 有多少出现在 b 里（0~1）。只用来判"这两版有没有差别"。"""
    def bg(x):
        x = re.sub(r"[\s，。！？、；：（）【】()\[\]…~—\-]+", "", x or "")
        return {x[i:i + 2] for i in range(len(x) - 1)} or ({x} if x else set())
    ba = bg(a)
    return 0.0 if not ba else len(ba & bg(b)) / len(ba)


# 摆给人看的东西，这几条过不了就扔（不是判"像不像她"，是判"有没有东西可判"）
MIN_CHARS = 4
NEAR_SAME = 0.75          # v1 是 0.92，太松：0.7~0.9 那批就是"看着几乎一样"的废题


def _ok_pair(t1: str, t2: str) -> bool:
    """两版**几乎一模一样**的题扔了：那种你只会答"没区别"，纯占你注意力。

    ⚠️ 取**双向**的最小值：只要有一个方向高度重合，就是"短的那个被长的包住了"，
    同样没有东西可判（实测 v1 的 0.92 放过了 13% 这种题）。
    """
    if len(t1) < MIN_CHARS or len(t2) < MIN_CHARS:
        return False
    return min(_overlap(t1, t2), _overlap(t2, t1)) <= NEAR_SAME


def _ok_single(msg: str, text: str) -> bool:
    return len(text) >= MIN_CHARS and text != msg


def _name_sym(t1: str, t2: str) -> bool:
    """两版的**自称状态是否对称**（都有 / 都没有 灰泽满那一串自称）。

    ⚠️ 为什么必须控这个（用户 2026-09-30 原话）：
    > 「出现 hzm 类似名字的会让我更加有倾向」
    实测：**43% 的对里只有一版自称**。这种题**测不了风格**——你是在对一个
    名字标志物做反应，不是在两版说话方式之间比。所以默认只出对称的题，
    不对称的单独留着做"名字算不算风格"的专门实验（`--name-asym`）。
    """
    return bool(SELF.search(t1)) == bool(SELF.search(t2))


# ==================== 用例库读写 ====================

def _load() -> dict:
    if CASES_FILE.exists():
        try:
            return json.loads(CASES_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"❌ 用例库读坏了（{type(e).__name__}: {e}）——先修 {CASES_FILE} 再跑")
            sys.exit(1)
    return {"cases": []}


def _save(doc: dict):
    CASES_FILE.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")


def _cases(doc: dict) -> list:
    return doc.setdefault("cases", [])


def _next_seq(cases: list) -> int:
    nums = [int(m.group(1)) for c in cases
            if (m := re.fullmatch(r"s_(\d+)", c.get("id") or ""))]
    return max(nums, default=0) + 1


# ==================== 造题（候选池） ====================

def _mk_pair(section, compare, msg, t1, c1, t2, c2) -> dict:
    """建一道成对题。**左右由内容哈希定**——同一内容永远同样的左右，重跑不会翻。"""
    section = _norm_section(section)
    if not _ok_pair(t1, t2):
        return None
    src = f"pair:{section}:{compare}:{_h(msg + '|' + t1 + '|' + t2)}"
    rnd = random.Random(int(_h(src)[:12], 16))
    (l, lc), (r, rc) = ((t1, c1), (t2, c2)) if rnd.random() < 0.5 else ((t2, c2), (t1, c1))
    return {"kind": "pair", "section": section, "compare": compare, "msg": msg,
            "left": {"text": l, "cond": lc}, "right": {"text": r, "cond": rc},
            "name_sym": _name_sym(t1, t2),
            "diff": None,                      # ← v2 第一问（内容/风格/都/无）
            "label": None, "why": "", "src": src}


def _mk_single(section, msg, text, cond) -> dict:
    if not _ok_single(msg, text):
        return None
    return {"kind": "single", "section": section, "msg": msg, "text": text,
            "cond": cond, "label": None, "why": "",
            "src": f"single:{cond}:{_h(msg + '|' + text)}"}


def _add(out: list, item):
    if item is not None:
        out.append(item)


def _pool_pairs() -> list:
    """从**已有的消融产物**收对照（零成本，不重新生成）。"""
    out = []

    p = EVALDIR / "persona_ablation.json"
    if p.exists():
        for r in json.loads(p.read_text(encoding="utf-8")).get("rows", []):
            w, wo, m = _clean(r.get("with")), _clean(r.get("without")), _clean(r.get("msg"))
            if w and wo and m:
                _add(out, _mk_pair("【性格基底】+【语言风格】", "with_vs_without", m,
                                   w, "with", wo, "without"))

    # 整块压缩对照（`persona_ablation.py --compress`）：现状 vs 更短的等价版本
    p = EVALDIR / "compress_ablation.json"
    if p.exists():
        doc = json.loads(p.read_text(encoding="utf-8"))
        meta = doc.get("meta") or {}
        sec = (f"★整块·现状({meta.get('full_chars', '?')}字)"
               f" vs 压缩版({meta.get('short_chars', '?')}字)")
        for r in doc.get("rows", []):
            full, sh, m = _clean(r.get("with")), _clean(r.get("without")), _clean(r.get("msg"))
            if full and sh and m:
                _add(out, _mk_pair(sec, "with_vs_without", m, full, "with", sh, "without"))

    for fname, sec_key in (("skeleton_ablation.json", "ablated"),
                           ("section_ablation.json", "ablated")):
        p = EVALDIR / fname
        if not p.exists():
            continue
        for r in json.loads(p.read_text(encoding="utf-8")).get("rows", []):
            base, m = _clean(r.get("reply")), _clean(r.get("msg"))
            if not (base and m):
                continue
            for sec, abl in (r.get("ablation") or {}).items():
                abl = _clean(abl)
                if not abl or abl == base:
                    continue
                _add(out, _mk_pair(sec, "with_vs_without", m, base, "base", abl, "ablated"))
                plc = _clean((r.get("placebo") or {}).get(sec))
                if plc:
                    # 第二问：「是那段的**内容**在起作用，还是只是上下文变短了？」
                    _add(out, _mk_pair(sec, "without_vs_placebo", m, abl, "ablated", plc, "placebo"))
    return out


def _real_replies() -> dict:
    """`用户那句话 → 她本人的回复`（私聊；用于造单条题的 `real` 那版）。"""
    out = {}
    if not CHAT_LOG.exists():
        return out
    for line in CHAT_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("kind") != "private" or not str(r.get("session", "")).isdigit():
            continue
        u, rep = _clean(r.get("user")), _clean(r.get("reply"))
        if u and rep and len(u) <= 40 and "\n" not in u and u not in out:
            out[u] = rep
    return out


def _pool_her_voice(limit: int = 400) -> list:
    """她本人的真实回复，**按会话轮流取**（整组题的对照用）。

    ⚠️ 两个坑，都会让对照组失去意义：
    ① **不许按 chat_log 顺序切**：同一个会话的回复连着排，直接切片会得到
       "甲组全是问语音的、乙组全是道晚安的" —— 两组差的是**话题**不是风格，控制就废了。
       → 按会话**轮转**取，保证两组话题混着。
    ② **组内不许有近似重复**："发出来是破音版晚安" 和 "行行行，破音版晚安" 摆一组里，
       你判的就变成"哪组更啰嗦"，不是"哪组更像她"了。
    """
    by_sess = defaultdict(list)
    if CHAT_LOG.exists():
        for line in CHAT_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kind") != "private" or not str(r.get("session", "")).isdigit():
                continue
            rep = _clean(r.get("reply"))
            if len(rep) >= MIN_CHARS:
                by_sess[str(r["session"])].append(rep)

    out, seen = [], []
    i = 0
    while len(out) < limit and by_sess:
        added = False
        for sess in sorted(by_sess, key=_h):                 # 定序，可复现
            lst = by_sess[sess]
            if i >= len(lst):
                continue
            t = lst[i]
            if any(min(_overlap(t, s), _overlap(s, t)) > NEAR_SAME for s in seen):
                continue                                     # 太像已有的，跳过
            out.append(t)
            seen.append(t)
            added = True
            if len(out) >= limit:
                break
        if not added:
            break
        i += 1
    return out


_CTX = None


def _context_index() -> dict:
    """`{用户那句话: [前面两轮的 (用户说, 她答)]}` —— 判题时要看得懂她为什么这么答。

    ⚠️ **为什么必须给前文**（用户 2026-09-30：「很多语句的信息也需要上下文」）：
    `persona_ablation` 那批是**单轮无历史**生成的（`handle_chat(session, msg)`，短期记忆是空的）
    ——正是 `交接文档.md` 负一节坑 5 记的那件事。判题的人看不到前文，
    就只能在"两个都接不上话的东西"里挑。
    （**彻底的修法**是生成时也带历史，那是下一轮；这轮先让**判的人**看得懂。）
    """
    global _CTX
    if _CTX is not None:
        return _CTX
    by_sess = defaultdict(list)
    if CHAT_LOG.exists():
        for line in CHAT_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kind") != "private" or not str(r.get("session", "")).isdigit():
                continue
            u, rep = _clean(r.get("user")), _clean(r.get("reply"))
            if u and len(u) <= 40 and "\n" not in u:
                by_sess[str(r["session"])].append((u, rep))
    idx = {}
    for turns in by_sess.values():
        for i, (u, _rep) in enumerate(turns):
            if u not in idx:
                idx[u] = turns[max(0, i - 2):i]
    _CTX = idx
    return idx


# ==================== 整组题（v3：人判风格是看一批，不是看一对）====================
#
# ⚠️ **为什么换这个**（用户 2026-09-30 实测）：成对题那一版，23 道里 20 道答"差不多"（87%）。
# 原因不是文件没用，是**问法不对**——两版是同一个模型 + 同一套 base 骨架生成的，
# 风格天然一样、只有信息量不同；而"这两句哪句更像她"在人看来经常是"都对"。
# 人判风格的方式本来是**看一批**：给她 8 条有那个文件的、8 条没那个文件的，问"哪一组更像她"。
#
# 代价说清楚：**每组只有 1 个判断**（不是 8 个）→ 统计功效低，要靠**多组重复**换可信度。
# 所以默认每个段出 3 组；读数报"3 组里你选对了几个" + 双侧 p，n 小就明说别当数。

def _set_cov(texts: list) -> dict:
    """一组的**混淆变量**：自称、平均长度、问号率。读数要跟它们一起看。"""
    xs = [t for t in texts if t]
    n = max(len(xs), 1)
    return {
        "n": len(xs),
        "self": sum(1 for t in xs if SELF.search(t)),
        "avg_len": round(sum(len(t) for t in xs) / n, 1),
        "q": sum(1 for t in xs if re.search(r"[?？]", t)),
    }


def _mk_set(section, left: list, right: list, key_side: str, tag: str = "") -> dict:
    """建一道整组题。`left`/`right` 是 `[(文本, 条件), …]`。

    甲/乙与"哪组是命中组"**解耦**：读数按 `key_side` 算，摆给你看的是甲/乙，
    而 key_side 随机落在甲或乙——否则你学会"甲就是带文件的"就完了。

    `key_side`：哪一侧算"命中"（验尺子时 = 她本人的真实回复那侧；
                测文件时 = **带那个文件**的那侧）。
    """
    if min(len(left), len(right)) < 3:
        return None
    # ⚠️ **两组条数必须一样**：一组 6 条一组 8 条，你能数出来 → 那是白送线索
    n = min(len(left), len(right))
    left, right = left[:n], right[:n]
    src = f"set:{tag}:{section}:{_h('|'.join(t for t, _ in left) + '#' + '|'.join(t for t, _ in right))}"
    rnd = random.Random(int(_h(src)[:12], 16))
    # 组内顺序打乱（免得"前两条特别像她"这种顺序效应）
    l, r = list(left), list(right)
    rnd.shuffle(l)
    rnd.shuffle(r)
    if rnd.random() < 0.5:
        a, b, key = l, r, key_side
    else:
        a, b, key = r, l, ({"left": "right", "right": "left"}.get(key_side))
    return {
        "kind": "set", "section": section, "tag": tag,
        "left": {"texts": [t for t, _ in a], "conds": [c for _, c in a]},
        "right": {"texts": [t for t, _ in b], "conds": [c for _, c in b]},
        "key_side": key, "cov": {"left": _set_cov([t for t, _ in a]),
                                 "right": _set_cov([t for t, _ in b])},
        "label": None, "why": "", "src": src,
    }


def _balance_sides(items: list) -> list:
    """同一批里"命中组"不许全落在同一侧。

    ⚠️ 真踩到过：两道验尺子题**恰好都**把"她本人"放在乙组 → 你判完第一题
    就知道第二题选乙，第二题白测。`_mk_set` 是逐题独立随机的，管不了批内平衡，
    所以在这里补一道：同侧超过半数就把后面几条翻过来。
    """
    if len(items) < 2:
        return items
    # 按 src 哈希定序（与展示顺序无关），交替指定命中侧 → 批内严格一半一半
    for i, c in enumerate(sorted(items, key=lambda x: _h(x["src"]))):
        want = "left" if i % 2 == 0 else "right"
        if c["key_side"] != want:
            c["left"], c["right"] = c["right"], c["left"]
            c["cov"]["left"], c["cov"]["right"] = c["cov"]["right"], c["cov"]["left"]
            c["key_side"] = want
    return items


def _pool_validate_sets(per_group: int = 8) -> list:
    """**验尺子**：阳性对照。分两档——

    · **硬对照**（她本人 vs 客服腔）：分不出来 → **这把尺子废**，别拿它测任何文件
    · **软对照**（她本人 vs 生成）：量的是「生成离她本人有多远」，
      也给出**灵敏度地板**——比这个差还小的文件效应，这把尺子根本测不出来
    """
    out = []
    her = _pool_her_voice()
    if len(her) < per_group:
        return out
    try:
        from persona_fingerprint import NOT_HER, _existing_generated
        not_her, gen = list(NOT_HER), _existing_generated()
    except Exception:
        not_her, gen = [], []
    # ⚠️ **两组不能共用同一批"她的真话"**：共用的话你连着看到同样 8 句、
    # 还能从第一组认出第二组（用户报过"语句重复"，这里就是一处）。
    pool = her[:per_group * 4]
    if len(pool) < per_group * 4:
        return out
    her_a, her_b = pool[:per_group], pool[per_group:per_group * 2]
    her_c, her_d = pool[per_group * 2:per_group * 3], pool[per_group * 3:per_group * 4]
    # 客服腔只有 6 条，补两条同类（只为凑够 8 条；**不动 persona_fingerprint 里的那份**，
    # 那是 `--validate` 的既有基线，改了会动到已记录的读数）
    extra = ["非常抱歉给您带来不便，我们会尽快为您处理，感谢您的理解与支持！",
             "建议您先尝试重启一下设备，如果问题仍然存在，请随时联系我们。"]
    not_her = (not_her + extra)[:per_group]
    if len(not_her) >= 3:
        out.append(_mk_set("★验尺子·硬对照（她本人 vs 客服腔）",
                           [(t, "real") for t in her_a],
                           [(t, "not_her") for t in not_her],
                           key_side="left", tag="val_hard"))
    gen = [t for t in gen if len(t) >= MIN_CHARS and t not in pool]
    if len(gen) >= per_group:
        out.append(_mk_set("★验尺子·软对照（她本人 vs 生成）",
                           [(t, "real") for t in her_b],
                           [(t, "gen") for t in gen[:per_group]],
                           key_side="left", tag="val_soft"))
    # ⚠️★ **阴性对照（难度匹配的那一个）**：两组**都是她的真话**，随机切一半。
    #
    # 这才是真正的校准，也是回答"硬对照/软对照有什么用"的那个答案：
    #   · 硬对照（vs 客服腔）= 大差别，提示词就能解决 → **过不过都不带信息**（用户 2026-09-30 指出）
    #   · 软对照（vs 生成）= 也是大差别 → 只能证明"能分辨大差别"，
    #     推不出"能分辨两个同源生成版本"（后者才是我们真正要判的）
    #   · **阴性对照 = 难度匹配**：两组本该**没有差别**。
    #       答"分不出" → 尺子干净 ✅
    #       选了一侧   → 你的判断会被**非风格因素**（长度/自称的偶然差异）带走
    #                    → 那么任何文件读数都得先扣掉这个偏置，否则量的是噪声
    #
    # 它回答的是**"87% 该怎么读"**：分不出 → 是"没分辨率"；分得出 → 是"那三段真的不动风格"。
    # 这两种情况的后续动作完全相反，所以这一个数比前面两个都值钱。
    out.append(_mk_set("★验尺子·阴性对照（她本人 vs 她本人）",
                       [(t, "real") for t in her_c],
                       [(t, "real") for t in her_d],
                       key_side=None, tag="val_neg"))
    return _balance_sides([c for c in out if c])


def _pool_sets(per_group: int = 8, sets_per_section: int = 3,
               only_section: str = None) -> list:
    """测文件的整组题：同一个段的**带它 / 不带它**各凑一组，问哪一组更像她。

    ⚠️ 三个约束（都是用户报过的"重复/不等"）：
    ① **同一个 msg 只用一次**（段内不重复）——两组的差别才只有"有没有那段"这一个变量；
    ② **一次只做一个段**（`only_section`）——跨段复用同一批句子 = 你会连着看到同样的话；
    ③ 组与组之间也不重复（按 msg 切成互不相交的块）。
    """
    by_sec = defaultdict(list)
    for c in _pool_pairs():
        if c.get("compare") == "with_vs_without":
            by_sec[c["section"]].append(c)
    out = []
    for sec, pairs in by_sec.items():
        if only_section and only_section != sec:
            continue
        seen, usable = set(), []
        for c in sorted(pairs, key=lambda x: _h(x["src"])):
            if c["msg"] in seen:
                continue
            seen.add(c["msg"])
            usable.append(c)
        if len(usable) < per_group:
            continue
        for s in range(sets_per_section):
            chunk = usable[s * per_group:(s + 1) * per_group]
            if len(chunk) < per_group:
                break
            with_side = [(c["left"]["text"] if c["left"]["cond"] in WITH_CONDS
                          else c["right"]["text"], "with") for c in chunk]
            wo_side = [(c["right"]["text"] if c["left"]["cond"] in WITH_CONDS
                        else c["left"]["text"], "without") for c in chunk]
            item = _mk_set(sec, with_side, wo_side, key_side="left", tag=f"set{s + 1}")
            if item:
                out.append(item)
    return out


def _set_sections_avail(per_group: int) -> dict:
    """每个段能出几组（供 cmd_sets 挑"哪个段"用）。"""
    by_sec = defaultdict(set)
    for c in _pool_pairs():
        if c.get("compare") == "with_vs_without":
            by_sec[c["section"]].add(c["msg"])
    return {s: len(m) // per_group for s, m in by_sec.items()}


def _pool_singles(limit: int = 200) -> list:
    """单条题：同一个问题下，**她的真话** vs **生成的**（蒙住来源）。

    来源是 `persona_ablation.json` 的那 36 条真实消息（它带 with/without 两版生成）。
    """
    real = _real_replies()
    out, seen = [], set()
    p = EVALDIR / "persona_ablation.json"
    if not p.exists():
        return out
    for r in json.loads(p.read_text(encoding="utf-8")).get("rows", []):
        m = _clean(r.get("msg"))
        if not m or m in seen:
            continue
        seen.add(m)
        if m in real:
            _add(out, _mk_single("单条", m, real[m], "real"))
        for cond, k in (("gen_with", "with"), ("gen_without", "without")):
            t = _clean(r.get(k))
            if t:
                _add(out, _mk_single("单条", m, t, cond))
        if len(out) >= limit:
            break
    return out


# ==================== 出题 ====================

PER_MSG_CAP = 2      # 同一个用户问句，一批里最多出几道题


def _pick_mixed(fresh: list, n: int, mix_single, sections_per_batch: int,
                target: int, labeled_count: dict) -> list:
    """配对题**集中出**：一批只做少数几个段，每个段出够条数——单条按比例混进去。

    ⚠️ **为什么不摊开**：第一版让 24 条摊到 18 个小节，每节 n=1 →
    读数表 18 行全是"n 太小，别当数"，**等于白标**。
    一段要 n≥12 才有统计意义，所以必须**先做深、再做广**：
    按"已标条数最少"挑 K 个段，把每个段填向 `target`。
    """
    pairs = [c for c in fresh if c["kind"] == "pair"]
    singles = [c for c in fresh if c["kind"] == "single"]
    if mix_single is None:
        mix_single = max(1, n // 4)
    n_single = min(mix_single, len(singles))
    n_pair = n - n_single

    groups = defaultdict(list)
    for c in pairs:
        groups[c["section"]].append(c)
    for g in groups.values():
        # 「删掉 vs 安慰剂」那问排后面（它问的是另一件事，别一上来就占满）
        g.sort(key=lambda c: (c.get("compare") == "without_vs_placebo", _h(c["src"])))

    # 候选段 = 还没标够 target 的。**先按 PRIORITY 排**（重要的做够 12 条再换下一个），
    # 同一优先级里挑"已标条数最少"的。
    cand = sorted((s for s in groups if labeled_count.get(s, 0) < target),
                  key=lambda s: (_prio(s), labeled_count.get(s, 0), _h(s)))[:sections_per_batch]
    if not cand:
        cand = sorted(groups, key=lambda s: _h(s))[:sections_per_batch]

    # ⚠️ **去重**（用户 2026-09-30：「很多语句都是重复的」）：实测全池 58% 的格位是重复文本，
    # 因为按段消融里**同一个 base 回复会对着 N 个段各出现一次**，而某个问句最多出了 26 道题。
    # 两条闸：① 同一个用户问句最多 PER_MSG_CAP 道；② 同一句回复最多出现 2 次。
    picked, used_text, used_msg, i = [], Counter(), Counter(), 0
    while len(picked) < n_pair:
        added = False
        for s in cand:
            if i >= len(groups[s]) or len(picked) >= n_pair:
                continue
            c = groups[s][i]
            if used_msg[c.get("msg")] >= PER_MSG_CAP:
                continue
            if max(used_text[c["left"]["text"]], used_text[c["right"]["text"]]) >= 2:
                continue
            picked.append(c)
            used_text[c["left"]["text"]] += 1
            used_text[c["right"]["text"]] += 1
            used_msg[c.get("msg")] += 1
            added = True
        if not added:
            break
        i += 1

    picked += sorted(singles, key=lambda c: _h(c["src"]))[:n_single]
    picked.sort(key=lambda c: _h(c["src"]))          # 定序混排：单条不会全挤在末尾
    return picked


def cmd_sets(per_group: int, sets_per_section: int, sections_per_batch: int,
             only: str, validate: bool) -> int:
    doc = _load()
    cases = _cases(doc)
    if MD_FILE.exists():
        cmd_ingest(quiet=True)
    known = {c["src"] for c in cases}

    if validate:
        pool = _pool_validate_sets(per_group)
        if not pool:
            print("❌ 造不出验尺子的对照（chat_log 里她的真实回复不够，或 CHATLOG=0）")
            return 1
        fresh = _balance_sides([c for c in pool if c["src"] not in known])
        if not fresh:
            print("✅ 验尺子的题已经出过了（要重出：把用例库里 kind=set 的删掉）")
            return 0
    else:
        # ⚠️ **一次只做一个段**：跨段必然复用同一批用户问句（好几个段的降维都基于同一行），
        # 你会连着看到同样的话——用户报过这个。做完一个段再跑一次做下一个。
        avail = _set_sections_avail(per_group)
        cand = [s for s, n in avail.items() if n >= 1 and (not only or only in s)]
        if not cand:
            print(f"❌ 没有能凑出整组（每组 {per_group} 条）的段")
            return 1
        sec = sorted(cand, key=lambda s: (_prio(s), _h(s)))[0]
        fresh = _balance_sides([c for c in _pool_sets(per_group, sets_per_section, sec)
                                if c["src"] not in known])
        nxt = [s for s in sorted(cand, key=lambda s: (_prio(s), _h(s))) if s != sec]
        if not fresh:
            print(f"✅ 段「{sec}」的整组题已经出过了"
                  f"（下一个可做的段：{nxt[0] if nxt else '没有了'}）")
            _render_md(cases, RELABEL)
            return 0

    seq = _next_seq(cases)
    import datetime
    today = datetime.date.today().isoformat()
    for c in fresh:
        c["id"] = f"s_{seq:03d}"
        c["created"] = today
        seq += 1
    cases += fresh
    _save(doc)
    _render_md(cases, RELABEL)
    print(f"✅ 新出 {len(fresh)} 组（每组 {per_group} 条 × 2 组）")
    for c in fresh:
        print(f"    {c['section'][:34]:<36}（{c.get('tag', '')}）")
    print("   👉 每组你只答一句『甲组/乙组/分不出』——打开 "
          f"{MD_FILE}")
    if not validate:
        print(f"   下一个可做的段：{nxt[0] if nxt else '没有了'}"
              f"（跑 `style-annotate --sets` 就做它）")
    return 0


def cmd_batch(n: int, only: str, mix_single, sections_per_batch: int, target: int) -> int:
    doc = _load()
    cases = _cases(doc)
    # 先把 md 里已经勾了的收回来（免得出一批新题把旧勾选冲掉）
    if MD_FILE.exists():
        cmd_ingest(quiet=True)

    known = {c["src"] for c in cases}
    fresh = [c for c in _pool_pairs() + _pool_singles() if c["src"] not in known]
    if NAME_SYM_ONLY:
        # 默认只要**自称状态对称**的题：不对称的测不了风格，只会被名字带偏（见 _name_sym）
        fresh = [c for c in fresh if c["kind"] != "pair" or c.get("name_sym")]
    if only:
        fresh = [c for c in fresh if only in c["section"] or only == c["kind"]
                 or only == c.get("compare")]
    if not fresh:
        print("✅ 没有新题可出了（候选池里的都出过了）")
        _render_md(cases, RELABEL)
        return 0

    labeled_count = Counter(c["section"] for c in cases
                            if c.get("label") and c["kind"] == "pair")
    picked = _pick_mixed(fresh, n, mix_single, sections_per_batch, target, labeled_count)
    seq = _next_seq(cases)
    import datetime
    today = datetime.date.today().isoformat()
    for c in picked:
        c["id"] = f"s_{seq:03d}"
        c["created"] = today
        seq += 1
    cases += picked
    _save(doc)
    _render_md(cases, RELABEL)
    secs = Counter(c["section"] for c in picked)
    print(f"✅ 新出 {len(picked)} 条（候选池还剩 {len(fresh) - len(picked)} 条没出）")
    for s, k in secs.most_common():
        print(f"    {s[:34]:<36}{k} 条   （这个段已标 {labeled_count.get(s, 0)}）")
    print(f"   👉 打开 {MD_FILE} 标，标完跑：python scripts/run_tool.py style-annotate --ingest")
    return 0


# ==================== 渲染工作台 ====================

def _one(t: str) -> str:
    """把一条回复压成一行：内部换行 → ` ／ `。

    ⚠️ 她的回复常是**分段发的**（好几个气泡）。直接换行放进 markdown，
    第二段会被空行切断、**归属变得含糊**（看着像不属于甲/乙）。
    用 ` ／ ` 显式标出"她分了几条发"，既不丢信息、也不歧义。
    """
    return re.sub(r"\s*\n+\s*", " ／ ", (t or "").strip())


def _ctx_lines(c: dict) -> list:
    """这道题的前情（最多两轮）。判题要看懂她为什么这么答。"""
    ctx = (_context_index().get(c.get("msg")) or []) if c.get("msg") else []
    if not ctx:
        return []
    out = ["**（前情**（她答这两版时**并没有看到**下面这两轮，是单轮生成的）**）**："]
    for u, rep in ctx:
        out.append(f"- 用户：{_one(u)}")
        if rep:
            out.append(f"  她当时答：{_one(rep)}")
    out.append("")
    return out


def _render_set_body(c: dict) -> list:
    """整组题的正文：甲乙各一列 8 条，编号只为了好数，不带条件标记。

    ⚠️ 两组**都不标条件**（不说哪组有那个文件）——标了就等于把答案写在题面上。
    """
    out = []
    for side, name in (("left", "甲组"), ("right", "乙组")):
        texts = c[side]["texts"]
        out.append(f"**{name}**（{len(texts)} 条）")
        out.append("")
        for k, t in enumerate(texts, 1):
            out.append(f"{k}. {_one(t)}")
        out.append("")
    return out


def _render_md(cases: list, include_labeled: bool = False):
    """把"还没标的"摆出来。

    `include_labeled=True`（= `--relabel`）：**连标过的一起重摆**，用来
    ① 改主意 ② **复标查一致性**（同一批前后两次答案一致率本身就是证据质量——
    本仓那个 traits/styles「两轮一致 89%」就是这么来的）。
    """
    todo = [c for c in cases if include_labeled or not c.get("label")]
    done = len(cases) - len([c for c in cases if not c.get("label")])
    L = ["# 风格标注 —— 先分轴，再比方向", "",
         "> **怎么标**：把方括号里的空格改成 `x`（`[x]`）。**不想碰文件就直接在聊天里答**，我解析。",
         "> **整组题**（两组各若干条）：只答一句——**哪一组更像灰泽满**。"
         "凭**整体感觉**选，别去逐条比；两组都不标来源。",
         "> **第①问是重点**：这两版的**说话方式**到底一样不一样？",
         "> 「只有内容不同」= 说话方式**一样**（那两版风格上没得比，第②问就选『一样』）。",
         "> ⚠️ 这一问**不是走过场**：它本身就是读数——『这个段动不动风格』。",
         "> **第②问只在说话方式真的不一样时才有意义**（一样就选「一样」）。",
         "> ⚠️ **不硬选**：真分不出就选「基本没区别 / 一样」——硬选一个是噪声。",
         "> **左右是打乱过的**，你也猜不出哪句是哪版 —— 凭感觉选，那才是要的。",
         "> **「理由」写一句最值钱**（「这句太书面了」「她不会这么解释」这种）——不写也行。",
         "> 回复里的 ` ／ ` = **她分了几条发**（不是标点，就是气泡分隔）。",
         "> 标完（**可以只标一部分**）跑：`python scripts/run_tool.py style-annotate --ingest`", "",
         f"> 进度：你已标 **{done}** 条 ｜ 还没标 **{len(todo)}** 条", "",
         "---", ""]

    for i, c in enumerate(todo, 1):
        sec = c["section"]
        L.append(f"### {i} · {sec}" + ("　（已标过——这是复标）" if c.get("label") else ""))
        L.append("")
        if c.get("msg"):
            L.append(f"用户说：{c['msg']}")
            L.append("")
        L += _ctx_lines(c)
        if c["kind"] == "pair":
            L += [f"**甲**：{_one(c['left']['text'])}", "",
                  f"**乙**：{_one(c['right']['text'])}", "",
                  f"<!-- s:{c['id']} -->  **① 两版的区别在哪？**  "
                  f"[ ] 只有内容不同   [ ] 只有说话方式不同   [ ] 两样都不同   [ ] 基本没区别",
                  "**② 说话方式上哪版更像她？**  [ ] 甲   [ ] 乙   [ ] 一样"]
        elif c["kind"] == "set":
            L += _render_set_body(c)
            L.append(f"<!-- s:{c['id']} -->  哪一组更像她？  "
                     f"[ ] 甲组   [ ] 乙组   [ ] 分不出")
        else:
            # ⚠️ 单条**绝不显示它来自哪儿**（她本人 / 生成 只在 JSON 里）——蒙住才测得出
            L += [f"**回复**：{_one(c['text'])}", "",
                  f"<!-- s:{c['id']} -->  这句像她吗？  [ ] 像她   [ ] 不像她   [ ] 说不准"]
        L += ["理由：", "", "---", ""]

    if not todo:
        L.append("（没有待标的了。要出新题：`python scripts/run_tool.py style-annotate`）")
    MD_FILE.write_text("\n".join(L), encoding="utf-8")


# ==================== 回读 ====================

def cmd_ingest(quiet: bool = False) -> int:
    if not MD_FILE.exists():
        print(f"❌ 没有 {MD_FILE}（先跑一次 `style-annotate` 出题）")
        return 1
    doc = _load()
    cases = _cases(doc)
    by_id = {c["id"]: c for c in cases}
    txt = MD_FILE.read_text(encoding="utf-8")
    parts = re.split(r"<!--\s*s:(s_\d+)\s*-->", txt)

    # ⚠️ 防呆：锚点外壳被破坏时，上面那个 split 会一条都切不出来，
    # 于是"一条都没收到"——看着像"你没勾"，其实是文件坏了。
    # 本仓的纪律是**别让脚本假报成功**，这里也别让它假报"你没标"。
    if len(parts) == 1 and re.search(r"\bs_\d{3}\b", txt):
        print("❌ 风格标注.md 里的锚点被破坏了（`<!-- s:xxx -->` 少了 `<!--` 或 `-->`）。")
        print("   多半是编辑器/脚本重写过这个文件。**重新出一批**即可：")
        print("   python scripts/run_tool.py style-annotate          # 会把还没标的重摆一遍")
        return 1

    got, missing_id, per = 0, [], Counter()
    for i in range(1, len(parts), 2):
        cid, seg = parts[i], parts[i + 1]
        c = by_id.get(cid)
        if c is None:
            missing_id.append(cid)
            continue
        # 第①问（只有 pair 有）：这两版的区别在内容还是在说话方式
        diff = None
        if c["kind"] == "pair":
            for rx, lab in DIFF_RE:
                if rx.search(seg):
                    diff = lab
                    break
        pick = None
        for rx, lab in PICK_RE[c["kind"]]:
            if rx.search(seg):
                pick = lab
                break
        if pick is None and diff is None:
            continue
        # ① 说「只有内容不同 / 基本没区别」= **说话方式上两版就是一样** →
        # 第②问留空也按"一样"记（否则这道题会一直挂在待标里）。
        if pick is None and diff in ("content", "none"):
            pick = "tie"
        if c.get("label") and c["label"] != pick and not RELABEL:
            continue                                   # 已有标注，不覆盖
        if diff is not None:
            c["diff"] = diff
        if pick is not None:
            c["label"] = pick
        m = re.search(r"理由[:：]\s*([^\n]+)", seg)
        if m and m.group(1).strip():
            c["why"] = m.group(1).strip()
        got += 1
        per[c["kind"]] += 1

    _save(doc)
    if not quiet:
        if got:
            print(f"✅ 收进 {got} 条（" + "、".join(f"{k} {v}" for k, v in per.items()) + "）")
        else:
            print("（一条都没收到。md 里没找到任何勾——也可能是**编辑器没保存**。）")
            print("   不想碰文件就用文字答：`--answers`（格式见 --help）")
        if missing_id:
            print(f"⚠️ md 里有 {len(missing_id)} 个锚点在库里找不到（{missing_id[:3]}…）"
                  f"—— 多半是我重出了题，那条作废即可")
    return 0


# ==================== 用文字答（不碰文件）====================

# 用户可以在聊天里直接答，格式随意（一行一题）：
#     1 内容
#     2 风格 甲
#     3 都有 乙
#     4 无
# 或者单条题：  5 像   /  6 不像  /  7 不准
DIFF_WORDS = {"内容": "content", "风格": "style", "都有": "both", "都": "both",
              "无": "none", "没区别": "none", "一样": "none", "同": "none"}
PICK_WORDS = {"甲": "left", "乙": "right", "一样": "tie", "差不多": "tie", "平": "tie",
              "a": "left", "b": "right"}
SINGLE_WORDS = {"像": "like", "像她": "like", "不像": "unlike", "不像她": "unlike",
                "不准": "unsure", "说不准": "unsure", "不知道": "unsure"}
SET_WORDS = {"甲组": "left", "乙组": "right", "分不出": "tie", "甲": "left", "乙": "right",
             "组甲": "left", "组乙": "right"}


def _parse_answer_line(line: str, kind: str):
    """解析一行文字答案 → (diff, label)。认不出来返回 (None, None)。"""
    toks = [t.strip(" ,，。.、:：") for t in re.split(r"[\s,，、]+", line.strip()) if t.strip()]
    diff = pick = None
    for t in toks:
        if t in DIFF_WORDS and kind == "pair":
            diff = DIFF_WORDS[t]
        elif t in SINGLE_WORDS and kind == "single":
            pick = SINGLE_WORDS[t]
        elif t in SET_WORDS and kind == "set":
            pick = SET_WORDS[t]
        elif t in PICK_WORDS and kind == "pair":
            pick = PICK_WORDS[t]
    if diff is None and pick is None:
        return None, None
    # 「只有内容不同 / 基本没区别」= 说话方式一样
    if kind == "pair" and pick is None and diff in ("content", "none"):
        pick = "tie"
    return diff, pick


def cmd_answers(text: str) -> int:
    """**不碰 md**：直接把文字答案灌进去。序号 = 工作台里的题号（1 起）。"""
    doc = _load()
    cases = _cases(doc)
    # 题号 = 工作台里的序号 = **还没标的那些**（顺序与 _render_md 一致）
    todo = [c for c in cases if c.get("label") is None]

    by_id = {c["id"]: c for c in cases}
    got, bad, applied = 0, [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # ① **id 形式**（最稳，推荐）：`s_013 乙` —— 不受题号顺序影响
        m = re.match(r"^(s_\d+)\s+(.*)$", line)
        if m:
            c = by_id.get(m.group(1))
            if c is None:
                bad.append(line)
                continue
            rest = m.group(2)
        else:
            # ② 序号形式：序号 = **`风格标注.md` 里的题号**（= 还没标的那些的顺序）
            m = re.match(r"^(\d+)\s*[.、:：)]?\s*(.*)$", line)
            if not m:
                bad.append(line)
                continue
            n, rest = int(m.group(1)), m.group(2)
            if not (1 <= n <= len(todo)):
                bad.append(line)
                continue
            c = todo[n - 1]
        diff, pick = _parse_answer_line(rest, c["kind"])
        if diff is None and pick is None:
            bad.append(line)
            continue
        if diff is not None:
            c["diff"] = diff
        if pick is not None:
            c["label"] = pick
        got += 1
        applied.append((c["id"], c["kind"], (c.get("msg") or c.get("section"))[:22],
                        diff or "", pick or ""))

    _save(doc)
    # ⚠️ **逐条回声**：序号形式一旦对不上（比如 md 里有你没看到的题排在前面），
    # 标注会整体错位贴到**别的题**上——而且它会"成功"、不报错。
    # 本仓真踩过：13 条答案错位贴到了 12 道不相干的题上。回声让人一眼看出错位。
    print(f"✅ 收进 {got} 条文字答案　（逐条核对下面这张表：句子对不对得上你说的那道题？）")
    for cid, kind, label, diff, pick in applied:
        print(f"     {cid:<7}{kind:<7}{label:<26}{diff:<8}{pick}")
    if bad:
        print(f"⚠️ 这几行没认出来（格式：`序号 甲/乙/一样` 或 `s_013 乙`；"
              f"成对题可加 `内容/风格/都有/无`）：")
        for b in bad[:6]:
            print(f"     {b}")
    return 0


# ==================== 读数 ====================

def _binom_two_sided(k: int, n: int) -> float:
    """双侧精确二项检验（p=0.5）。纯 python，不引 scipy。

    ⚠️ **必须双侧**：本仓踩过——只算下尾再乘 2，会出现"100% 却 p=1.0"。
    这里用"小概率法"：把所有 p(i) ≤ p(k) 的结果概率加起来。
    """
    from math import comb
    if n <= 0:
        return 1.0
    pmf = [comb(n, i) * 0.5 ** n for i in range(n + 1)]
    obs = pmf[k]
    return min(1.0, sum(p for p in pmf if p <= obs * (1 + 1e-9)))


def _sig(p: float, n: int) -> str:
    if n < 12:
        return "n 太小（<12），别当数"
    return "显著 p<0.05" if p < 0.05 else ("弱 " if p < 0.15 else "不显著")


def cmd_report() -> int:
    doc = _load()
    cases = _cases(doc)
    labeled = [c for c in cases if c.get("label")]
    if not labeled:
        print("❌ 还没有任何标注（跑 --batch 出题 → 标 → --ingest）")
        return 1

    # ---- 成对题 ----
    per = defaultdict(lambda: {"with": 0, "without": 0, "tie": 0, "n": 0, "diff": Counter()})
    plc = defaultdict(lambda: {"ablated": 0, "placebo": 0, "tie": 0, "n": 0, "diff": Counter()})
    for c in labeled:
        if c["kind"] != "pair":
            continue
        bucket = plc if c.get("compare") == "without_vs_placebo" else per
        d = bucket[c["section"]]
        d["n"] += 1
        if c.get("diff"):
            d["diff"][c["diff"]] += 1
        if c["label"] == "tie":
            d["tie"] += 1
            continue
        side = c["left"] if c["label"] == "left" else c["right"]
        if bucket is plc:
            d[side["cond"]] += 1                             # "ablated" / "placebo"
        else:
            d["with" if side["cond"] in WITH_CONDS else "without"] += 1

    # ---- 单条题：生成离她本人有多远 ----
    sg = defaultdict(Counter)
    for c in labeled:
        if c["kind"] == "single":
            grp = "她本人真实" if c["cond"] == "real" else "生成"
            sg[grp][c["label"]] += 1

    n_pair = sum(d["n"] for d in per.values()) + sum(d["n"] for d in plc.values())
    n_tie = sum(d["tie"] for d in per.values()) + sum(d["tie"] for d in plc.values())

    # ==================== ① 先报仪器的分辨力 ====================
    # ⚠️ **这一节放最前面**。2026-09-30 第一批：23 道成对题里 20 道答"差不多"（87%）
    # ——那不是在测那几个文件，是在测**这批题有没有东西可判**。
    # 不先报这个数，后面那些"占比"会把人骗死（分母只剩 3 条）。
    print(f"\n{'=' * 76}")
    print(f"【一、仪器分辨力】你答「差不多/一样」的比例　（共 {n_pair} 道成对题）")
    print("-" * 76)
    if n_pair:
        rate = n_tie / n_pair
        print(f"  {n_tie}/{n_pair} = **{rate * 100:.0f}%** 分不出")
        if rate > 0.6:
            print("  ⚠️ **这批题大半没东西可判** → 后面每段的『占比』分母都很小，别当真。")
            print("     根因通常是：两版是同一个模型 + 同一套 base 骨架生成的，")
            print("     风格天然一样、只有信息量不同 → 要问的是**『差在内容还是说话方式』**（v2 第一问）。")
        elif rate > 0.35:
            print("  勉强能用，但每一段的读数都要看 n + p，别只看百分比。")
        else:
            print("  分辨力可以：多数题你能分出高下。")
    print()
    print(f"{'文件 / 小节':<26}{'题数':>5}{'差不多':>7}{'差多少':>8}   还能用吗")
    print("-" * 76)
    for sec, d in sorted(per.items(), key=lambda kv: kv[1]["tie"] / max(kv[1]["n"], 1)):
        r = d["tie"] / max(d["n"], 1)
        tag = "❌ 基本全平（这题没东西可判）" if r > 0.75 else \
              "⚠️ 偏多，读数要打折" if r > 0.5 else "✅ 有分辨力"
        print(f"{sec[:24]:<26}{d['n']:>5}{d['tie']:>7}{r * 100:>7.0f}%   {tag}")

    # ==================== ② 分轴：差在内容还是在说话方式 ====================
    have_diff = {s: d for s, d in per.items() if d["diff"]}
    if have_diff:
        print(f"\n【二、差在哪一轴】（v2 第一问）")
        print(f"{'文件 / 小节':<26}{'只有内容':>9}{'说话方式':>9}{'两样都':>8}{'没区别':>8}")
        print("-" * 76)
        for sec, d in sorted(have_diff.items()):
            c = d["diff"]
            print(f"{sec[:24]:<26}{c['content']:>9}{c['style']:>9}{c['both']:>8}{c['none']:>8}")
        print("\n  ⚠️ 读法：『只有内容』多 = 这个段管的是**内容**，该改用**任务正确性**判据测它")
        print("     （答案在文件里、可自动判，见 `docs/待办清单.md` 那节）；")
        print("     『说话方式』多 = 这个段真的在动**风格**，下面那张表才成立。")

    # ==================== ③ 方向 ====================
    print(f"\n【三、方向】「有那个文件/小节」的那版被选中的比例")
    print(f"{'文件 / 小节':<26}{'有它更像':>8}{'删掉更像':>9}{'占比':>8}{'p':>8}   判定")
    print("-" * 76)
    rows_out = []
    for sec, d in sorted(per.items(), key=lambda kv: -(kv[1]["with"] + kv[1]["without"])):
        dec = d["with"] + d["without"]
        if dec == 0:
            print(f"{sec[:24]:<26}{'—':>8}{'—':>9}{'—':>8}{'—':>8}   ⛔ 全是『差不多』，无方向")
            rows_out.append({"section": sec, "compare": "with_vs_without", **d,
                             "rate": None, "p": None, "n_decided": 0})
            continue
        rate = d["with"] / dec
        p = _binom_two_sided(d["with"], dec)
        print(f"{sec[:24]:<26}{d['with']:>8}{d['without']:>9}"
              f"{rate * 100:>7.0f}%{p:>8.3f}   {_sig(p, dec)}")
        rows_out.append({"section": sec, "compare": "with_vs_without",
                         "with": d["with"], "without": d["without"], "tie": d["tie"],
                         "n": d["n"], "rate": rate, "p": p, "n_decided": dec})
    if not rows_out:
        print("（还没有成对题被标）")

    if plc:
        print(f"\n【四、是内容在起作用，还是只是上下文变短了？】")
        print("（删掉 vs **等长无关填充**；选『等长无关填充更像她』= 那段内容确实有作用）")
        print(f"{'文件 / 小节':<26}{'等长填充':>9}{'删掉':>7}{'差不多':>7}   判定")
        print("-" * 76)
        for sec, d in sorted(plc.items(), key=lambda kv: -(kv[1]["placebo"] + kv[1]["ablated"])):
            dec = d["placebo"] + d["ablated"]
            if dec == 0:
                print(f"{sec[:24]:<26}{'—':>9}{'—':>7}{d['tie']:>7}   ⛔ 全是差不多")
                continue
            p = _binom_two_sided(d["placebo"], dec)
            print(f"{sec[:24]:<26}{d['placebo']:>9}{d['ablated']:>7}{d['tie']:>7}"
                  f"   {_sig(p, dec)}")
            rows_out.append({"section": sec, "compare": "without_vs_placebo",
                             "placebo": d["placebo"], "ablated": d["ablated"],
                             "tie": d["tie"], "n": d["n"],
                             "rate": d["placebo"] / dec, "p": p, "n_decided": dec})

    print(f"\n【五、单条读数】她本人真实 vs 生成（**来源是蒙住的**）")
    if sg:
        for grp, cc in sg.items():
            tot = sum(cc.values())
            print(f"  {grp:<8}：像她 {cc['like']}｜不像她 {cc['unlike']}｜说不准 {cc['unsure']}"
                  f"　（共 {tot}）")
        if len(sg) == 2:
            a = sg["她本人真实"]; b = sg["生成"]
            ra = a["like"] / max(sum(a.values()), 1)
            rb = b["like"] / max(sum(b.values()), 1)
            print(f"  → 她本人被判像她的比例 {ra * 100:.0f}%，生成 {rb * 100:.0f}%")
            print("    （这个差就是「生成离她本人有多远」的第一次量；n 小，只当方向）")
    else:
        print("（还没有单条题被标）")

    # ==================== 六、整组读数（v3：人判风格是看一批）====================
    sets = [c for c in labeled if c["kind"] == "set"]
    if sets:
        print(f"\n{'=' * 76}")
        print(f"【六、整组读数】「哪一组更像她」　（共 {len(sets)} 组）")
        print("-" * 76)
        val = [c for c in sets if c["section"].startswith("★验尺子")]
        abl = [c for c in sets if not c["section"].startswith("★验尺子")]

        for c in val:
            if c.get("key_side") is None:                 # 阴性对照：没有"正确答案"
                mark = ("⛔ 分不出（**这是干净的表现**）" if c["label"] == "tie"
                        else "⚠️ 你选了" + ("甲组" if c["label"] == "left" else "乙组")
                             + "——但两组**都是她的真话**，这一选是噪声/偏置")
            elif c["label"] == "tie":
                mark = "⛔ 分不出"
            else:
                mark = "✅ 选对" if c["label"] == c["key_side"] else "❌ 选反"
            print(f"  {c['section'][:34]:<36} {mark}")
        dec = [c for c in val if c.get("key_side") and c["label"] != "tie"]
        if dec:
            ok = sum(1 for c in dec if c["label"] == c["key_side"])
            print(f"\n  → 大差别对照：{ok}/{len(dec)} 组选对"
                  f"{'　⚠️ 大差别都分不出 → 什么都别测了' if ok < len(dec) else ''}"
                  "　⚠️ **但过了它不代表能用**：大差别（vs 客服腔、vs 生成）提示词就能解决，"
                  "推不出能分辨两个同源生成版本。")
        neg = [c for c in val if c.get("key_side") is None]
        if neg:
            clean = sum(1 for c in neg if c["label"] == "tie")
            print(f"  → **阴性对照（难度匹配，真正算数的那个）**：{clean}/{len(neg)} 组答『分不出』")
            if clean == len(neg):
                print("     ✅ 尺子干净：两组本该一样，你没硬选 → 那么『差不多率』高**是文件的真实情况**，")
                print("        不是尺子没分辨率。可以据此下结论。")
            else:
                print("     ⚠️ 两组**都是她的真话**你却选出高下 → **你的判断会被非风格因素带走**")
                print("        （看下面的协变量：多半是长度/自称的偶然差）→ 任何文件读数都得先扣掉这个偏置，")
                print("        否则量的不是风格，是噪声。")

        if abl:
            per_set = defaultdict(lambda: {"n": 0, "hit": 0, "tie": 0})
            for c in abl:
                d = per_set[c["section"]]
                d["n"] += 1
                if c["label"] == "tie":
                    d["tie"] += 1
                elif c["label"] == c["key_side"]:
                    d["hit"] += 1
            print(f"\n{'文件 / 小节':<26}{'组数':>5}{'选中带它的':>10}{'分不出':>7}{'占比':>8}{'p':>8}")
            print("-" * 76)
            for sec, d in sorted(per_set.items()):
                dec = d["n"] - d["tie"]
                if dec == 0:
                    print(f"{sec[:24]:<26}{d['n']:>5}{'—':>10}{d['tie']:>7}{'—':>8}{'—':>8}")
                    continue
                rate = d["hit"] / dec
                p = _binom_two_sided(d["hit"], dec)
                print(f"{sec[:24]:<26}{d['n']:>5}{d['hit']:>10}{d['tie']:>7}"
                      f"{rate * 100:>7.0f}%{p:>8.3f}   {_sig(p, dec)}")
                rows_out.append({"section": sec, "compare": "group_set",
                                 "n": d["n"], "hit": d["hit"], "tie": d["tie"],
                                 "rate": rate, "p": p, "n_decided": dec})

            # ⚠️ 协变量：你选的那组是不是**只是更长 / 更多自称**？
            print(f"\n  【协变量】你选的那组，平均长度/自称数是不是更大？（不然读数可能是这两样带的）")
            print(f"{'文件 / 小节':<26}{'选中组均长':>10}{'另一组':>8}{'选中组自称':>10}{'另一组':>8}")
            print("-" * 76)
            for sec, _d in sorted(per_set.items()):
                cs = [c for c in abl if c["section"] == sec and c["label"] in ("left", "right")]
                if not cs:
                    continue
                hit_cov = [c["cov"][c["label"]] for c in cs]
                oth_cov = [c["cov"]["right" if c["label"] == "left" else "left"] for c in cs]
                al = sum(x["avg_len"] for x in hit_cov) / len(hit_cov)
                bl = sum(x["avg_len"] for x in oth_cov) / len(oth_cov)
                asf = sum(x["self"] for x in hit_cov) / len(hit_cov)
                bsf = sum(x["self"] for x in oth_cov) / len(oth_cov)
                print(f"{sec[:24]:<26}{al:>10.1f}{bl:>8.1f}{asf:>10.1f}{bsf:>8.1f}")
            print("  ⚠️ 两列**差得远** → 你的判断可能跟着长度/自称走了，不一定是风格")
    else:
        print(f"\n（还没有整组题被标。出题：`style-annotate --validate` 或 `--sets`）")

    print("\n⚠️ 怎么读：")
    print("  · **先看【一】再往下看其余**——『差不多』率太高时，后面的占比没有意义")
    print("  · 占比**明显高于 50%** = 那个文件在让她更像她；**50% 附近** = 分不出")
    print("  · 只报占比不报 p 是骗人（本仓踩过『100% 却 p=1.0』）——n 和 p 一起看")

    out = EVALDIR / "style_readings.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_labeled": len(labeled), "n_pair": n_pair, "n_tie": n_tie,
                               "sections": rows_out,
                               "axes": {s: dict(d["diff"]) for s, d in per.items() if d["diff"]},
                               "singles": {k: dict(v) for k, v in sg.items()},
                               "why": [{"id": c["id"], "section": c["section"],
                                        "diff": c.get("diff"), "label": c["label"],
                                        "why": c["why"]}
                                       for c in labeled if c.get("why")]},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✅ 读数已保存：{out}")
    return 0


def cmd_status() -> int:
    doc = _load()
    cases = _cases(doc)
    todo = [c for c in cases if not c.get("label")]
    print(f"用例库：{CASES_FILE}")
    print(f"  共 {len(cases)} 条 ｜ 已标 {len(cases) - len(todo)} ｜ 还没标 {len(todo)}")
    per = Counter((c["kind"], c["section"]) for c in cases if c.get("label"))
    for (k, s), v in sorted(per.items()):
        print(f"    {k:<7}{s[:30]:<32}{v}")
    return 0


# ==================== 给没测过的段现生成对照 ====================

def cmd_run(section: str, n: int) -> int:
    """调 `persona_ablation.py --sections --only-section X` 现生成一批对照。

    ⚠️ 要跑 LLM（每条 1 次检索 + N 次生成）——只在"已有产物里没有这个段"时才需要。
    """
    cmd = [sys.executable, str(SCRIPTS / "persona_ablation.py"),
           "--sections", "--n", str(n), "--only-section", section]
    print(f"↳ 现生成对照：{' '.join(cmd[-5:])}\n")
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"❌ 生成失败（退出码 {rc}）")
        return rc
    print("\n✅ 生成完了。接着出题：python scripts/run_tool.py style-annotate")
    return 0


# ==================== 入口 ====================

RELABEL = False
NAME_SYM_ONLY = True


def main(argv=None) -> int:
    global RELABEL
    ap = argparse.ArgumentParser(description="风格层标注集：出题 / 回读 / 读数",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--batch", type=int, default=20, help="出一批题，默认 20 条")
    ap.add_argument("--ingest", action="store_true", help="回读 风格标注.md 里你勾的")
    ap.add_argument("--report", action="store_true", help="出消融读数（按文件/小节 + p 值）")
    ap.add_argument("--status", action="store_true", help="看进度")
    ap.add_argument("--only", default="", help="只出这一类：pair / single / real / 段名的一部分")
    ap.add_argument("--mix-single", type=int, default=None, help="一批里混几道单条题（默认 1/4）")
    ap.add_argument("--sections-per-batch", type=int, default=3,
                    help="一批只做几个段（默认 3）。**摊开会让每段的 n=1，读数全废**")
    ap.add_argument("--target", type=int, default=12,
                    help="每个段标够多少条算够（默认 12；不到 12 的读数别当数）")
    ap.add_argument("--relabel", action="store_true", help="允许覆盖已有的标注（默认不覆盖）")
    ap.add_argument("--name-asym", action="store_true",
                    help="也出『只有一版自称灰泽满』的题（默认不出——那种测不了风格，只会被名字带偏）")
    ap.add_argument("--answers", default=None,
                    help="★用文字答，不碰文件。每行：`序号 内容/风格/都有/无 [甲/乙/一样]`"
                         "（单条题：`序号 像/不像/不准`；整组题：`序号 甲组/乙组/分不出`）。传 `-` 从 stdin 读")
    ap.add_argument("--sets", action="store_true",
                    help="★出**整组题**（每组 8 条 × 2 组，你只答一句）——人判风格是看一批，不是看一对")
    ap.add_argument("--validate", action="store_true",
                    help="★先跑这个：出**验尺子**的两道整组题（她本人 vs 客服腔 / vs 生成）。"
                         "分不出就别拿这把尺子测任何文件")
    ap.add_argument("--per-group", type=int, default=8, help="整组题每组几条（默认 8）")
    ap.add_argument("--sets-per-section", type=int, default=3, help="每个段出几组（默认 3，用来查稳定性）")
    ap.add_argument("--run", default="", help="★现生成某个段的对照（要跑 LLM，慢、花钱）")
    args = ap.parse_args(argv)
    _common.ensure_utf8_stdout()
    RELABEL = args.relabel
    NAME_SYM_ONLY = not args.name_asym

    if args.validate or args.sets:
        return cmd_sets(args.per_group, args.sets_per_section,
                        args.sections_per_batch, args.only, args.validate)
    if args.run:
        return cmd_run(args.run, args.batch)
    if args.answers is not None:
        text = sys.stdin.read() if args.answers == "-" else args.answers
        return cmd_answers(text)
    if args.ingest:
        return cmd_ingest()
    if args.report:
        return cmd_report()
    if args.status:
        return cmd_status()
    return cmd_batch(args.batch, args.only, args.mix_single,
                     args.sections_per_batch, args.target)


if __name__ == "__main__":
    sys.exit(main())
