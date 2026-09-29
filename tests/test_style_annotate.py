# -*- coding: utf-8 -*-
"""风格层标注集（`scripts/style_annotate.py`）的测试。

为什么值得测：这份是**风格层唯一的判据**（检索层有 retrieval_eval_cases、
生成层有 regression_cases，风格层原来什么都没有 → 所有风格类结论都不可靠）。
它的失效方式和本仓吃过的亏是同一种：**静默**。

踩过的实例（都写成了用例）：

1. **锚点被吃掉** → 回读时一条都切不出来，打印的却是"一条都没收到——
   是不是还没把 `[ ]` 改成 `[x]`？"，**把文件坏了说成你没标**。
   根因：`re.split` 带捕获组会把 `<!-- s:xxx -->` 外壳一起吃掉，
   谁拿它的结果去重写 md，锚点就没了。
2. **单条题泄露来源** → 单条题比的是"她本人真实回复 vs 生成的"，
   一旦显示 cond（real / gen），题就变成"猜这题出得对不对"，量的东西全歪。
   所以渲染出的 md **不许出现** cond 或"她本人/生成"这类字眼。
3. **p 值只能双侧** → 本仓踩过"只算下尾再乘 2 → 100% 却 p=1.0"。
4. **两版几乎一样的题是废题** → 你只会答"差不多"，白占注意力。

不联网、不调模型：只测纯函数与文件读写。
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import style_annotate as sa  # noqa: E402


# ==================== 纯函数 ====================

class TestDegenerate:
    """摆给人看之前，先扔掉"没有东西可判"的题。"""

    def test_near_identical_pair_is_dropped(self):
        assert not sa._ok_pair("晚上好晚上好，复活了复活了。", "晚上好晚上好，复活了复活了！")

    def test_real_difference_survives(self):
        # 一版多一句 —— 这就是**有东西可判**，不能因为"是前缀"就扔掉
        assert sa._ok_pair("晚上好晚上好，复活了复活了。",
                           "晚上好晚上好，复活了复活了。抱抱一下，至于吗你。")

    def test_too_short_is_dropped(self):
        assert not sa._ok_pair("？", "？")
        assert not sa._ok_single("？", "？")

    def test_single_echoing_the_question_is_dropped(self):
        # 她答的就等于用户问的那句 —— 没有任何信息量
        assert not sa._ok_single("真的骚", "真的骚")

    def test_mk_pair_returns_none_for_junk(self):
        assert sa._mk_pair("【X】", "with_vs_without", "在吗", "嗯。", "base", "嗯", "ablated") is None

    def test_mk_single_returns_none_for_junk(self):
        assert sa._mk_single("单条", "？", "？", "gen_with") is None


class TestSectionName:
    """段名既当库里的 key、又要摆给人看，两个要求不一样。"""

    def test_strips_markdown_heading(self):
        assert sa._norm_section("## 【核心人格】") == "【核心人格】"

    def test_repairs_truncated_bracket(self):
        # 骨架小节标题在 persona_ablation._base_units 里被截到 16 字
        assert sa._norm_section("## 【括号与省略号：例外不是常") == "【括号与省略号：例外不是常】"

    def test_leaves_good_name_alone(self):
        assert sa._norm_section("【性格基底】+【语言风格】") == "【性格基底】+【语言风格】"

    def test_priority_beats_hash_order(self):
        # 【性格基底】必须排在没进 PRIORITY 的段前面（顺序 = 先做哪个）
        assert sa._prio("【性格基底】+【语言风格】") < sa._prio("【灰泽满的周表】")


class TestBinomial:
    """读数必须带 p 值，而且**必须双侧**。"""

    @pytest.mark.parametrize("k,n,expect", [
        (0, 6, 0.03125),        # 2 * (1/2^6)
        (6, 6, 0.03125),        # 对称 —— 这正是"只算下尾再乘 2"会算错的地方
        (3, 6, 1.0),            # 正好一半 → 完全没有证据
        (0, 12, 0.00048828125),
        (6, 12, 1.0),
    ])
    def test_two_sided_exact(self, k, n, expect):
        assert sa._binom_two_sided(k, n) == pytest.approx(expect, rel=1e-6)

    def test_symmetric(self):
        for k in range(0, 9):
            assert sa._binom_two_sided(k, 8) == pytest.approx(sa._binom_two_sided(8 - k, 8))

    def test_zero_n_is_one(self):
        assert sa._binom_two_sided(0, 0) == 1.0

    def test_sig_refuses_small_n(self):
        # p 再小，n<12 也一律"别当数"——本仓反复吃过小样本的亏
        assert "别当数" in sa._sig(0.001, 6)
        assert "显著" in sa._sig(0.001, 12)


class TestOneLine:
    """她的回复常分几条发，换行直接进 md 会让归属含糊。"""

    def test_newlines_become_separator(self):
        assert sa._one("第一句\n\n第二句") == "第一句 ／ 第二句"

    def test_single_line_untouched(self):
        assert sa._one(" 就一句 ") == "就一句"


# ==================== 文件闭环 ====================

@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """把用例库和 md 都指到临时目录 —— 绝不碰真的 风格标注.md。"""
    cases = tmp_path / "style_eval_cases.json"
    md = tmp_path / "风格标注.md"
    monkeypatch.setattr(sa, "CASES_FILE", cases)
    monkeypatch.setattr(sa, "MD_FILE", md)
    monkeypatch.setattr(sa, "RELABEL", False)
    cases.write_text(json.dumps({"cases": []}, ensure_ascii=False), encoding="utf-8")
    return cases, md


def _seed(cases_path, items):
    cases_path.write_text(json.dumps({"cases": items}, ensure_ascii=False), encoding="utf-8")


def _case(cid, kind, **kw):
    base = {"id": cid, "kind": kind, "section": "【性格基底】+【语言风格】",
            "msg": "你终于复活了我想死你了", "label": None, "why": "", "src": f"x:{cid}"}
    base.update(kw)
    return base


# ⚠️ 两版必须**真的不一样**，否则会被 `_ok_pair` 挡掉（NEAR_SAME=0.75）。
PAIR = dict(left={"text": "行吧行吧，那灰泽满先去睡了，你也别熬太晚。", "cond": "with"},
            right={"text": "嗯，知道了。", "cond": "without"},
            compare="with_vs_without")
SINGLE = dict(text="灰泽满刚洗完澡瘫着呢，你咋还不睡", cond="gen_with")


def _pair_texts(case):
    return case["left"]["text"], case["right"]["text"]


class TestRoundTrip:
    def test_render_ingest_report(self, sandbox, capsys):
        cases_path, md = sandbox
        _seed(cases_path, [_case("s_001", "pair", **PAIR), _case("s_002", "single", **SINGLE)])
        sa._render_md(sa._cases(sa._load()))

        # 假标：pair 选「甲」，single 选「像她」
        md.write_text(md.read_text(encoding="utf-8")
                      .replace("[ ] 甲", "[x] 甲", 1)
                      .replace("[ ] 像她", "[x] 像她", 1)
                      .replace("理由：", "理由：这句才像她", 1), encoding="utf-8")

        assert sa.cmd_ingest(quiet=True) == 0
        by_id = {c["id"]: c for c in sa._cases(sa._load())}
        assert by_id["s_001"]["label"] == "left"
        assert by_id["s_002"]["label"] == "like"
        assert "这句才像她" in by_id["s_001"]["why"]

        assert sa.cmd_report() == 0
        out = capsys.readouterr().out
        assert "仪器分辨力" in out and "单条读数" in out

    def test_ingest_does_not_overwrite_existing_label(self, sandbox):
        cases_path, md = sandbox
        _seed(cases_path, [_case("s_001", "pair", label="right", **PAIR)])
        sa._render_md(sa._cases(sa._load()))
        md.write_text(md.read_text(encoding="utf-8").replace("[ ] 甲", "[x] 甲", 1),
                      encoding="utf-8")
        sa.cmd_ingest(quiet=True)
        assert {c["id"]: c for c in sa._cases(sa._load())}["s_001"]["label"] == "right"

    def test_relabel_reshows_labelled_items_and_allows_overwrite(self, sandbox, monkeypatch):
        """`--relabel` 不是一个死开关：它得**把标过的重新摆出来**才能改。

        （第一版它只改了"允不允许覆盖"，而标过的根本不再渲染 → 无从下手，
        被测试当场抓住。顺带它也是**复标查一致性**的入口。）
        """
        cases_path, md = sandbox
        _seed(cases_path, [_case("s_001", "pair", label="right", **PAIR)])
        monkeypatch.setattr(sa, "RELABEL", True)
        sa._render_md(sa._cases(sa._load()), include_labeled=sa.RELABEL)
        txt = md.read_text(encoding="utf-8")
        assert "<!-- s:s_001 -->" in txt and "复标" in txt

        md.write_text(txt.replace("[ ] 甲", "[x] 甲", 1), encoding="utf-8")
        sa.cmd_ingest(quiet=True)
        assert {c["id"]: c for c in sa._cases(sa._load())}["s_001"]["label"] == "left"

    def test_labelled_items_are_not_reshown_by_default(self, sandbox):
        cases_path, md = sandbox
        _seed(cases_path, [_case("s_001", "pair", label="right", **PAIR),
                           _case("s_002", "pair", **PAIR)])
        sa._render_md(sa._cases(sa._load()))
        txt = md.read_text(encoding="utf-8")
        assert "<!-- s:s_001 -->" not in txt          # 标过的不再占你注意力
        assert "<!-- s:s_002 -->" in txt

    def test_unlabelled_survives_a_new_batch(self, sandbox):
        """只标了一半，重出题不许把没标的丢掉。"""
        cases_path, md = sandbox
        _seed(cases_path, [_case("s_001", "pair", **PAIR), _case("s_002", "pair", **PAIR)])
        sa._render_md(sa._cases(sa._load()))
        md.write_text(md.read_text(encoding="utf-8").replace("[ ] 甲", "[x] 甲", 1),
                      encoding="utf-8")
        sa.cmd_ingest(quiet=True)
        sa._render_md(sa._cases(sa._load()))
        txt = md.read_text(encoding="utf-8")
        assert txt.count("<!-- s:") == 1             # 没标的那条**还在**（没被丢掉）
        assert "<!-- s:s_002 -->" in txt


class TestBlinding:
    """单条题**绝不能**让人看出它来自哪儿。"""

    def test_single_render_hides_cond(self, sandbox):
        cases_path, md = sandbox
        _seed(cases_path, [_case("s_001", "single", cond="real", text="灰泽满认栽了行吧")])
        sa._render_md(sa._cases(sa._load()))
        txt = md.read_text(encoding="utf-8")
        assert "real" not in txt
        assert "她本人" not in txt and "生成" not in txt
        assert "cond" not in txt

    def test_pair_render_hides_cond(self, sandbox):
        cases_path, md = sandbox
        _seed(cases_path, [_case("s_001", "pair", **PAIR)])
        sa._render_md(sa._cases(sa._load()))
        txt = md.read_text(encoding="utf-8")
        assert "cond" not in txt and "with" not in txt and "without" not in txt


class TestDamagedAnchor:
    """锚点被破坏时，要**说文件坏了**，别说"你没标"。"""

    def test_damaged_anchor_reports_loudly(self, sandbox, capsys):
        cases_path, md = sandbox
        _seed(cases_path, [_case("s_001", "pair", **PAIR)])
        sa._render_md(sa._cases(sa._load()))
        # 复刻 `re.split` 捕获组的坑：外壳被吃掉、只剩 id
        txt = md.read_text(encoding="utf-8")
        md.write_text(txt.replace("<!-- s:s_001 -->", "s_001"), encoding="utf-8")

        assert sa.cmd_ingest() == 1                      # 非 0 → 脚本能失败
        out = capsys.readouterr().out
        assert "锚点被破坏" in out
        assert "一条都没收到" not in out                  # 不许假报成"你没标"

    def test_missing_id_is_reported_not_fatal(self, sandbox, capsys):
        cases_path, md = sandbox
        _seed(cases_path, [_case("s_001", "pair", **PAIR)])
        md.write_text("<!-- s:s_999 -->  [x] 甲  [ ] 乙  [ ] 差不多\n", encoding="utf-8")
        assert sa.cmd_ingest() == 0
        assert "找不到" in capsys.readouterr().out


class TestShuffleIsStable:
    """左右是打乱的，但**必须可复现** —— 同一内容两次出题不能翻边。"""

    @staticmethod
    def _mk(msg, t1, t2):
        return sa._mk_pair("【性格基底】+【语言风格】", "with_vs_without", msg, t1, "with", t2, "without")

    def test_same_input_same_sides(self):
        a = self._mk("在吗", "行吧行吧，那灰泽满先去睡了。", "嗯，知道了。")
        b = self._mk("在吗", "行吧行吧，那灰泽满先去睡了。", "嗯，知道了。")
        assert _pair_texts(a) == _pair_texts(b)
        assert a["src"] == b["src"]

    def test_both_orders_occur(self):
        """两种左右顺序都得出现，否则"打乱"是假的。"""
        sides = set()
        for i in range(40):
            c = self._mk(f"消息{i}", f"带文件的版本内容{i}啊", f"删掉版本{i}不同的话")
            if c:
                sides.add(c["left"]["cond"])
        assert sides == {"with", "without"}


class TestPoolHygiene:
    """用户 2026-09-30 报的四条数据质量问题，每条都得有测试守着。"""

    def test_near_duplicate_pairs_are_dropped(self):
        # 「很多语句都是重复的」：只差一个字的两版没有东西可判
        t = "晚上好晚上好，复活了复活了复活了。抱抱一下，至于吗你。"
        assert not sa._ok_pair(t, t.replace("你", "妳"))
        # 0.92 的旧阈值会放过这种（实测 13% 的题是这类）
        assert not sa._ok_pair("灰泽满只是一个16岁的风纪委员，你在说什么调不调试的",
                               "灰泽满只是一个16岁的风纪委员，你在说什么呢？")

    def test_name_asymmetry_is_flagged(self):
        # 「出现 hzm 类似名字的会让我更加有倾向」：只有一版自称 → 测不了风格
        assert not sa._name_sym("灰泽满先去睡了", "我先去睡了")
        assert not sa._name_sym("灰泽满先去睡了", "嗯，知道了。")
        assert sa._name_sym("嗯，知道了。", "行，那你早点睡")

    def test_pick_caps_repeated_text_and_question(self):
        """同一个问句最多出 2 道；同一句回复最多出现 2 次（实测全池 58% 是重复文本）。"""
        fresh = []
        for i in range(30):                      # 同一个 msg，30 个不同的小节
            c = sa._mk_pair(f"## 【段{i}】", "with_vs_without", "同一个问句",
                            "行吧行吧，那灰泽满先去睡了，你也别熬太晚。",
                            "base", f"删掉版本{i}的另一种说法", "ablated")
            if c:
                fresh.append(c)
        picked = sa._pick_mixed(fresh, 20, 0, 3, 12, {})
        cnt = Counter(c["msg"] for c in picked)
        assert max(cnt.values()) <= sa.PER_MSG_CAP

    def test_context_index_finds_preceding_turns(self, tmp_path, monkeypatch):
        log = tmp_path / "chat.jsonl"
        rows = [{"kind": "private", "session": "1", "user": "在吗", "reply": "在"},
                {"kind": "private", "session": "1", "user": "今天干啥了", "reply": "没干啥"}]
        log.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                       encoding="utf-8")
        monkeypatch.setattr(sa, "CHAT_LOG", log)
        monkeypatch.setattr(sa, "_CTX", None)
        ctx = sa._context_index()
        assert ctx["今天干啥了"] == [("在吗", "在")]
        assert ctx["在吗"] == []


class TestTwoAxisQuestion:
    """v2 的核心：**先分轴，再比方向**（用户："大多数语句在风格上是一致的，只不过内容上更具体"）。"""

    def test_render_shows_both_questions(self, sandbox):
        cases_path, md = sandbox
        _seed(cases_path, [_case("s_001", "pair", **PAIR)])
        sa._render_md(sa._cases(sa._load()))
        txt = md.read_text(encoding="utf-8")
        assert "只有内容不同" in txt and "只有说话方式不同" in txt
        assert "两样都不同" in txt and "基本没区别" in txt
        assert "[ ] 一样" in txt

    def test_diff_content_implies_tie(self, sandbox):
        """① 说「只有内容不同」= 说话方式一样 → ② 留空也按『一样』记（否则这题永远挂着）。"""
        cases_path, md = sandbox
        _seed(cases_path, [_case("s_001", "pair", **PAIR)])
        sa._render_md(sa._cases(sa._load()))
        md.write_text(md.read_text(encoding="utf-8").replace("[ ] 只有内容不同", "[x] 只有内容不同", 1),
                      encoding="utf-8")
        sa.cmd_ingest(quiet=True)
        c = {x["id"]: x for x in sa._cases(sa._load())}["s_001"]
        assert c["diff"] == "content" and c["label"] == "tie"

    def test_diff_style_then_direction(self, sandbox):
        cases_path, md = sandbox
        _seed(cases_path, [_case("s_001", "pair", **PAIR)])
        sa._render_md(sa._cases(sa._load()))
        md.write_text(md.read_text(encoding="utf-8")
                      .replace("[ ] 只有说话方式不同", "[x] 只有说话方式不同", 1)
                      .replace("[ ] 甲", "[x] 甲", 1), encoding="utf-8")
        sa.cmd_ingest(quiet=True)
        c = {x["id"]: x for x in sa._cases(sa._load())}["s_001"]
        assert c["diff"] == "style" and c["label"] == "left"

    def test_report_leads_with_resolution(self, sandbox, capsys):
        """**先报分辨力**：87% 答『差不多』时后面的占比没有意义（2026-09-30 第一批实测）。"""
        cases_path, md = sandbox
        _seed(cases_path, [_case(f"s_{i:03d}", "pair", label="tie", **PAIR)
                           for i in range(1, 8)])
        assert sa.cmd_report() == 0
        out = capsys.readouterr().out
        assert "仪器分辨力" in out
        assert "分不出" in out
        assert out.index("仪器分辨力") < out.index("方向")   # 必须排在方向前面

    def test_old_style_marks_still_parse(self, sandbox):
        """v1 的『差不多』不能因为 v2 改名成『一样』就作废——真丢过一次标注。"""
        cases_path, md = sandbox
        _seed(cases_path, [_case("s_001", "pair", **PAIR)])
        md.write_text("<!-- s:s_001 -->  [ ] 甲   [ ] 乙   [x] 差不多\n", encoding="utf-8")
        sa.cmd_ingest(quiet=True)
        assert {x["id"]: x for x in sa._cases(sa._load())}["s_001"]["label"] == "tie"


class TestGroupSets:
    """v3 整组题：**人判风格是看一批，不是看一对**（87% 答「差不多」换来的）。

    守住两个当场抓到的坑：
    ① 两组条数不等 → 你能数出来，那是白送线索；
    ② 两道验尺子题共用同一批"她的真话" → 连着看同样 8 句，还能从第一组认出第二组。
    """

    @staticmethod
    def _mk(n=8, key_side="left", variant=0):
        # ⚠️ variant 必须有：src 是内容哈希，内容一样 → 种子一样 → 左右永远同一侧
        left = [(f"甲侧第{variant}组第{i}句完全不同的内容", "with") for i in range(n)]
        right = [(f"乙侧第{variant}组第{i}句另外的说法", "without") for i in range(n)]
        return sa._mk_set("【X】", left, right, key_side=key_side, tag=f"t{variant}")

    def test_group_sizes_are_equal(self):
        c = sa._mk_set("【X】", [(f"甲{i}", "with") for i in range(8)],
                       [(f"乙{i}", "without") for i in range(5)], key_side="left", tag="t")
        assert len(c["left"]["texts"]) == len(c["right"]["texts"]) == 5

    def test_too_few_is_rejected(self):
        assert sa._mk_set("【X】", [("甲1", "w"), ("甲2", "w")],
                          [("乙1", "wo"), ("乙2", "wo")], key_side="left") is None

    def test_key_side_is_decoupled_from_jia_yi(self):
        """命中组必须**随机**落在甲或乙，否则你学会"甲就是带文件的"就完了。"""
        sides = {self._mk(key_side="left", variant=i)["key_side"] for i in range(30)}
        assert sides == {"left", "right"}

    def test_balance_sides_splits_hit_side_across_a_batch(self):
        """批内"命中组"不许全在同一侧——真踩到过：两道验尺子题都把她本人放乙组。"""
        items = [self._mk(key_side="right", variant=i) for i in range(4)]
        for c in items:
            c.update({"id": f"s_{len(items)}", "label": None, "why": ""})
        sa._balance_sides(items)
        sides = [c["key_side"] for c in sorted(items, key=lambda x: sa._h(x["src"]))]
        assert sides == ["left", "right", "left", "right"]

    def test_balance_sides_keeps_covariates_with_their_group(self):
        """翻转左右时协变量必须跟着走，否则读数里的"选中组均长"是错的。

        断言的是**不变量**：`cov[side]` 必须描述 `texts[side]`（自己重算一遍对照）。
        """
        items = []
        for i in range(4):
            left = [(f"灰泽满第{i}组的话，带自称", "with")] * 4
            right = [(f"嗯，第{i}组另一句", "without")] * 4
            c = sa._mk_set("【X】", left, right, key_side="right", tag=f"t{i}")
            c.update({"id": f"s_{i}", "label": None, "why": ""})
            items.append(c)
        sa._balance_sides(items)
        for c in items:
            for side in ("left", "right"):
                assert c["cov"][side] == sa._set_cov(c[side]["texts"])

    def test_covariates_are_recorded(self):
        left = [("灰泽满今天没睡好", "with")] * 4
        right = [("嗯。", "without")] * 4
        c = sa._mk_set("【X】", left, right, key_side="left", tag="t")
        assert c["cov"]["left"]["self"] == 4 and c["cov"]["right"]["self"] == 0

    def test_validate_sets_use_disjoint_her_replies(self, monkeypatch):
        monkeypatch.setattr(sa, "_pool_her_voice",
                            lambda limit=400: [f"她本人的第{i}句话内容" for i in range(40)])
        sets_ = sa._pool_validate_sets(per_group=8)
        her_groups = [set(c["left"]["texts"]) | set(c["right"]["texts"]) for c in sets_]
        # 两道题的"她的真话"不能重叠
        a = [t for t in her_groups[0] if t.startswith("她本人的")]
        b = [t for t in her_groups[1] if t.startswith("她本人的")]
        assert a and b and not (set(a) & set(b))

    def test_render_hides_conditions(self, sandbox):
        cases_path, md = sandbox
        c = self._mk()
        c.update({"id": "s_001", "created": "2026-09-30", "label": None, "why": ""})
        _seed(cases_path, [c])
        sa._render_md(sa._cases(sa._load()))
        txt = md.read_text(encoding="utf-8")
        assert "甲组" in txt and "乙组" in txt
        assert "with" not in txt and "without" not in txt and "cond" not in txt
        assert "key_side" not in txt

    def test_ingest_group_answer(self, sandbox):
        cases_path, md = sandbox
        c = self._mk()
        c.update({"id": "s_001", "created": "2026-09-30", "label": None, "why": ""})
        _seed(cases_path, [c])
        sa._render_md(sa._cases(sa._load()))
        md.write_text(md.read_text(encoding="utf-8").replace("[ ] 甲组", "[x] 甲组", 1),
                      encoding="utf-8")
        sa.cmd_ingest(quiet=True)
        assert {x["id"]: x for x in sa._cases(sa._load())}["s_001"]["label"] == "left"

    def test_report_prints_validation_verdict(self, sandbox, capsys):
        cases_path, md = sandbox
        c = self._mk()
        c.update({"id": "s_001", "created": "2026-09-30", "why": "",
                  "section": "★验尺子·硬对照（她本人 vs 客服腔）",
                  "label": "right", "key_side": "right"})
        _seed(cases_path, [c])
        assert sa.cmd_report() == 0
        out = capsys.readouterr().out
        assert "整组读数" in out and "选对" in out

    def test_report_flags_when_hard_control_fails(self, sandbox, capsys):
        """大差别对照都选错 → 必须明说"什么都别测了"，别让人拿它去测文件。"""
        cases_path, md = sandbox
        c = self._mk()
        c.update({"id": "s_001", "created": "2026-09-30", "why": "",
                  "section": "★验尺子·硬对照（她本人 vs 客服腔）",
                  "label": "left", "key_side": "right"})
        _seed(cases_path, [c])
        sa.cmd_report()
        out = capsys.readouterr().out
        assert "什么都别测了" in out

    def test_report_shows_covariates(self, sandbox, capsys):
        cases_path, md = sandbox
        c = self._mk()
        c.update({"id": "s_001", "created": "2026-09-30", "why": "", "section": "【X】",
                  "label": "left", "key_side": "left", "tag": "set1"})
        _seed(cases_path, [c])
        sa.cmd_report()
        out = capsys.readouterr().out
        assert "协变量" in out and "选中组均长" in out


class TestTextAnswers:
    """**不碰文件也能答**（用户 2026-09-30：文件改了没保存，标注白丢了）。"""

    def _setup(self, sandbox):
        cases_path, md = sandbox
        _seed(cases_path, [_case("s_001", "pair", **PAIR),
                           _case("s_002", "single", **SINGLE)])
        return sa._cases(sa._load())

    def test_parse_content_and_direction(self, sandbox):
        self._setup(sandbox)
        assert sa.cmd_answers("1 风格 甲\n2 像") == 0
        by_id = {x["id"]: x for x in sa._cases(sa._load())}
        assert by_id["s_001"]["diff"] == "style" and by_id["s_001"]["label"] == "left"
        assert by_id["s_002"]["label"] == "like"

    def test_parse_shorthand(self, sandbox):
        self._setup(sandbox)
        sa.cmd_answers("1 内容\n2 不像")
        by_id = {x["id"]: x for x in sa._cases(sa._load())}
        assert by_id["s_001"]["diff"] == "content" and by_id["s_001"]["label"] == "tie"
        assert by_id["s_002"]["label"] == "unlike"

    def test_garbage_lines_are_reported_not_swallowed(self, sandbox, capsys):
        self._setup(sandbox)
        sa.cmd_answers("1 风格 甲\n这不是一行答案")
        out = capsys.readouterr().out
        assert "收进 1 条" in out and "没认出来" in out

    def test_out_of_range_index_reported(self, sandbox, capsys):
        self._setup(sandbox)
        sa.cmd_answers("99 甲")
        assert "没认出来" in capsys.readouterr().out

    def test_id_form_is_immune_to_ordering(self, sandbox):
        """`s_013 乙` 按 id 定位，不受序号顺序影响。

        ⚠️ 为什么必须有这条：序号形式按"还没标的那些"的顺序算，
        一旦文件里有你没看到的题排在最前，**整个答案列表会错位贴到别的题上**，
        而且它会"成功"、不报错。本仓真踩过：13 条答案错位贴到了 12 道不相干的题上。
        """
        self._setup(sandbox)
        sa.cmd_answers("s_002 不像")
        assert {x["id"]: x for x in sa._cases(sa._load())}["s_002"]["label"] == "unlike"

    def test_echoes_every_applied_answer(self, sandbox, capsys):
        """逐条回声：错位时得能一眼看出来（句子和题号对不上）。"""
        self._setup(sandbox)
        sa.cmd_answers("1 风格 甲")
        out = capsys.readouterr().out
        assert "逐条核对" in out and "s_001" in out

    def test_unknown_id_reported_not_swallowed(self, sandbox, capsys):
        self._setup(sandbox)
        sa.cmd_answers("s_999 甲")
        assert "没认出来" in capsys.readouterr().out
