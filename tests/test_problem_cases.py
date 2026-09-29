# -*- coding: utf-8 -*-
"""问题驱动入口（`scripts/problem_cases.py`）的测试。

为什么值得测：它把"**看到的问题**"变成"**回归用例**"，是标注集的入口。
入口坏掉的后果是**静默的**——解析器认不出你写的字段，用例就没生成，
而你看到的是"跑完了，没报错"。本仓已经吃过一次同类的亏：
`retrieval_eval.py` 导入即崩、坏了很久没人知道。

所以这里测三件事：
1. **解析要容错但不能装懂** —— 人随手换行要能接住；缺字段要留 TODO，别编。
2. **id 必须可复现** —— 曾用内置 `hash()`（每进程加随机盐），两次跑 id 不同。
3. **两个文件之间的"约定"不能各说各话** —— `spot_check.py` 写出去的标记串
   与 `problem_cases.py` 回读的正则必须对得上（历史上这里就漂过：
   表头写 `[ ] 有问题：`、每条写 `[ ] 有：`，两套标记）。

不联网、不调模型：只测纯函数与文件读写。
"""
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import problem_cases as PC  # noqa: E402


# ==================== 解析：问题记录.md ====================

class TestParseNotes:
    def test_基本三字段(self):
        ps = PC.parse_notes("用户说：你多高啊\n她答：一米七吧\n应该是：该想起身高那条\n")
        assert len(ps) == 1
        assert ps[0].user == "你多高啊"
        assert ps[0].reply == "一米七吧"
        assert ps[0].should == "该想起身高那条"

    def test_多条以用户说为界(self):
        ps = PC.parse_notes(
            "用户说：第一条\n她答：a\n\n用户说：第二条\n她答：b\n")
        assert [p.user for p in ps] == ["第一条", "第二条"]

    def test_用户说之前的说明文字被忽略(self):
        ps = PC.parse_notes("# 标题\n> 说明\n随便一句话\n用户说：真正的问题\n")
        assert len(ps) == 1 and ps[0].user == "真正的问题"

    def test_缺用户说的块被忽略(self):
        assert PC.parse_notes("她答：没有用户那句\n应该是：x\n") == []

    def test_禁词支持多种分隔符(self):
        p = PC.parse_notes("用户说：q\n禁词：搬家,搬东西、房东 中介\n")[0]
        assert p.forbid == ["搬家", "搬东西", "房东", "中介"]

    def test_跑次取数字_缺省3(self):
        assert PC.parse_notes("用户说：q\n跑次：7\n")[0].runs == 7
        assert PC.parse_notes("用户说：q\n")[0].runs == 3

    def test_该中不该中按行累积(self):
        p = PC.parse_notes("用户说：q\n该中：preference:routine\n该中：corpus:*\n不该中：phrase,behavior\n")[0]
        assert p.wants == ["preference:routine", "corpus:*"]
        assert p.nots == ["phrase", "behavior"]

    def test_续行接到上一个已有值的字段(self):
        # 人随手换行：第二行没有「字段：」前缀，应该接在「应该是」后面，而不是丢掉
        p = PC.parse_notes("用户说：q\n应该是：不该拿旧事\n当成现在的理由\n")[0]
        assert p.should == "不该拿旧事 当成现在的理由"

    def test_层和备注(self):
        p = PC.parse_notes("用户说：q\n层：生成\n备注：随口说的\n")[0]
        assert p.layer == "生成" and p.note == "随口说的"

    def test_代码围栏里的示例不被当成真问题(self):
        # 模板里的"示例"都写在围栏里 —— 不加这条会把示例当真的问题处理（自己踩过）
        ps = PC.parse_notes(
            "## 示例\n\n```\n用户说：这是示例\n她答：x\n```\n\n"
            "## 我记的\n\n用户说：这才是真问题\n")
        assert [p.user for p in ps] == ["这才是真问题"]

    def test_空文本不炸(self):
        assert PC.parse_notes("") == []
        assert PC.parse_notes(None) == []


# ==================== 解析：抽查勾选 ====================

SPOT_SAMPLE = """# 抽查清单（自动挑出来的）

> 说明行

---

### 1. 「你多高啊」
一米七吧

`可疑信号：问的是她`   `session=123 kind=private t=1790322968.877`

**重点看**：B1 知识准确

**这条有没有明显不对？** [ ] 没有  [x] 有：她明明1米6，编了

### 2. 「今天天气不错」
是呀

`可疑信号：带括号`   `session=456 kind=group t=1790322999.5`

**重点看**：B4 说话风格

**这条有没有明显不对？** [ ] 没有  [ ] 有：__________
"""


class TestParseSpotMarks:
    def test_只收勾了有的(self):
        ps = PC.parse_spot_marks(SPOT_SAMPLE)
        assert len(ps) == 1
        assert ps[0].spot_why == "她明明1米6，编了"

    def test_抽出会话与时间戳当唯一键(self):
        p = PC.parse_spot_marks(SPOT_SAMPLE)[0]
        assert p.session == "123" and p.t == 1790322968.877

    def test_没有勾选时返回空(self):
        no_marks = SPOT_SAMPLE.replace("[x] 有：她明明1米6，编了", "[ ] 有：__________")
        assert PC.parse_spot_marks(no_marks) == []

    def test_回查不到时退回markdown里的文本(self, monkeypatch):
        # 日志轮转过 → lookup_turn 取不到 → 不能把条目整条丢掉
        monkeypatch.setattr(PC, "lookup_turn", lambda *_a: None)
        p = PC.parse_spot_marks(SPOT_SAMPLE)[0]
        assert p.user == "你多高啊"

    def test_下划线占位不算备注(self):
        marks = SPOT_SAMPLE.replace("她明明1米6，编了", "__________")
        assert PC.parse_spot_marks(marks)[0].spot_why == ""


class TestLookupTurn:
    def test_按会话加时间戳取回原文(self, tmp_path, monkeypatch):
        log = tmp_path / "chat.jsonl"
        log.write_text(json.dumps({"t": 1.5, "session": "9", "kind": "private",
                                   "user": "原话", "reply": "回复"}, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        monkeypatch.setattr(PC, "CHAT_LOG", log)
        assert PC.lookup_turn("9", 1.5)["user"] == "原话"

    def test_会话对但时间戳不对_取不到(self, tmp_path, monkeypatch):
        log = tmp_path / "chat.jsonl"
        log.write_text(json.dumps({"t": 1.5, "session": "9"}, ensure_ascii=False) + "\n", encoding="utf-8")
        monkeypatch.setattr(PC, "CHAT_LOG", log)
        assert PC.lookup_turn("9", 2.5) is None

    def test_没有日志文件时取不到而不是崩(self, tmp_path, monkeypatch):
        monkeypatch.setattr(PC, "CHAT_LOG", tmp_path / "nope.jsonl")
        assert PC.lookup_turn("1", 1.0) is None


# ==================== 期望写法 → 判据 ====================

class TestExpectOf:
    def test_星号是should_fire(self):
        assert PC._expect_of("corpus:*") == ("corpus", {"should_fire": True})

    def test_只给路名也是should_fire(self):
        assert PC._expect_of("corpus") == ("corpus", {"should_fire": True})

    def test_corpus只能写关键词_走contains(self):
        # corpus 的 item_id 是向量库下标，改语料就重排 —— 不能钉，只能 contains
        route, cond = PC._expect_of("corpus:1米6")
        assert route == "corpus" and cond == {"contains": ["1米6"]}

    @pytest.mark.parametrize("route,spec", [
        ("preference", "routine"), ("phrase", "brag_deny"),
        ("behavior", "被夸时嘴硬否认"), ("voice_sample", "daily_short_1"),
    ])
    def test_id稳定的路走should(self, route, spec):
        assert PC._expect_of(f"{route}:{spec}") == (route, {"should": [spec]})


# ==================== 草案 ====================

class TestDraft:
    def test_检索层草案把该中不该中都翻成expect(self):
        p = PC.Problem(user="q", should="该想起作息", wants=["preference:routine"], nots=["corpus"])
        case = PC.draft_retrieval(p)
        assert case["expect"]["preference"] == {"should": ["routine"]}
        assert case["expect"]["corpus"] == {"should_not_hit": True}
        assert case["query"] == "q" and case["why"] == "该想起作息"
        assert case["source"] == "问题记录"

    def test_没写该中不该中时留占位而不是编(self):
        # 关键：不替人拍板。占位必须显眼，写完要人改。
        case = PC.draft_retrieval(PC.Problem(user="q", should="说不清"))
        assert case["expect"] == {"corpus": {"should_fire": True}}

    def test_同一路既该中又不该中_保留该中并告警(self, capsys):
        # 拼出一个两头堵的用例 = 评测永久红，所以只留更具体的那个，并喊一声
        case = PC.draft_retrieval(PC.Problem(user="q", wants=["corpus:女室友"], nots=["corpus"]))
        assert case["expect"] == {"corpus": {"contains": ["女室友"]}}
        assert "既写了该中又写了不该中" in capsys.readouterr().out

    def test_该中不该中在不同路时不冲突(self):
        case = PC.draft_retrieval(PC.Problem(user="q", wants=["preference:routine"], nots=["corpus"]))
        assert case["expect"]["preference"] == {"should": ["routine"]}
        assert case["expect"]["corpus"] == {"should_not_hit": True}

    def test_生成的草案一定不自相矛盾(self):
        # 数据文件那边也有同样的测试（TestMergedCasesFile）——这里保入口不产坏数据
        for wants, nots in ((["corpus:x"], ["corpus"]), (["preference:a"], []), ([], ["phrase"])):
            case = PC.draft_retrieval(PC.Problem(user="q", wants=wants, nots=nots))
            for route, cond in case["expect"].items():
                assert not ("should_not_hit" in cond and
                            ("should" in cond or "contains" in cond or cond.get("should_fire"))), \
                    f"{route} 自相矛盾：{cond}"

    def test_生成层草案用禁词当判据(self):
        case = PC.draft_generation(PC.Problem(user="q", should="别拿旧事当理由",
                                              forbid=["搬家"], reply="搬家没弄完"))
        assert case["forbid"] == ["搬家"] and case["runs"] == 3
        assert "搬家没弄完" in case["source"]
        assert "_TODO" not in case

    def test_生成层缺判据时显式标记TODO(self):
        case = PC.draft_generation(PC.Problem(user="q", should="答得不对"))
        assert "_TODO" in case

    def test_抽查来源会记进source(self):
        case = PC.draft_retrieval(PC.Problem(user="q", spot_why="编了", should="该想起X",
                                             wants=["corpus:*"]))
        assert case["source"] == "抽查"
        assert "编了" in case["why"]


class TestGuessLayer:
    def test_写明的层优先(self):
        assert PC.guess_layer(PC.Problem(user="q", layer="检索", forbid=["x"])) == "检索"

    def test_有禁词就是生成层(self):
        assert PC.guess_layer(PC.Problem(user="q", forbid=["搬家"])) == "生成"

    def test_明说别拿旧事当理由是生成层(self):
        assert PC.guess_layer(PC.Problem(user="q", should="别拿旧事当现在的理由")) == "生成"

    def test_默认检索层(self):
        assert PC.guess_layer(PC.Problem(user="q", should="该想起身高那条")) == "检索"


# ==================== id 可复现 ====================

class TestSlug:
    def test_同一个问题两次跑得到同一个id(self):
        assert PC._slug("你多高啊") == PC._slug("你多高啊")

    def test_不同问题id不同(self):
        assert PC._slug("你多高啊") != PC._slug("你几岁啊")

    def test_纯中文也能出id(self):
        assert PC._slug("你多高啊").strip("_0123456789")


# ==================== 写文件 ====================

class TestWriteCases:
    def _doc(self):
        return {"_readme": "说明", "_change_log": ["改动一", "改动二"],
                "cases": [{"id": "a", "query": "q1", "expect": {"corpus": {"should_fire": True}}}]}

    def test_紧凑写回后内容不变(self, tmp_path):
        f = tmp_path / "c.json"
        f.write_text("{}", encoding="utf-8")
        PC._write_cases_json(f, self._doc(), compact_cases=True)
        assert json.loads(f.read_text(encoding="utf-8")) == self._doc()

    def test_紧凑写回时每条case在一行(self, tmp_path):
        f = tmp_path / "c.json"
        PC._write_cases_json(f, self._doc(), compact_cases=True)
        lines = [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip().startswith('{"id"')]
        assert len(lines) == 1

    def test_非紧凑写回也是合法json(self, tmp_path):
        f = tmp_path / "c.json"
        PC._write_cases_json(f, self._doc(), compact_cases=False)
        assert json.loads(f.read_text(encoding="utf-8")) == self._doc()


class TestAddCases:
    def _file(self, tmp_path, cases):
        f = tmp_path / "c.json"
        f.write_text(json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8")
        return f

    def test_追加新用例(self, tmp_path):
        f = self._file(tmp_path, [{"id": "old", "query": "q0"}])
        added, skipped = PC.add_cases(f, [{"id": "new", "query": "q1"}], compact_cases=True)
        assert [c["id"] for c in added] == ["new"] and skipped == []
        assert [c["id"] for c in json.loads(f.read_text(encoding="utf-8"))["cases"]] == ["old", "new"]

    def test_同样的query不重复写(self, tmp_path):
        # 用例以"那句话"为身份 —— 同一句话换个 id 也是重复
        f = self._file(tmp_path, [{"id": "old", "query": "同一句话"}])
        added, skipped = PC.add_cases(f, [{"id": "new", "query": "同一句话"}], compact_cases=True)
        assert added == [] and len(skipped) == 1

    def test_同id不覆盖(self, tmp_path):
        f = self._file(tmp_path, [{"id": "dup", "query": "q0", "why": "原有的"}])
        PC.add_cases(f, [{"id": "dup", "query": "q1", "why": "新的"}], compact_cases=True)
        cases = json.loads(f.read_text(encoding="utf-8"))["cases"]
        assert len(cases) == 1 and cases[0]["why"] == "原有的"

    def test_全是重复时不重写文件(self, tmp_path):
        f = self._file(tmp_path, [{"id": "old", "query": "q0"}])
        before = f.read_text(encoding="utf-8")
        PC.add_cases(f, [{"id": "old", "query": "q0"}], compact_cases=True)
        assert f.read_text(encoding="utf-8") == before


# ==================== 跨文件约定（历史上漂过） ====================

class TestMarkerContract:
    def test_抽查写出去的标记能被问题入口读回来(self):
        import spot_check
        line = spot_check.MARK.replace("[ ] 有：", "[x] 有：她编了")
        marked = ("### 1. 「q」\nanswer\n\n`session=1 kind=private t=2.5`\n\n" + line + "\n")
        assert PC.parse_spot_marks(marked)[0].spot_why == "她编了"

    def test_没有勾的行不会被读成有问题(self):
        import spot_check
        marked = ("### 1. 「q」\nanswer\n\n`session=1 kind=private t=2.5`\n\n" + spot_check.MARK + "\n")
        assert PC.parse_spot_marks(marked) == []

    def test_两个文件的勾选判断一致(self):
        # spot_check 用 FLAG_RE 判"勾了"，problem_cases 用同样的形状 —— 别各写一套
        import spot_check
        assert spot_check.FLAG_RE.search("[x] 有：x")
        assert spot_check.FLAG_RE.search("[X] 有：x")
        assert not spot_check.FLAG_RE.search("[ ] 有：__________")


# ==================== 数据文件本身 ====================

class TestMergedCasesFile:
    """合并后的标注集是**唯一真值**，它自己必须始终是合法的。"""

    @pytest.fixture
    def doc(self):
        return json.loads((SCRIPTS / "retrieval_eval_cases.json").read_text(encoding="utf-8"))

    def test_能加载且有条目(self, doc):
        assert doc["cases"]

    def test_id不重复(self, doc):
        ids = [c["id"] for c in doc["cases"]]
        assert len(ids) == len(set(ids))

    def test_query不重复(self, doc):
        # 同一句话的期望只在一个地方 —— 否则又回到"两条各钉一路"的老问题
        qs = [c["query"] for c in doc["cases"]]
        assert len(qs) == len(set(qs))

    def test_每条都有来源和理由(self, doc):
        for c in doc["cases"]:
            assert c.get("source"), f"{c['id']} 缺 source"
            assert c.get("why"), f"{c['id']} 缺 why"

    def test_期望写法只有这四种(self, doc):
        for c in doc["cases"]:
            assert c["expect"], f"{c['id']} 没有任何期望"
            for route, cond in c["expect"].items():
                assert set(cond) <= {"should", "should_not_hit", "contains", "should_fire"}, \
                    f"{c['id']}/{route} 写法不认识：{cond}"

    def test_同一路不能说既该中又不该中(self, doc):
        for c in doc["cases"]:
            for route, cond in c["expect"].items():
                assert not ("should_not_hit" in cond and
                            ("should" in cond or "contains" in cond or cond.get("should_fire"))), \
                    f"{c['id']}/{route} 期望自相矛盾"

    def test_旧标注集已并入且文件已删(self):
        assert not (SCRIPTS / "judge_labels.json").exists()
        assert "judge_labels" in json.dumps(
            json.loads((SCRIPTS / "retrieval_eval_cases.json").read_text(encoding="utf-8")),
            ensure_ascii=False)      # 合并记录要留在 _change_log 里

    def test_负例泛化到了多路(self, doc):
        # 钉住合并的核心收益：跨路负例不能退回"只钉 corpus"（那正是旧版测不出
        # "某一路乱开火"的原因：phrase 100% 开火时评测照样全绿）。
        by_id = {c["id"]: c for c in doc["cases"]}
        for cid in ("co3_neg_time", "co4_neg_weather", "co5_neg_lan", "co7_neg_meeting", "co8_neg_cold"):
            routes = by_id[cid]["expect"]
            assert "corpus" in routes, f"{cid} 丢了 corpus 期望"
            assert len(routes) >= 2, f"{cid} 又退回只钉一路了：{list(routes)}"
            assert "preference" in routes, f"{cid} 的泛化被退化掉了：{list(routes)}"

    def test_跨路负例不误钉behavior和phrase(self, doc):
        # 2026-09-29 实测踩过：行为层判对（被质疑/被越界）时，phrase 里那一组
        # 正是跟着该行为走的措辞，把它钉成"不该命中"会误杀（co5/n2 各错一次）。
        by_id = {c["id"]: c for c in doc["cases"]}
        for cid in ("co5_neg_lan", "n2_kiss"):
            routes = by_id[cid]["expect"]
            assert "behavior" not in routes and "phrase" not in routes, \
                f"{cid} 又把 behavior/phrase 钉成负例了：{list(routes)}"

    def test_天气那条仍然钉住behavior(self, doc):
        # 反过来：纯闲聊（聊天气）确实不该套任何行为模式 —— 这条泛化是对的
        by_id = {c["id"]: c for c in doc["cases"]}
        assert "behavior" in by_id["co4_neg_weather"]["expect"]


class TestJudgeExperimentDerivation:
    """判据实验的 want 现在从评测集推导 —— 推导规则必须有测试钉住。"""

    def test_推导逻辑(self, monkeypatch, tmp_path):
        f = tmp_path / "cases.json"
        f.write_text(json.dumps({"cases": [
            {"query": "正例", "expect": {"corpus": {"should": ["x"]}}},
            {"query": "含关键词也算正例", "expect": {"corpus": {"contains": ["k"]}}},
            {"query": "非空也算正例", "expect": {"corpus": {"should_fire": True}}},
            {"query": "负例", "expect": {"corpus": {"should_not_hit": True}}},
            {"query": "不参与", "expect": {"preference": {"should": ["food"]}}},
        ]}, ensure_ascii=False), encoding="utf-8")
        import judge_experiment as JE
        monkeypatch.setattr(JE, "CASES_FILE", f)
        tasks = JE.load_tasks()
        got = {c["q"]: c["want"] for c in tasks["corpus"]["cases"]}
        assert got == {"正例": True, "含关键词也算正例": True, "非空也算正例": True, "负例": False}
        assert "不参与" not in got
