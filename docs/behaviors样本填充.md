# behaviors 样本填充 —— 从直播素材挖「处境」并填进 behaviors

> 建立于 2026-10-04。回答一个问题：**怎么从几十小时的直播转写里，挖出「她在一类人际处境下怎么应对」，并写成 behaviors 条目？**
>
> 相关：`情景条目设计模板.md`（条目怎么写才算好）、`注入设计原理.md`（为什么要这么设计）。

---

## 零、为什么需要这套流程（旧工具为什么不行）

旧的 `scripts/extract_persona.py` 吃的是**已经切好的单轮问答对**（`user_situation` + `reply`），
问题是：切分时同一句提问会被复制给她的多个碎语 —— 实测 `converted_0726.json` 前 3 条：

```
user_situation 全是「粉丝说：中午好」     ← 同一句复制 3 次
reply 分别是她一段话切出来的 3 个碎片
```

模型在 50 条这种对里找"稳定模式"，看到的全是"同一情境配不同回答" → **只能得出"没有稳定模式"**。

**成绩：877 对 → 2 条。**

根因三条：

| # | 问题 |
|---|---|
| 1 | **单轮对在结构上表达不了"情景"** —— 情景是连续几轮推进出来的（像"要语音"那条弧线：挡 → 给理由 → 给台阶 → 松口 → 兑现） |
| 2 | 硬性上限：`BATCH_SIZE=50` + 提示词"通常 2-5 条" + `merge_behaviors` 再合并一次 |
| 3 | `SCENE_GROUPS` 是旧清单（含已删的"冷场/被调戏"，缺新拆的条目）→ 归错类甚至被悄悄丢掉 |

---

## 一、四层工具链

```
outputs/clean/*.json（连续话轮，带 start/end）
   │
   ├─ ① scene_segment.py   切情景（在连续时间轴上标区间）
   │      产出 outputs/scene_segments/<场次>_scenes.json
   │
   ├─ ② scene_cluster.py   分批归并（45 条一批）
   │      产出 outputs/scene_clusters.json
   │
   └─ ③ scene_merge.py     层级归并 + 标「已覆盖 / 缺口 / 直播专属丢弃」
          产出 outputs/scene_final.json
```

一键跑全部场次：

```bash
cd D:\my_qq_bot\my_qq_bot
.\.venv\Scripts\python.exe outputs\_experiments\run_scene_pipeline.py
```

单场跑（调试用）：

```bash
# 只切分，不写文件
.\.venv\Scripts\python.exe outputs\_experiments\scene_segment.py outputs/clean/cleaned_百日.json --dry
# 切分 + 写文件
.\.venv\Scripts\python.exe outputs\_experiments\scene_segment.py outputs/clean/cleaned_百日.json
```

---

## 二、每层在干什么（以及为什么这么设计）

### ① 切情景 —— 三个关键判据

工具会要求模型对每个片段同时回答三件事：

**判据 1：这是「处境」还是「事件」？**

> **"把这件事换成别的话题，这套应对还成立吗？"**
> 成立 → 处境，标它。不成立（只因为那件具体事才成立）→ 事件，别标。

反例（事件，不要）：`被弹幕起哄怀孕`、`被追问孩子他爸`、`质疑周表真假`、`弹幕追问断播原因`
正例（处境，要）：`被反复索要同一样东西`、`被质疑说法真实性`、`被真心对待时别扭缩回`

**判据 2：能不能搬到一对一私聊？**

> **"把'弹幕/观众/投稿/直播'这些词从这段里去掉，处境还成立吗？"**

```
❌ 直播专属，丢弃：
   念投稿/点评二创/看粉丝视频 ｜ 游戏实况解说 ｜ 答谢打赏/欢迎舰长
   念弹幕/回应刷屏/被弹幕打断 ｜ 展示形象/换装/3D回 ｜ 直播设备操作 ｜ 下播催播/周表

✅ 私聊照样发生：
   被夸 / 被质疑 / 被催 / 被反复要东西 / 被追问私事 / 被索要亲密 / 被逼表态
   对方道歉 / 自我否定 / 情绪崩溃 / 骂她 / 造谣 / 真诚感谢她 / 倒苦水
```

**⚠️ 没有这个筛子时，实测 17 条缺口里 12 条是直播专属，白跑。**

**判据 3：模型只许标区间，不许写原文。**

所有原文由程序按话轮号回填。模型写的 `her_move`（她怎么应对）里**不许出现原句** ——
从源头杜绝"模型改写素材"，比事后用 `validate_samples` 校验可靠。

### ② 分批归并 —— 合并同处境

模型自己起的情景名会不一致（`被弹幕打断发言` / `被反复打断抢话`），
所以一批片段喂进去让它**合并同一处境**、并挑出纯事件标 `discard`。

### ③ 层级归并 —— 全局去重 + 标缺口

**为什么要"层级"**：实测 149 条一次喂进去，模型输出超过 `max_tokens` 被截断
（`Unterminated string starting at: line 659`）。所以分批并 → 再并结果 → 直到并不动。

**同时标三件事**：

| 标记 | 含义 |
|---|---|
| `covered: "现有条目的名字"` | 和现有 behaviors 是同一个处境 |
| `covered: ""` | **缺口**（这才是要找的） |
| `drop: true` | 直播专属，丢掉 |

---

## 三、实测框架（2026-10-04，11 场 / 3080 话轮）

```
11 场 → 329 个片段 → 149 条候选 → 层级归并 → 44 条
  其中：6 条已被现有 11 条覆盖 ｜ 38 条缺口 ｜ 2 条判为直播专属丢弃
```

**覆盖验证（说明现有 11 条主干是对的）**：

| 覆盖的处境 | 段数 | → 对应现有 |
|---|---|---|
| 被质疑说法真实性时辩解自证 | 44 | 被质疑戳穿时心虚辩解 |
| 被推着要求表态或做事时打太极 | 39 | 被推着要求时打太极 |
| 被夸赞时嘴硬否认或害羞 | 23 | 被夸时嘴硬否认 |
| 被真心对待时别扭缩回 | 20 | 被真心对待时别扭缩回 |

**高频处境全部已在 11 条里** —— 缺口都在中低频侧。

---

## 四、从候选到条目：三条硬规则

### 规则 1：段数 > 10 才值得入选

低频条目进 behaviors 有三个代价：检索用的 `samples[0..1]` 会稀释分类器、占注入预算、且难以验证。
实测头部与尾部的差距很大（37 段 vs 1 段），**尾部那批大多是个别用户的特殊行为**。

### 规则 2：先合并明显重复的候选

实测例子：

```
【对方讲自己的事时陪聊投入】23段  +  【对方讲事时陪聊接话】4段   → 同一条
【被编造故事起哄时半推半就】12段  +  【被起哄加码时顺势挡回】9段 → 同一条
```

### 规则 3：和现有条目划清边界

写新条目之前，先回答"它和现有哪条**最像**，区别在哪"。实测写过的三个区分：

| 新条目 | 最像的现有条目 | 区别 |
|---|---|---|
| 被追问私事细节时含糊带过 | 被质疑戳穿时心虚辩解 | 那条是**对方抓到她前后矛盾**（她做错了）→ 心虚认账；这条是**对方想套她私人信息**（她没做错）→ 含混带过 |
| 被索要亲密动作或称呼时推脱 | 被关系向越界时不当回事地略过 | 那条管**关系定性**（要当她男友、喊老公、说分手）；这条管**一个具体动作/称呼**（啵啵、摸摸头、叫宝宝）。⚠️ 两条有重叠，可考虑合并 |
| 对方陪她聊时顺着讲自己的事 | （无） | **现在完全没有东西管"她主动展开讲自己"**，只有 corpus（第三人称背景）和 voice_samples（问答对） |

---

## 五、⚠️ 已知的坑（都踩过）

### 坑 1：`samples.user` 与真实输入撞车 → 逐字复读

**这是最频繁踩的坑。** 样本若是"对某个问题的直接回答"，而被问到时输入与样本同话题，
模型会把样本当标准答案照搬。

实测：样本 `给我一个啵啵呢 → 啵啵没有，灰泽满的嘴留着吃饭的`，
输入 `满满，能摸摸我的头吗？` → **9 次生成 8 次逐字复读**。

**检查工具**（写完样本必跑）：

```bash
.\.venv\Scripts\python.exe outputs\_experiments\check_collision.py
```

它拿全部评测/回归用例的 query 比每条样本的 `user`，**只在"特有词"上比重合**
（去掉"灰泽满/是不是/直播"这类到处都有的词，否则误报一堆），重合 ≥50% 就报出来。

**发现撞车改样本，不要改用例** —— 用例代表"用户真会说的话"，样本才是可换的。

### 坑 2：`samples` 前 2 条兼任分类器路标

`routing.py` 里的 L3 分类器**只取每条 behavior 的 `samples[:2].user`**（不取 reply）当参照。
所以**前 2 条必须是"用户最可能怎么说出这个处境"的真实话**。

实测：把前两条写成"嗯。""哦。" → L3 判空、用例失败；换成真实说法立刻恢复。

### 坑 3：括号概括式 `user` 字段可用，但不解决撞车

实测三种写法（真实提问 / 括号概括 / 陈述式情境）**回复质量相当**，模型不会把括号当怪东西。
但**照抄与否取决于撞不撞车，与写法无关**。

### 坑 4：层级归并要防死循环

`scene_merge.py` 的停止条件是"这一轮没并动"，不是"达到某个条数"。
实测 149 → 107 → 87 → 78 → 65 → 62 → 49 → 47 → 46 才停。

### 坑 5：`段数` 的绝对值不可信

归并时脚本没保留"这处境来自哪几场"，所以 `total_segments` 大多是**单个批次内**的计数。
**排序方向可用，绝对值偏低。** 想用绝对频率做入选判断时要注意这点。

---

## 六、完整操作流程（下次直接用）

```
1. 转写 + 清洗（没有现成清洗稿时）
   # 转写（large-v3 已就位，模型缓存在 D:/my_ai_models/faster-whisper）
   D:\Huizeman-AI-Voice-Cloning\GPT-SoVITS_V4\GPT-SoVITS_V4_250424\env\python.exe \
     scripts/run_tool.py transcribe "assets\audio\xxx.m4a" --model large-v3 --out-dir outputs\transcribe_lv3
   # 清洗
   .\.venv\Scripts\python.exe scripts\run_tool.py clean-transcript -i <转写.json> -o outputs\clean\cleaned_xxx.json

2. 跑情景流水线
   .\.venv\Scripts\python.exe outputs\_experiments\run_scene_pipeline.py

3. 看结果
   outputs/scene_final.json  ← gaps（缺口）/ covered（已覆盖）/ dropped（直播专属）

4. 人工筛选：段数 > 10 的先看，合并明显重复的

5. 逐条写条目（按 情景条目设计模板.md）+ 挑 samples（话题必须互不相同）

6. 写完必跑三件套
   .\.venv\Scripts\python.exe outputs\_experiments\check_collision.py     # 撞车
   .\.venv\Scripts\python.exe -m pytest -q                                # 单测（812）
   .\.venv\Scripts\python.exe scripts\run_tool.py retrieval-eval          # 检索评测
   .\.venv\Scripts\python.exe scripts\run_tool.py regression --check      # 工程回归
```

---

## 七、工具清单

| 文件 | 作用 |
|---|---|
| `outputs/_experiments/scene_segment.py` | ① 切情景（三条判据写死在提示词里） |
| `outputs/_experiments/scene_cluster.py` | ② 分批归并 |
| `outputs/_experiments/scene_merge.py` | ③ 层级归并 + 标覆盖/缺口/丢弃 |
| `outputs/_experiments/run_scene_pipeline.py` | 一键跑全部场次 |
| `outputs/_experiments/scene_recount.py` | （实验性）数回原始片段，**结论：不靠谱，别依赖** |
| `outputs/_experiments/check_collision.py` | 样本撞车检查（写完样本必跑） |
