# 文件地图（谁是什么 · 一看就懂）

> 给使用者/接手的开发者看。按"入口 → 运行时 → 人格数据 → 工具 → 测试 → 产物"组织。
> 版本史见 `docs/版本史.md`；**还没做的问题**见 `待办清单.md`。
> **要新增直播素材或更新 persona 数据**：看 `人格素材流水线.md`。
> **想把整条回复链从 QQ 收到消息一路看到最终 messages**：先看 `注入设计·人话版.md`；
> **想看"素材怎么被注进去"的全貌**（每层注入什么、多少字符、阈值多少）见 `注入链路.md`；
> **想知道"判据怎么选、素材该长什么形式、改了什么会变好/变坏"**（带实验数据）见 `注入设计原理.md`。

## 零、先看这个：根目录长什么样

**「这条东西我要不要管」的速查**。`入库` = 跟着 git 走；`不入库` = 只在你本机。

| 根目录条目 | 是什么 | 入库 |
|---|---|---|
| `bot.py` | 启动入口（双击 `启动.bat` 就是跑它） | ✅ |
| `memory_manager.py` | **长期记忆**实现（用户画像/承诺/印象卡的读写 + LLM 提取） | ✅ |
| `启动.bat` / `更新周表.bat` | 一键开两个窗口 / 拖一张周表图就识图更新 | ✅ |
| `src/` | **跑起来要用的代码**（`src/plugins/chatbot/` 是核心，见第二、三节） | ✅ |
| `persona/` | **她是谁/懂什么/怎么说**——所有素材（见第三节，**你会改的主要是这里**） | ✅ |
| `scripts/` | **离线工具**：蒸馏素材、跑评测、做实验（56 个，见第四节 + `脚本清单.md`） | ✅ |
| `tests/` | 单元测试（800 个，`pytest -q`） | ✅ |
| `docs/` | 全部说明文档，索引在 `docs/README.md` | ✅ |
| `video/` | **Remotion 注入链路说明视频**：场景、旁白时间轴、字幕和渲染脚本；成片在 `video/out/` | ✅ |
| `assets/` | 素材文件：`audio/` 原始音频、`transcripts/` 转写、`stickers/` 表情包、`voice_refs/` 语音参考、`emotes/` `img/` | ⚠️ 部分（`audio/` 不入库，太大） |
| `bin/` | ffmpeg 三件套（音频处理用，**不是我们写的**） | ✅ |
| `materials/` | 用户收集的原始素材文本（如 `灰泽满动态合集.txt`） | ✅ |
| `outputs/` | **所有工具的产物**（第六节，**可以整个删掉重跑**） | ❌（已忽略） |
| `data/` | **机器人运行时的状态**：聊天落盘、B站/微博状态、语音缓存、心跳 | ❌（已忽略） |
| `user_memory/` | **记忆文件**：短期/长期/会话/群（清空 = 换新开聊） | ❌（已忽略） |
| `logs/` | watchdog 的运行日志 | ❌（`*.log` 已忽略） |
| `env/` | GPT-SoVITS 的 Python 环境（**不是本项目的代码**） | ❌ |
| `.env.prod` | API key / cookie 等**机密** | ❌（已忽略） |
| `记录.txt` / `问题记录.md` / `风格标注.md` | 你手上的三个工作文件：真实翻车对话 / 看到的问题记一笔（`problem_cases.py`）/ 风格标注题（`style_annotate.py`）。**后两个含真实用户发言，绝不入库** | ❌（故意的） |
| 根目录 20 个杂物：`api.xml` `astra.*` `cr*.json` `oa*.json` `h1.html` `h2.html` `page.html` `bing2.html` `smc.txt` `up2.json` `j1.json` `w1.json` `*.png` | **跟本项目无关**（以前放这儿的），**可以直接删** | 未跟踪 |

> ✅ **怎么自查仓库干不干净**：跑 `git status --short`。**正常的输出只有两类**——
> ① 你/我新加的文件；② 上面那 20 个杂物。**不该有别的**：
> `outputs/` `data/` `user_memory/` `logs/` `.env.prod` 都已经在 `.gitignore` 里
> （其中 `user_memory` 那条写得挺细：连 `_backup_*/` 里的真实聊天记忆和 `.bak-*` 后缀都单独收掉了
> ——因为它是**你和他人的私聊记录**，绝不该进仓库）。

## 一、入口（怎么跑起来）

| 文件 | 作用 |
|---|---|
| `bot.py` | 启动入口：初始化 NoneBot、注册 OneBotV11、加载 `src/plugins` 插件 |
| `memory_manager.py` | **长期记忆**实现（user_memory/long_term.json）：用户画像/承诺/印象卡读写与 LLM 提取 |
| `启动.bat` | 一键开两个窗口：GPT-SoVITS api + bot（NapCat 自己开） |
| `更新周表.bat` | **把周表图拖上去**即识图并写入 `persona/world/schedule.json`（只覆盖 weekly）。双击=处理 `data/schedule_inbox/` 里最新一张。改完要重启 bot |
| `记录.txt` | 用户手放的**真实翻车对话**（不入库），供分析用 |

## 二、运行时核心（`src/plugins/chatbot/`）

| 文件 | 作用 | 关键点 |
|---|---|---|
| `__init__.py` | 事件入口 | 收消息→进读秒窗口；心跳/在线信号(data/heartbeat\|qq_alive\|qq_offline)；自动通过好友 |
| `core.py` | 主循环 | 分层提示注入、两路 RRF + 两路直达、生成、记忆提取、防复读、梗/行为路由落地。**`gather_retrieval()` = 取素材的唯一入口**（聊天与主动发言共用；内含"检索层炸了按没检索到继续"的兜底）|
| `chat_window.py` | 读秒攒批窗口 | 群=整群一窗；回复前读图+归纳；**群聊先过接话门**（私聊不过）；语音优先；`split_reply` 分批发（打字感） |
| `group_gate.py` | **群聊接话门** | 攒批安静下来时判"这批要不要开口"（判据=用户标定：点名/情绪/经历才接，纯事务附和收尾不接，**纯事件按概率**）。**点名（手打名字 / @她）确定性放行、不调 LLM**。失败按"接"放行。`GROUP_GATE=0` 可关 |
| `corpus_judge.py` | **corpus 语义判** | **只判不筛**：候选由 `retrieval.py` 给（`retrieve_corpus_candidates`），它只回答"这段经历和用户刚说的话是不是一回事"（填平"口语问句 vs 第三人称陈述"的鸿沟——「你多高啊」靠余弦只有 0.443，永远进不来）。失败**不带经历**（与接话门相反：宁可漏不可错）。⚠️ 2026-09-29 起**它成了 corpus 唯一的判据**（原先"门放行的直通"已取消）。`CORPUS_JUDGE=0` = **整路关掉** |
| `proactive.py` | **主动发言** | 抓到动态/微博 → 转成「她会主动说的那句话」。**素材走主链路**（`core.gather_retrieval` + `build_message_list`，与聊天共用同一份人格数据；`is_user_msg=False` 跳过"用户说的话"才成立的两层：L3 行为/措辞、corpus 判）。**开口前先过一道判**：内容缺关键信息、单独看不懂（"写不完了"）→ 这句不发；她指了配图（⬇️）才去读图。**原文通知照发**，返回 None 只是少她那句（两个桥都是「原文 + 她那句」两条）。`PROACTIVE=0` 全关 / `PROACTIVE_JUDGE=0` 只关那道判 |
| `chatlog.py` | **聊天落盘** | 每轮追加 JSONL（`data/chat_log/`，**gitignore**）供事后分析——短时记忆只有 10 条滚动窗口，超出就没了。`CHATLOG=0` 可关 |
| `stickers.py` | **表情包** | 判「她这句话配哪张表情」，十分对应才发（失败**不发**，与接话门相反）。标注在 `persona/media/stickers.json`（31 张 / 18 类），图在 `assets/stickers/`。**去重按 `group`（类别）不按单张**——同一类里好几张（委屈 3、无语无奈 4），按单张挡不住「换个还是同味道」；窗口 = 最近 3 类。`STICKER=0` 可关 |
| `reply_style.py` | 纯函数后处理 | 拆句/分批延迟/`clean_reply`（换行归一、去括号、省略号纪律、自指兜底）/防复读检测 |
| `retrieval.py` | 素材获取与融合 | 两路 RRF（corpus/behavior）+ preference 关键词子串、core_story 命中直达。另有预算控制。**判据怎么选见文件头**（三种任务三种工具，别再给 preference 上向量）。`voice_sample` 那路已于 2026-10-04 停用（见函数头） |
| `routing.py` | 硬路由 | legendary 梗库（含 LLM 语境确认）+ 行为意图分类 L3（**同时判行为 + 措辞组 + 连续索要**） |
| `persona.py` | 人格加载 | 骨架、behaviors、terms、schedule 的读取与拼装（traits/styles 已拆，函数保留返回空表） |
| `rag.py` | 向量工具 | embedding 客户端封装（`embed_query` 等），检索层的底座 |
| `qq_faces.py` | QQ 表情表 | face id → 中文名（248 条）。NapCat 拿不到 `faceText` 时兜底，免得消息变成"QQ表情6"。**别手改**，来源与提取规则见文件头 |
| `memory.py` | 短期记忆 | short_term.json 带锁读写；并加载根 memory_manager 供长期 |
| `session_memory.py` | 会话记忆 | session.json（话题/事件/指代补全）。**隔 >12h 的上一场不当"当前会话"**，降级成「上次聊过（3天前）…」单独注入——`previous_session_note()` **必须在 probe 之前调**（probe 会重写 last_active） |
| `group_memory.py` | 群记忆 | groups.json：成员 id→昵称 + 群近况 events |
| `context_probe.py` | 感知 | 时间/农历/天气/直播状态 →【当前时间】（天气异步预热，不卡主循环） |
| `vision.py` | 看图 | glm-4.6v 把图片描述成文字注入 |
| `bili_bridge.py` | B站联动 | 开播/动态变化 → 私聊广播（含表情本地化/压小）。**转发别人的动态整条不推**（原文取不全） |
| `weibo_bridge.py` | 微博联动 | 新微博 → 私聊广播（cookie jar 会话续期）。**转发微博整条不推** |
| `voice.py` | 语音 | GPT-SoVITS 合成→QQ 语音（静音裁剪、达标重试、截断兜底） |
| `_bridge_common.py` | 公共小件 | bili/weibo 共用的状态读写 + 图片下载 |
| `config.py` | 配置 | NoneBot 配置读取 + DeepSeek/智谱客户端工厂 |
| `constants.py` | 常量 ★ | 所有路径/API/阈值集中，阈值旁都写了实验理由 |

## 三、人格数据（`persona/` —— 她是谁/懂什么/怎么说）

> ⚠️ **2026-10-04 重构**：`traits.json` / `styles.json` / `phrases.json` / `voice_samples.json`
> **四个文件已拆掉**（内容并进 `core/system_prompt.txt` 与 `behavior/behaviors.json`，
> 原因见 `注入链路.md` 第六节）。**现在只有两层**：
> **人格层（2 个文件）+ 事实层（其余）**。分类口径见 `注入链路.md` 第二节。

### 3.1 人格层（就这两个）

| 文件 | 装什么 | 注入方式 |
|---|---|---|
| `core/system_prompt.txt` | 骨架：身份框架/自我称呼/说话节奏/括号语义/篇幅纪律 | **每轮常驻**（3452 字，占输入 91%） |
| `behavior/behaviors.json` | **16 条情景 → 反应 + 真实原句 samples**（74 条样本） | L3 命中哪条注哪条 |
| `behavior/behavior_keywords.json` | 行为判别词（确定性兜底，跳 LLM） | retrieval |

### 3.2 事实层（其余全部）

| 文件 | 装什么 | 判定方式 |
|---|---|---|
| `world/terms.json` | 名词库 lorebook（人物/梗/黑话 + meaning/reaction/aliases/pattern） | 关键词/正则（`always` 的常驻） |
| `world/legendary.json` | 经典梗固定应答（含 LLM 语境确认） | 关键词/正则 → 硬路由 |
| `world/schedule.json` | 周表（根目录 `更新周表.bat` 拖图 OCR 更新） | **常驻**（只注 weekly） |
| `world/preferences.json` | 偏好档案 | 关键词子串（**不走向量**） |
| `world/core_stories.json(+_vectors)` | 印象最深的经历 | 向量 |
| `world/statement_final.json → corpus_vectors.json` | 直播记忆语料源(421) → 向量 | **向量初筛 + LLM 再审** |
| `world/corpus_keywords.json` | corpus 召回钩子：用户换个说法时决定谁能进候选池；**不直接注入 messages** | 候选门（关键词） |
| `world/corpus_asks.json` | 语料自指代词审计表 | 辅助 |
| `media/stickers.json` | 表情包标注（31 张 / 18 类） | 关键词/判定 |
| `speech/phrase_vectors.json` | ⚠️ **孤儿缓存**（源文件 `phrases.json` 已删）——可删，没有代码读它 |

**向量缓存**（`*_vectors.json`）跟源文件放一起，改了源要重跑对应
`precompute core-stories` / `generate-vectors`。**behaviors 不用重算**（不走向量）。

## 四、离线工具（`scripts/`）

统一入口 `python scripts/run_tool.py <工具>`。分组：

- **人格流水线**：`persona-pipeline corpus|behaviors`
  （可选从音频开始，`--whisper-python` 可指定 faster-whisper 环境；默认只产候选和 review，
  `--apply` 才写回并备份；corpus 自动补 `corpus_keywords` 钩子、审计自指代词并重建向量；
  `--sanitize-existing --from-index N` 可定向清理存量语料）
  > ⚠️ `voice-samples` / `phrases` 两个子命令**已于 2026-10-04 下线**（对应文件已拆）。
- **蒸馏**：transcribe → clean-transcript → analyze-pace → convert-to-chat
- **生成**：extract-persona、mine-phrases、mine-theme；`generate-statements` 作为 corpus 流水线内部步骤
- **向量**：`generate-vectors -i persona/world/statement_final.json`
  （`--only-index 335,410,420` 只重算指定索引）；`precompute core-stories`
  （⚠️ `precompute phrases|preferences` 与产出的 `*_vectors.json` **已停用**——2026-09-26 这两路改判据，不再走向量）
- **评测**：regression / persona-eval / retrieval-eval / **problems**（★问题驱动：看到的问题 → 回归用例）
  / **style-annotate**（⚠️ 已停用；保留作历史实验工具）
- **工具**：bili-check / bili-login / vision-test
- **独立运行（不在 run_tool）**：`watchdog.py`（假死自愈进程）、`notifier.py`（SMTP，被 watchdog 用）、`update_schedule.py`（周表识图；日常用根目录 `更新周表.bat` 拖图，也可丢 `data/schedule_inbox/` 后无参跑）、`label_stickers.py`（给 `assets/stickers/` 打标，**加了新表情就重跑它**，幂等）
- **★ 实验/诊断工具**（都在 `scripts/` 根下，直接 `python scripts/<名字>.py`）：
  - **`trace_chain.py`** —— **链路追踪**：对一条消息打印整条链的每个决策（谁开火、注入了什么、最终回复）。
    `--reply` 连回复一起生成，`--json x.json` 存下来供**改动前后 diff**。**包装真函数、不重新实现**，
    所以追踪的就是线上跑的东西。（改检索/注入先跑它）
  - **`form_experiment.py`** —— **形式对照实验**：同一个事实写成 system 陈述 / 偏好条目 / 词表 / 指令 /
    assistant 问答对，各生成 N 次，量"采用率"与"逐字复读率"——结论见 `注入设计原理.md` 第三节
  - **`position_experiment.py`** —— **位置对照实验**：同一段内容插在 messages 的最前/中间/最后，
    或 assistant 的早/晚，看"采用率"和"逐字率"——结论见 `注入设计原理.md`（位置决定她把这内容当
    "刚发生过的事"还是"参考资料"）
  - **`empty_turn_experiment.py`** —— **空轮次实验**：素材层全空的轮次垫随机风格样本有没有用（**否定结果**：没用且有跑题风险）
  - **`entry_audit.py`** —— **入口审计**：造"用户会怎么说"测判据能不能勾出来
  - **`judge_experiment.py`** —— **判据横向对比**：同一份手工标注集上比 子串/余弦/BM25/LLM/HyDE/并集
    哪个更好用——结论见 `注入设计原理.md` 2.3 节。
    **标注只在 `scripts/retrieval_eval_cases.json` 一处**（`want` 由它的 `expect` 推导，2026-09-29 起）；
    曾有一份 `judge_labels.json` + 一份内置兜底，三处同源互相打架，已删
  - **`problem_cases.py`** —— **★问题驱动**：读抽查勾选或 `问题记录.md`，对每条**跑一遍线上真实检索**
    把"实际召回了什么"摆出来，你看着事实判该中/不该中 → 生成用例（检索层/生成层都管）。
    `--query "一句话"` 是最常用的用法（只看一条的检索实况）
- **判据回归集**：`scripts/regression_cases.json`（真实翻车固化成确定性判据）→ `run_tool.py regression --check`
- **检索评测集**：`scripts/retrieval_eval_cases.json`（query→各路期望，**唯一真值**）→ `run_tool.py retrieval-eval`
- **风格层标注集（已停用）**：`scripts/style_eval_cases.json` + 根目录 `风格标注.md`（工作台，**不入库**）
  → `run_tool.py style-annotate`。实测无法分辨当前粒度的消融差异，代码保留作历史资产，
  **不要再据此决定人格文件去留**

> ⚠️ `generate-persona` 会覆盖人格三件套，需 `--danger`；`generate-vectors` 缺省指向 statement_final，别靠内置 RAW_CORPUS。

### 4.1 ⭐ 按"我想干什么"找（**找不到工具就看这张**）

| 我想…… | 用这个 | 结果落哪 |
|---|---|---|
| 把机器人跑起来 | 双击 `启动.bat`（NapCat 自己先开） | — |
| **看一条消息为什么这么答** | `run_tool.py problems --query "那句话"`（看检索实况）｜`trace_chain.py "那句话" --reply`（看整条链每个决策） | 打印在屏幕上 |
| **看到一条答得不对，记一笔** | 写进根目录 `问题记录.md`，然后 `run_tool.py problems --from-note` | 屏幕上是**草案**；加 `--write` 才写进用例文件 |
| **抽查她最近答得怎么样** | `python scripts/spot_check.py`（自动挑最可疑的几条）→ 手勾 `[x] 有：…` → `run_tool.py problems --from-spot` | `outputs/spot_check.md` |
| **改完检索/素材，看有没有变坏** | `run_tool.py retrieval-eval`（48 条标注，**改语料必跑**，见黄金律 5） | `outputs/eval/retrieval/eval_report.json` |
| **改完生成/人格，看有没有变坏** | `run_tool.py regression --check` | `outputs/eval/regression/check.txt` |
| 想知道每条素材**有没有召回入口** | `python scripts/entry_audit.py all`（也可 `terms`/`legendary`/`core_story`） | 打印 |
| 想知道某一路现在**开火率/覆盖**多少 | `python scripts/coverage_check.py` | 打印 |
| 想知道某一路的**噪声地板/阈值** | `python scripts/threshold_scan.py` | 打印 |
| 想比**同一任务用哪种判据更好** | `python scripts/judge_experiment.py --task corpus` | 打印（标注来自 4.2） |
| 比较**同一个事实写成不同形式**的差别 | `python scripts/form_experiment.py` | 打印 |
| 比较**同一段内容放不同位置**的差别 | `python scripts/position_experiment.py` | 打印 |

> ⚠️ 上表里 `python scripts/xxx.py` 那几个是**独立脚本**（没进 `run_tool` 统一入口）——
> 因为它们是"改链路时才跑的实验/诊断"，不是日常流程。
> **日常那些都在 `run_tool.py` 里**（`python scripts/run_tool.py --help` 能列全）。
| **加新的直播素材** | `persona-pipeline <corpus/behaviors>`；可直接给 `--audio`，也可先 `transcribe` + `clean-transcript` | `outputs/persona_pipeline/<session>/` |
| 加了新表情包 | 丢进 `assets/stickers/` 再跑 `label_stickers.py`，**再人工补 `group`**（类别；缺了它拦不住同类） | `persona/media/stickers.json` |
| 更新周表 | 把图拖到根目录 `更新周表.bat` | `persona/world/schedule.json` |
| 检查 B站/微博能不能连上 | `run_tool.py bili-check` | 打印 |
| 想知道**测试**过不过 | `.venv\Scripts\python.exe -m pytest -q` | 打印（800 个） |

### 4.2 ⭐ 评测/标注在哪、怎么用（**新合并的那份**）

| 东西 | 路径 | 说明 |
|---|---|---|
| **标注数据**（唯一真值） | `scripts/retrieval_eval_cases.json` | 48 条「一句话 → 各路该不该命中」。**改这里就是改用例**，不用碰代码。每条带 `source`（哪来的）/ `why`（为什么这么标） |
| 跑它的代码 | `scripts/retrieval_eval.py` | `run_tool.py retrieval-eval`；`--case co6` 只看一条 |
| 判据对比实验的标注 | **同一份**（2026-09-29 合并） | 它的 `want` 由 `expect` **推导**，不再单独维护 |
| 生成层（回复）回归集 | `scripts/regression_cases.json` | 真实翻车固化成"禁词/复读"判据，`run_tool.py regression --check` |
| 问题驱动入口 | `scripts/problem_cases.py` | `run_tool.py problems`——看到问题 → 摆出检索实况 → 变成用例。**怎么用看 `docs/问题驱动.md`**（含 `--audit` 复核老用例） |
| **风格层标注**（第三份，2026-09-30 已停用） | `scripts/style_eval_cases.json` | 保留作历史资产；路线已停用。不要再让 `style-annotate` 决定人格文件去留 |

## 五、测试（`tests/`，800 个）
`conftest.py` 初始化 NoneBot 并加载插件。核心逻辑（reply_style/retrieval/session/voice/chat_window/bili/group_memory/weibo/short_memory 纯函数）覆盖较全；weibo 推送、config、`__init__` 心跳、watchdog/notifier 覆盖少。

## 六、产物 / 状态（**全部不入库**）

### 6.1 机器人**运行中**写的东西（删了会丢状态，别乱删）

| 路径 | 谁写的 | 是什么 | 能删吗 |
|---|---|---|---|
| `data/chat_log/chat.jsonl` | `chatlog.py` | **每轮对话落盘**（她说了什么、用户说了什么）——事后分析/抽查的唯一事实源。超 20MB 会轮转，只留最近 10 个 | 能，但**抽查和问题驱动都靠它** |
| `user_memory/short_term.json` | `memory.py` | 短期记忆（每会话最近 10 轮滚动窗口） | 能（= 忘掉刚聊的） |
| `user_memory/long_term.json` | `memory_manager.py` | **长期记忆**：用户画像 / 承诺 / 印象卡 | 能（= 忘掉所有人） |
| `user_memory/session.json` | `session_memory.py` | 会话记忆（话题/事件/指代补全） | 能 |
| `user_memory/groups.json` | `group_memory.py` | 群记忆（成员昵称、群近况） | 能 |
| `data/bili_state.json` `weibo_state.json` | 两个 bridge | 去重状态（"这条动态推过了"） | 能，但会**重复推送** |
| `data/heartbeat` / `qq_alive` / `qq_offline` | `__init__.py` | 心跳与在线信号（watchdog 靠它判假死） | 能 |
| `data/voice_cache/` | `voice.py` | 语音合成缓存 | 能 |
| `data/schedule_inbox/` | 你放 | 丢进周表图 → 双击 `更新周表.bat` 处理 | 能 |
| `data/weibo_cookies.json` | `weibo_bridge.py` | 微博会话（**敏感**，已忽略） | 能（要重新扫码） |
| `logs/watchdog.log` | `watchdog.py` | 自愈进程日志 | 能 |

### 6.2 工具**跑出来的**分析产物（**可以整个删掉重跑**）

`outputs/` 按"流水线阶段"分子目录——**名字就是阶段**：

| 子目录 | 哪个工具写的 | 里面是什么 |
|---|---|---|
| `transcribe/` | `transcribe-whisper` | 音频 → 带时间戳的转写 |
| `clean/` | `clean-transcript` | 转写清洗（去口水词/断句） |
| `pace/` | `analyze-pace` | 直播"节奏地图" |
| `convert/` | `convert-to-chat` | 直播原文 → 聊天化（behaviors 的原料） |
| `mine/` | `mine-phrases` / `mine-theme` | 挖措辞指纹 / 挖主题素材 |
| `statements/` | `generate-statements` | 场景化陈述（corpus 的原料） |
| `persona_extract/` | `extract-persona` | behaviors 候选（877 条 → 只出 2 条，**用户决定先放着**） |
| `eval/` | 三个评测工具 | `eval/retrieval/` 检索评测报告、`eval/regression/` 判据回归明细、`eval/persona/` 人格评测 |
| `trace_baseline/` | `trace_chain --baseline` | 链路追踪基线（供改动前后 diff） |
| `_qqbridge/` | **不是我们的** | 参考项目 qq-bridge 的规则文件副本（调研用） |

> ⚠️ `outputs/` **根下**还有 53 个 `_` 开头的文件（`_clean_*.json` `_corpus_*.py` …）——
> 是历次**一次性**实验的中间产物，**可以放心删**。
> 几个人读产物值得留着：`spot_check.md`（抽查）、`eval/`（评测报告）、
> `statement_final.before-rewrite.json`（corpus 改造前的存档）。
