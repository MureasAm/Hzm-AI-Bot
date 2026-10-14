# 第一集验收报告：人格核心与完整注入链路

> 对应规格：[`EP01_PERSONA_INJECTION_PRODUCTION_SPEC.md`](EP01_PERSONA_INJECTION_PRODUCTION_SPEC.md) §16 验收标准
> 制作时冻结 commit：`97a1e9540912372ad993c5e4249d2b97dd325030`
> 数据生成时间：见各 `video/data/ep01/*.json` 的 `generated_at`

## 0. 交付物

| 文件 | 说明 |
|---|---|
| `video/out/hazel-bot-ep01-persona-injection-with-voice.mp4` | 第一集成片（1920×1080 / 30fps / 配音+字幕） |
| `video/out/ep01-preview.mp4` | 0.5 倍速预览（同画面，无配音） |
| `video/src/episodes/ep01/` | 第一集 Remotion 工程（10 场景 + 9 公共组件） |
| `video/data/ep01/` | Phase 0 冻结的真实证据（4 个 JSON） |
| `video/public/audio/ep01/` | 151 行灰泽满 GPT-SoVITS 旁白 + 时长报告 + 对齐整轨 |
| `video/public/subtitles/ep01.srt` | 字幕 |
| `video/EP01_ACCEPTANCE_REPORT.md` | 本文件 |

**规模**：10 场景 / 151 行旁白 / 608.3s（10:08，规格允许 9:30～10:30）。

---

## 1. 内容正确性

| 验收项 | 结论 | 证据 |
|---|---|---|
| 不再出现已删除的 `traits/styles/voice_samples/phrases` 作为当前架构 | ✅ | 旁白与画面都没有把它们当现行机制。唯一出现处是 `BehaviorCoreScene` 的"旧的四个文件（2026-10-04 拆分）"卡片，按 §5.3 只作历史背景 |
| 旧文件只作为历史背景出现 | ✅ | 同上；旧卡降透明度并标注并入去向（稳定内容→骨架 / 真人示范→behavior），无"垃圾桶式嘲讽" |
| 三板对照来自真实模型输出 | ✅ | `capture-persona-comparison.py`：真实 system_prompt（3413 字）+ 真实 `_format_behavior_rule` + 真实参数，3 版 × 5 次 = 15 条；`raw` 与 `cleaned` 都保留，未人工改写 |
| full trace 来自真实代码路径 | ✅ | `capture-full-trace.py` **包装真实函数**（`gather_retrieval`/`classify_l3`/`retrieve_corpus_candidates`/`judge_corpus`/`select_behavior_item`/`retrieve_preferences`/`retrieve_core_stories`/`fuse_and_truncate`/`build_message_list`/`generate_reply`），不重新实现 |
| 未命中的层明确保持灰色 | ✅ | 21 条判别中 **14 命中 / 7 沉默**；沉默项在画面里灰化并进"本轮不注入"区 |
| 不宣称 Memory V2 已正式影响回复 | ✅（并已如实更新） | 规格写作时（2026-10-11）称 V2 仍是影子模式，但代码里 `MEMORY_V2_SHADOW` / `MEMORY_V2_INJECT` 都默认 `"1"`（`memory_v2.py:207-220`），V2 是现行长期记忆。旁白据实改为"长期记忆现在已有新的类型化实现，正在参与回复；它怎么记是第二集的内容" |
| 语音与文字发送顺序符合当前代码 | ✅ | `ReplyPipeline` 按真实顺序画：生成 → 防复读 → 清洗 → `should_voice` →（语音成功**硬 return**，不发拆分文字 / 否则 `split_reply`）→ 回写。画面明确画成**互斥二选一** |

### 1.1 与规格逐项对照的额外发现（已在制作中修正，不留错）

| 规格原文 | 代码实际 | 处理 |
|---|---|---|
| `format_behavior_rule`（§6.2） | 真名是 `_format_behavior_rule`（`persona.py:95`） | 视频不显示函数名；脚本调用真名 |
| "17 段注入"、"behavior 16 条" | 实际 21 个逻辑槽位（1 个 v1 死槽）；behaviors **17** 条 | 视频不写死数字：`DECISIONS` 与 behaviors 条数都从数据读 |
| §11 把"写回记忆"画在最后 | 写回在 `handle_chat` 内、`clean_reply` 之前 | 画面按**真实顺序**画（写回在第 6 步，且在发送之前） |
| §10.3「连续索要」预期亮起 | 由 `classify_l3` 的 LLM 判定（**非**本地计数；`memory.count_consecutive_requests` 是死代码） | 画面标 `method=LLM`；`local-reference` 只在结果文件里留作人工核对，明确标注不冒充线上判定 |
| §4.2 参数基线 temperature 0.85 / frequency_penalty 0.3 / max_tokens 150 / thinking disabled | 与 `constants.py:64-69` 一致 | 画面显示的模型参数直接从 `persona-comparison.json` 读 |

---

## 2. 叙事

| 验收项 | 结论 | 证据（scene 相对时间） |
|---|---|---|
| 前 45 秒看到真实差异 | ✅ | `cold-open` 0:00–0:48.8 三栏真实回复：裸模型"谢谢你的夸奖！…我其实是个AI" / 仅骨架"灰泽满，你今天是不是吃错药了" / +behavior"哎呦天啊，别这么夸灰泽满，唱得也就那样吧" |
| 前 3 分钟讲清两份人格核心 | ✅ | `system-prompt` 0:48.8–1:35.0 + `behavior-core` 1:35.0–2:49.3（情景→反应→例句三段 + few-shot） |
| 中段完成 files → messages 的转折 | ✅ | `files-to-messages` 2:49.3–3:27.0：文件卡 → FORMATTER（读取/判断/格式化）→ messages 卡 |
| 观众能复述人格/事实/现场 | ✅ | `three-domains` 3:27.0–4:22.9 三色列 + 三条职责边界 + 长期记忆归"现场"的边界说明 |
| 观众能复述四种判别问题 | ✅ | `four-gates` 4:22.9–6:03.6 四个 Gate（每轮都需要吗 / 提到了吗 / 意思相近吗 / 是哪种人际情景）+ 组合路径说明 |
| 结尾重新跑通整条链，而不是停在静态架构图 | ✅ | `full-trace` 6:03.6–7:31.2 用真实 trace 走完 21 条判别 → 15 段 messages；尾部**停留 2 秒以上**让观众看清 |

---

## 3. 视觉与声音

| 验收项 | 结论 | 证据 |
|---|---|---|
| 1920×1080、30fps | ✅ | `ep01-narration.json` 的 `width/height/fps`；成片 ffprobe 1920×1080 |
| 全部动画由 frame 驱动 | ✅ | 只有 `useCurrentFrame()` + `spring`/`interpolate`；仓内无 `transition:`/`@keyframes`/Tailwind 动画类（`theme.ts` 注释亦写明禁止） |
| 每个 Sequence premount | ✅ | `Series.Sequence` 与配音 `Sequence` 都设 `premountFor={narration.fps}` |
| 所有 public 资源走 `staticFile()` | ✅ | 立绘 `staticFile("hazel_stand.png")`、配音 `staticFile(\`audio/ep01/${id}.wav\`)` |
| 字幕无溢出、最多两行 | ✅ | 字幕轨在标点处折行、最多 2 行；`export-captions.mjs` 对超 22 字的 36 行在 SRT 里折行；抽查帧未见溢出 |
| 全程灰泽满 GPT-SoVITS | ✅ | 151/151 行由 HZM-SPEAK（`HZM-SPEAK-e5.ckpt` + `HZM-SPEAK_e15_s525.pth`，`tts_infer.yaml` 的 `custom` 段）合成，无中性配音；无 `toneFrequency` |
| 立绘不遮挡信息 | ✅ | 立绘只在开场/吐槽/例句/转场/结尾出现，位置避让 messages 面板与字幕安全区；抽查帧确认 |
| 没有模板化 AI 大脑/粒子视觉 | ✅ | 全部为折角文件卡、矢量 Gate 图标、聊天气泡、骨架线条；无发光大脑/神经元/粒子流 |
| 旁白、画面、音效不同时争抢注意力 | ⚠️ 部分 | 旁白与画面按 §2.2 第 3 条编排（立绘出现时旁白短）；**音效未做**——规格 §13.4 是"可使用少量"（可选），本集未加 |

---

## 4. 隐私与可复现性

| 验收项 | 结论 | 证据 |
|---|---|---|
| 演示用户明确标记为虚构 | ✅ | 画面顶部常驻 `固定 fixture · 全部虚构`；fixture `fictional: true` |
| 不读取真实聊天日志或真实用户记忆 | ✅ | capture 脚本把 `memory.use_storage` / `session_memory.use_storage` / `core._MEMORY_V2_STORE` 全部重定向到临时目录；`data/chat_log/` 未触碰 |
| 固定时间、天气、周表均写入 fixture | ✅ | `full-trace-fixture.json`：`clock=2026-10-08T18:40:00+08:00`、`weather={广州,小雨,24℃}`、周表取自真实 `schedule.json`（周四 19:00） |
| 记录生成模型、参数、commit 和数据生成时间 | ✅ | 每个数据文件都带 `model` / `params` / `commit` / `generated_at` |
| Remotion 渲染不依赖网络或在线模型 | ✅ | 组件只读 `video/data/ep01/*.json` 与 `ep01-narration.json`；无 fetch/axios/模型调用 |

---

## 5. 修改记录

### 5.1 第二轮反馈后的修改（按段落）

| 段 | 反馈 | 处置 |
|---|---|---|
| **01** | 回复看不出是谁说的（内部应该是「灰泽满：」这种标签） | 回复卡加**说话人标签** `灰泽满：`，用户消息加 `用户：`。前缀与 `_format_behavior_rule` 注入给模型的格式（`粉丝说：… → 灰泽满：…`）是同一套，观众看到的就是模型实际读到的样子。同时补了一张**证据条**：三栏选中那条的 `raw → cleaned` 并列显示（`clean_reply` 会把 `……` 归一成 `，`，只显示 cleaned 会让人以为模型就是那么写的） |
| **03** | 例句超出底部边界；立绘大半在界外 | 例句从"压第三张卡下面"改成**右栏**，与"情景/反应"并排 → 总高从 1000+ 降到约 450px；few-shot 说明与历史演变改成横向两栏；立绘从 `right:-96 / bottom:-210` 收进画面内（右上方空白处）。顺带修掉一处数据渲染问题：`behaviors.json` 的 response 里有 `**重点**` markdown 标记，原来被当纯文本渲染出裸露的 `**`，现在统一走 `RichText` 解析 |
| **07** | 重构：记忆部分不讲；把所有可能注入的文件都列出来；主要讲判断和文件形式 | 整段重做成一张**注入清单**：三列（人格 / 事实 / 现场），共 **17 项**，每条一张卡只放「文件 + 判别 + 形式」三行；旁白讲到哪条哪条点亮。**移除了全部记忆层级的讲解**（长期/会话/短期），只保留"用户长期记忆"作为现场类的一条条目 |
| **10** | 版式不美观；下一集主要讲记忆 | 两条结论做成**结构一致的左右对照行**；第二部分用细线分隔居中、role 标签等宽排列；预告独立成卡并列出下一集三个看点；立绘移到右下角空白处（原来贴左边会压住结论卡）。旁白第 4、5 行合成一句，腾出时间给记忆预告；预告明确改成"下一集讲记忆" |

### 5.2 第一轮反馈后的修改

| 反馈 | 原因 | 处置 |
|---|---|---|
| **进度条与大标题重叠** | 标题是 58px × lineHeight 1.08，从 y≈89 排到 y≈152；而章节进度条固定在 `top:148`，正好压在标题字上 | 进度条改成**贴顶 3px 细线**（`top:0`），不再占任何内容区 |
| **配音语速有时太慢不自然** | 两件事叠加：① 参考音频基准语速偏慢（中位 4.96 字/秒）；② **随机 seed 会抽到"拖慢"**——实测有行抽到 26.4 秒（期望 3.7 秒） | ① `speed_factor` 设 1.2 → 语速中位数 **4.96 → 6.48 字/秒**；② 检查从"只拦太短"扩成**双向**（>期望 140% 也重试）。旁白纯时长 544.6s → 444.4s |
| **要求逐段拆开看** | 整片 10 分钟，改一处要重渲全片 | `scripts/split-ep01.mjs` → `out/ep01-parts/01-cold-open.mp4` … `10-closing.mp4`（0.5 倍，各带该段配音） |

### 5.3 修改中发现并修掉的四个隐患

1. **动画锚点写死秒数会漂移** —— 时间轴每次重算后，场景里写死的 `sec(6.4)` 不再落在对应那句旁白上。改成 `at("scene", "line-id", offset)` 从旁白反查，并新增 `check-scene-timing.mjs`。
2. **折句会让 line id 重新编号，而 `at()` 的参数 TypeScript 查不出来** —— typecheck 过、单帧也过，但**渲染整段时抛错**（第一轮 10 段里有 2 段落不下来就是这个）。新增 `check-anchors.mjs`。
3. **"取最不烂的一次"会把坏音频发出去** —— 质量重试原来在多次不合格时取"最接近期望"的那次，结果 `closing-3`（期望 5300ms）被写成 2140ms 的退化音频，而且校验也放它过了。改成**不合格就抛错**，宁可留上一版也不要写坏音。
4. **折句后行号复用，旧音频"看起来"还在** —— 校验原来以"缓存里有没有同文本"为主判据，行号复用后旧音频的缓存条目仍然存在 → 判成 ok，但字幕和声音对不上。改成**以时长比对为主判据**（与当前文本的期望时长差得远就是有问题）。

### 5.4 复核后的时长与语速

| 指标 | 第一版 | 现在 |
|---|---|---|
| 总时长 | 608.3s | **600.0s**（整轨与时间轴差 0ms） |
| 旁白纯时长 | 544.6s | 444.4s |
| 旁白行数 | 151 | 148 |
| 语速中位数 | 4.96 字/秒 | **6.48 字/秒** |
| 最慢行 | 1.35 字/秒 | 3.82 字/秒 |
| 行间停顿 | 180~300ms（偏挤） | 603~700ms（富余摊匀） |

---

## 6. 已知偏差与决定（如实记录）

1. **总时长 608.3s（10:08）** — 规格给的是"约 10 分钟（允许 9:30～10:30）"，在区间内。
2. **5 行旁白超过规格的"建议 6 秒上限"（6.0～7.0s）** — 这是 HZM-SPEAK 在这个音色/语速下的真实产出
   （例如"所以真的不是把所有文件都塞进去。"念 7.02s）。仍远低于规格的硬上限 12 秒；
   `check-narration.mjs` 把它们标为 `~`（建议上限告警）而不是 `!!`（错误）。
3. **音效未做** — 规格 §13.4 措辞是"可使用少量"（可选），本集未加，以免在时间轴稳定前引入额外变量。
4. **Memory V2 的表述按代码事实更新**（见 §1）——规格写作时的"影子模式"已过期。
5. **`behaviors.json` 的确定性兜底有命名不匹配**（`behavior_keywords.json` 三个组名里有两个
   与 `behaviors.json` 的名字对不上，那两组永远不会触发）。这不影响本集画面（本集 behavior 由
   LLM 情景分类命中），但**是一个真实的代码问题，未在本集制作中改动**，已记录在此备查。

---

## 7. 复现步骤

```bash
# 1) 证据（Phase 0）
python video/scripts/capture-persona-comparison.py --runs 5
python video/scripts/capture-full-trace.py --generate --runs 3
python video/scripts/build-decisions.py
python video/scripts/build-persona-structure.py

# 2) 旁白与时间轴
python video/scripts/build-narration.py          # 改文案只改这个脚本里的 LINES
node video/scripts/redistribute-narration.mjs ep01 --write
node video/scripts/export-captions.mjs ep01

# 3) 配音（需要 GPT-SoVITS 在 9880）
python video/scripts/generate-voiceover.py ep01 --port 9880
python video/scripts/verify-voiceover.py ep01 --fix   # 缺/旧/退化的一起修
node video/scripts/fit-narration-to-audio.mjs ep01 --write
node video/scripts/export-captions.mjs ep01

# 4) 校验与渲染
node video/scripts/check-narration.mjs ep01
node video/scripts/check-scenes.mjs --multi
npx remotion render src/index.ts HazelBotEp01PersonaInjection \
  out/hazel-bot-ep01-persona-injection-with-voice.mp4 \
  --codec h264 --props=render-props-ep01-voice.json
```

### 制作中踩到的四个坑（写进脚本/配置注释了，避免复发）

1. **PowerShell 提交中文会乱码** —— `Invoke-WebRequest` 按本地代码页编码请求体，TTS 返回固定
   0.34 秒桩音频。改用 Python 显式 UTF-8（`generate-voiceover.py`）。
2. **固定 `seed=42` 会让某些句子稳定退化** —— 本集有 11 行中招（420ms 垃圾音频）；
   同一句把 seed 改成 `-1` 就正常。现在改成"不固定 seed + 合成后按时长模型检查质量 + 不合格换 seed 重试"。
3. **151 路 `<Audio>` 会让最终 ffmpeg 拼接失败** —— 改成
   `build-voiceover-track.py` 把逐行旁白按时间轴拼成**一条整轨**，Remotion 侧只剩 1 个 `<Audio>`；
   实测整轨 608.32s vs 时间轴 608.32s，**差 0ms**。整轨再转一份 160k AAC（8.8MB，时长一致）供渲染用。
4. **x264 编码阶段 OOM** —— 症状是渲染跑到 100% 后 ffmpeg 报
   `x264 [error]: malloc of size 7088704 failed` → `Conversion failed!` 且不产出文件
   （0.5 倍预览不触发）。本机 15.7GB 内存、空闲常只剩 ~3GB。
   处置：`remotion.config.ts` 调轻编码档（`veryfast` + CRF 18），必要时 `--concurrency 1`。
