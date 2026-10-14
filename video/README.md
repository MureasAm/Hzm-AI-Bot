# 灰泽满注入链路：Remotion 工程

这个工程有两个 composition：

| composition id | 内容 | 状态 |
|---|---|---|
| `HazelInjectionExplainer` | 旧样片（92s，8 场景，含已删除架构 traits/styles/voice_samples/phrases） | 保留作历史样片与组件参考，**不复用其内容** |
| `HazelBotEp01PersonaInjection` | **第一集：人格核心与完整注入链路**（约 10 分钟，10 场景） | 正式集数，见下 |

第一集的制作规格见 [`EP01_PERSONA_INJECTION_PRODUCTION_SPEC.md`](EP01_PERSONA_INJECTION_PRODUCTION_SPEC.md)。

---

## 第一集：人格核心与完整注入链路

### 目录结构

```text
video/
├── data/ep01/                          # Phase 0 冻结的证据（Remotion 只读，渲染时不联网）
│   ├── persona-comparison.json         # 三板对照：真实模型输出，每版 5 次共 15 条
│   ├── persona-structure.json          # system_prompt 章节表 + behaviors 那一条（从真实文件抽）
│   ├── full-trace-fixture.json         # 固定虚构 fixture（时钟/天气/用户/批次）
│   ├── full-trace-result.json          # 真实代码跑出的 trace + 最终 15 段 messages
│   └── full-trace-decisions.json       # 由 trace 派生的 21 条判别（命中/沉默 + 真实理由）
├── src/episodes/ep01/
│   ├── Ep01PersonaInjection.tsx        # 主组件：时间轴 + 字幕轨 + 配音轨
│   ├── ep01-narration.json             # 旁白（生成物，源在 scripts/build-narration.py）
│   ├── ep01-fixtures.ts                # 数据桥（唯一读 data/ep01 的地方）
│   ├── theme.ts                        # 色彩语义 + 帧驱动动画工具
│   ├── scenes/                         # 10 个场景
│   └── components/                     # 9 个公共视觉模块
├── narration/ep01.csv                  # 待合成语音清单
├── public/audio/ep01/<line-id>.wav      # 灰泽满 GPT-SoVITS 旁白
└── public/subtitles/ep01.srt            # 字幕
```

### 旁白与时间轴：三段时间轴，别手写帧数

旁白有两条硬约束（规格 §13.1、§13.3）：**单行 2.5~6 秒、每行不超过 22 个中文字符**。
手写时间槽几乎一定违反其中一条，所以时间轴分三步算：

```bash
# 1) 重建旁白（改文案只改 scripts/build-narration.py 里的 LINES）
python video/scripts/build-narration.py

# 2) 按字数求解时间轴（场景时长由台词预算等比例放大到目标总时长）
npm run ep01:refit              # node scripts/redistribute-narration.mjs ep01 --write

# 3) 导出 CSV + SRT
npm run ep01:captions           # node scripts/export-captions.mjs ep01
```

`ep01-narration.json` 是**生成物**——不要手改它的 `startMs/endMs`，会被下一步覆盖。

### ⚠️ 场景内的动画锚点必须用 `at()`，不要写死秒数

时间轴会被重算（改文案、改语速、改停顿都会变）。场景里若写死 `sec(6.4)`，
重算后画面就会"讲 A 演 B"。所以锚点统一走：

```ts
import {at, atSec} from "../ep01-fixtures";

const gateStart = at("full-trace", "full-trace-5", 0.6);  // 那句旁白之后 0.6 秒
const w = atSec("content-forms", "content-forms-17", -0.4); // 提前 0.4 秒
```

`at()` 从 `ep01-narration.json` 反查时刻，画面永远跟着旁白走。

**折句会让 line id 重新编号**（`cold-open-11` 变 `-10`），而 `at()` 的参数是字符串——
TypeScript 查不出来，要到渲染时才抛错。所以有一个专门的校验：

```bash
npm run ep01:check-anchors      # 所有 at() 引用的行是否真实存在且属于该场景
npm run ep01:scene-timing       # 锚点是否落在旁白区间内
```

### 配音（灰泽满 GPT-SoVITS）

```bash
# 1) 启动 TTS（bot 的 9880 已在跑就直接用；否则起独立实例）
npm run voice-api               # scripts/start-speak-api.ps1（默认 9881）

# 2) 合成全部旁白（语速 1.2 + 双向质量检查 + 换 seed 重试）
npm run ep01:voiceover          # python scripts/generate-voiceover.py ep01 --port 9880 --speed 1.2

# 3) 核对（缺音频 / 文本已变 / 退化 / 拖慢 一次修）
npm run ep01:verify-voice -- --fix

# 4) 拼成**一条整轨**（关键：151 路音频会让最终 ffmpeg 拼接失败）
npm run ep01:voice-track

# 5) 用实测音频钉准时间轴，再重出字幕
npm run ep01:fit-voice -- --write
npm run ep01:captions
```

⚠️ **四个必须知道的坑**（本集都踩过，脚本注释里也写了）：

1. **不要用 PowerShell 的 `Invoke-WebRequest` 提交中文旁白** —— 它按本地代码页编码请求体，
   服务端收到乱码后返回一个固定 0.34 秒的桩音频。`generate-voiceover.py` 显式用 UTF-8。
2. **不要固定 `seed=42`** —— 某些句子会**稳定**合成出 0.42 秒垃圾音频（本集多行中招）。
   而且随机 seed 还有另一个方向的坏运气：**拖慢**（实测有行抽到 26.4 秒 / 期望 3.7 秒，
   听感就是"语速太慢不自然"）。现在改成不固定 seed + **双向**检查（太短 / 太长都重试）。
3. **基准语速要显式提**：参考音频语速偏慢，`speed_factor=1.2` 能让语速中位数从
   4.96 提到 6.48 字/秒，慢读行显著减少。
4. **配音轨要用整轨，不要 151 个 `<Sequence><Audio/>`** —— 151 路音频输入会让
   最终 ffmpeg 拼接阶段失败。`build-voiceover-track.py` 把逐行旁白按时间轴（含静音间隔）
   拼成一条 wav + 一份 AAC；Remotion 侧只剩 1 个 `<Audio>`。

### 分段导出（逐段看细节 / 逐段改）

整片 10 分钟，改一处要重渲全片。分段之后每段只有 20~128 秒：

```bash
npm run ep01:parts                                  # 全部 10 段，0.5 倍，带配音
npm run ep01:parts -- --full                        # 原始分辨率
npm run ep01:parts -- --only four-gates,closing     # 只出某几段
npm run ep01:parts -- --no-voice                    # 只看版面（更快）
```

输出 `out/ep01-parts/01-cold-open.mp4` … `10-closing.mp4`。分段音频是从整轨
**按同一时间区间裁**出来的，所以分段里的语速与整片完全一致。

### 渲染

```bash
npm run studio                  # 打开 studio，两个 composition 都在
npm run ep01:check              # 校验时间轴约束
npm run ep01:scene-timing       # 校验场景内动画锚点是否还跟旁白对齐
npm run ep01:still              # 单帧
npm run ep01:preview            # 0.5 倍预览（无配音）
npm run ep01:render:voice       # 1080p 带配音
node video/scripts/check-scenes.mjs --multi   # 每场景 3 帧抽查（输出 out/ep01-checks/）
```

⚠️ **1080p 渲染要按内存调档**（本集踩过）：如果机器空闲内存不多（本机 15.7GB，常剩 ~3GB），
x264 会在最后拼接阶段 OOM：
`x264 [error]: malloc of size 7088704 failed` → `Conversion failed!`，且不产出文件
（0.5 倍预览不会触发）。`remotion.config.ts` 已经把编码档位调轻（`veryfast` + CRF 18）；
再不行就显式降并发：

```bash
npx remotion render src/index.ts HazelBotEp01PersonaInjection out/out.mp4 \
  --codec h264 --concurrency 1 --x264-preset veryfast --crf 18 --props=render-props-ep01-voice.json
```

### 数据是怎么来的（Phase 0）

第一集的画面**不允许手工伪造命中**（规格 §10.5），所以先冻结证据：

```bash
# 三板对照：真实模型 + 真实 system_prompt + 真实 behavior 格式化，每版 5 次
python video/scripts/capture-persona-comparison.py --runs 5

# 完整链路：固定 fixture 替换时钟/天气/存储，调用真实 build_message_list 等
python video/scripts/capture-full-trace.py --generate --runs 3

# 把 trace 派生成画面要用的 decisions（hit/method/reason 全部来自真实痕迹）
python video/scripts/build-decisions.py

# system_prompt 章节表 + behaviors 那一条
python video/scripts/build-persona-structure.py
```

`full-trace` 的读秒窗口、判别、确认都是**真实 LLM 调用**，所以同一输入多次运行结果可能不同；
`capture-full-trace.py` 会保存每次 run 的摘要并把选择规则写进 `runSelection`，不挑好看的。

---

## 旧样片（保留）

`1920x1080` / `30fps` / `92s` / 8 场景，无配音也能渲染（画面内置逐句字幕）。

```bash
npm run studio
npm run render:preview
npm run render
npm run captions
npm run fit-voice
npm run voiceover
```

配音时间轴在 `src/narration.json`；改完跑 `npm run captions` 生成 `narration.csv` 与
`public/subtitles.srt`，把每个 `line_id` 合成同名文件放进 `public/audio/`，再 `npm run render:voice`。

模型权重在 `D:\Huizeman-AI-Voice-Cloning\GPT-SoVITS_V4\...`（`tts_infer.yaml` 的 `custom` 段：
`GPT_weights_v2/HZM-SPEAK-e5.ckpt` + `SoVITS_weights_v2/HZM-SPEAK_e15_s525.pth`），
参考音频是 `assets/voice_refs/ref_voice.wav`。

> ⚠️ 旧样片的旁白（`src/narration.json`）讲的是**已删除的架构**（traits / styles /
> voice_samples / phrases、四路 RRF），只作历史对照，不要当成当前实现。
