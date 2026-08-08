# 灰泽满 AI 聊天机器人

基于 **NoneBot2** 框架构建的虚拟主播"灰泽满"QQ 聊天机器人，部署于 QQ 平台。通过**素材驱动的人格系统**（真实直播语料 → 声音样本 + 措辞指纹 → 按需检索注入），高度还原主播的语言风格、行为模式和情感表达。

## 核心特性

- **素材驱动人格系统（本轮最大重构）**：从真实直播语料提取声音样本（voice_samples.json）与措辞指纹（phrases.json），用"样本示范"而非"规则规定"来塑造说话方式——模型看真实对话学会怎么说话，而不是背规则
- **五路融合检索（V3）**：直播记忆 / 声音样本 / 行为触发三路向量检索 + RRF 加权融合 + 预算截断，按用户消息情境**只注入当前相关的信息**；用户长期/短期记忆两路确定性注入。全程只调 1 次 embedding
- **精简人设提示词**：system_prompt 从 318 行/3.2 万字精简到 3500 字（原版备份 v9），删掉数字配额规则，保留核心人格与说话节奏
- **措辞指纹库**：同一意思 → 她的真实原话（被夸→"也没有啦"、被戳穿→"啊？没有吧"），模型表达同类意思时用她的措辞而非自创
- **分层人格引擎**：经典梗硬匹配 → 人设提示词 → 行为触发检索 → 直播记忆 RAG → 声音样本 few-shot
- **长期记忆系统**：记住用户身份、印象标签与聊天历史（默认一律当熟人，不再按互动次数分关系等级）
- **离线工具箱**：`run_tool.py` 统一入口，9 个子命令覆盖从"直播音频"到"人格数据"的完整蒸馏流程

## 技术栈

| 模块 | 技术 |
|:---|:---|
| 机器人框架 | NoneBot2 + OneBot V11 |
| QQ 接入 | NapCatQQ |
| 对话模型 | DeepSeek-V4-Flash |
| Embedding 模型 | 智谱 AI embedding-3 |
| 语音转写 | faster-whisper (medium, GPU) |
| 数据存储 | JSON (记忆、向量、声音样本、人格规则) |

## 快速开始

### 前置要求
- Python 3.10+
- QQ 账号（用于 NapCatQQ 登录）
- DeepSeek API Key
- 智谱 AI API Key
- 本地 GPU（可选，用于 faster-whisper 转写）

### 安装与配置
1. 克隆仓库
```bash
git clone https://github.com/MureasAm/Hzm-AI-Bot.git
cd Hzm-AI-Bot
```
2. 安装依赖
```bash
pip install -r requirements.txt
```
3. 配置环境变量：项目根目录创建 `.env.prod`
```
OPENAI_API_KEY=你的DeepSeek_API_Key
OPENAI_API_BASE=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-v4-flash
ZHIPU_API_KEY=你的智谱AI_API_Key
```
4. 在 NapCat 里登录机器人
5. 启动：`python bot.py`

## 离线工具箱

统一入口：`python scripts/run_tool.py <工具> [参数]`（旧脚本仍可直接运行，新入口为推荐用法）

| 工具 | 干什么 | 常用示例 |
|---|---|---|
| transcribe | 音频 → 原始转写 JSON | `run_tool transcribe assets/audio/live1.mp3` |
| clean-transcript | 原始转写 → 清洗（合并碎片/修错字/加标点） | `run_tool clean-transcript -i 转写.json` |
| convert-to-chat | 直播原文 → QQ 聊天回复（灰泽满自称+保留括号） | `run_tool convert-to-chat -i 原文.json` |
| analyze-pace | 清洗后 → 节奏地图（可多场合并） | `run_tool analyze-pace -i cleaned.json --session 第一场` |
| generate-vectors | 场景化陈述 → 语料向量库 | `run_tool generate-vectors` |
| generate-persona | 场景化陈述 → 人格三件套 | `run_tool generate-persona` |
| precompute | 预计算 trigger/声音样本/措辞向量 | `run_tool precompute --all` |
| pipeline | 四步人格蒸馏流水线 | `run_tool pipeline -i 转写.json` |
| regression | 回复灵性回归测试（A/B 对比） | `run_tool regression --ab` |

**典型工作流**
1. 转写：`run_tool transcribe assets/audio/live1.mp3`
2. 清洗：`run_tool clean-transcript -i outputs/transcribe/live1_raw.json`
3. 转化聊天：`run_tool convert-to-chat -i 原文.json`
4. 节奏地图（多场合并）：`run_tool analyze-pace -i outputs/transcribe/cleaned.json --session 第一场`，第二场加 `--merge`
5. 向量化：`run_tool generate-vectors && run_tool precompute --all`
6. 蒸馏：`run_tool pipeline -i outputs/transcribe/cleaned.json`

**目录分工**
- `assets/` 放你的原始素材（`audio/` 音频、`transcripts/` 转写）
- `outputs/<工具>/` 放人读分析产物
- `data/` 与 `persona/` 是机器人启动要读的缓存，路径固定，一般不用手动改
- 改输入一律用 `-i / --input`，不用改源码

## 项目结构

```
bot.py                           # 项目启动入口
src/plugins/chatbot/             # 核心对话逻辑（模块化）
├── __init__.py                  # 插件入口
├── constants.py                 # 路径 / API / 阈值 / 检索参数常量集中
├── persona.py                   # 人格加载 + 行为触发匹配（trigger 向量缓存）
├── memory.py                    # 短期记忆（带锁）+ 长期记忆封装
├── rag.py                       # 直播记忆 RAG（embed_query / cosine）
├── retrieval.py                 # V3 三路检索 + RRF 融合 + 预算截断 ★
└── core.py                      # 消息处理主循环
persona/                         # 人格数据
├── system_prompt.txt            # 人设提示词（精简版 V2）
├── system_prompt_v9_backup.txt  # 原 318 行完整版备份
├── persona_traits/styles/behaviors.json  # 人格三件套
├── voice_samples.json           # 声音样本库（few-shot）★
├── phrases.json                 # 措辞指纹库 ★
data/                            # 运行数据 + 向量缓存（机器人启动要读）
├── memory.json                  # 用户短期记忆
├── long_term_memory.json        # 用户长期记忆
├── corpus_vectors.json          # 直播记忆向量库
├── trigger_vectors.json         # 行为 trigger 向量缓存
├── voice_sample_vectors.json    # 声音样本向量缓存
└── phrase_vectors.json          # 措辞指纹向量缓存
assets/                          # 原始素材（audio/ 音频、transcripts/ 转写）★
outputs/                         # 分析产物（pace/ 节奏地图、transcribe/ 转写等）
scripts/                         # 离线工具箱（统一入口 run_tool.py）★
prompts/                         # 提示词模板（蒸馏/清洗/描述）
materials/                       # 素材备份
memory_manager.py                # 长期记忆管理（带文件锁）
tests/                           # 核心函数单元测试（pytest，46 条）
bin/                             # ffmpeg 等二进制
```

## 人格蒸馏流水线

从直播素材到人格数据的完整流程（核心是"素材驱动"）：

1. **收集素材**：直播录播 → `assets/audio/`
2. **转写**：`run_tool transcribe` 或 `run_whisper.bat` → faster-whisper GPU
3. **清洗**：`run_tool clean-transcript` → 合并碎片、修错字、加标点
4. **转化聊天**：`run_tool convert-to-chat` → 直播叙述体 → QQ 聊天回复形态
5. **节奏地图**：`run_tool analyze-pace` → 分析她的场景节奏、高频措辞
6. **向量化**：`run_tool generate-vectors && run_tool precompute --all`
7. **人格分析**：`run_tool generate-persona` → 更新 persona/ 下三个 JSON

> 改过 `persona_behaviors.json` / `voice_samples.json` / `phrases.json` 后需重跑对应 precompute。

## 长期记忆设计

为每位用户维护独立的记忆卡片，存储：
- 印象标签（如"上班族"、"喜欢催播"）
- 用户事实（如"在准备考研"）
- 自我披露记录（避免 AI 重复自曝）

> 注：已移除"按互动次数自动升级关系等级"机制，聊天一律默认当作熟人（老朋友语气）。

## 说明

- 本项目仅供学习和交流使用
- 所有 API 密钥均通过 `.env.prod` 管理，已加入 .gitignore
- 人格数据来源于公开直播内容的二次创作
