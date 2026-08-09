# 灰泽满 AI 聊天机器人

基于 **NoneBot2** 框架构建的虚拟主播"灰泽满"QQ 聊天机器人，部署于 QQ 平台。通过**素材驱动的人格系统**（真实直播语料 → 声音样本 + 措辞指纹 + 直播记忆 → 按需检索注入），高度还原主播的语言风格、行为模式和情感表达。

**当前版本：V5**（感知增强 + 节奏系统：时间/农历/天气、glm-4.6v 视觉、B站开播/动态联动推送、读秒窗口攒批、分批发送、智能归纳）。完整重构历程见 `ROADMAP.md`。

## 核心特性

- **素材驱动人格系统**：从真实直播语料提取声音样本（voice_samples.json，56 条）与措辞指纹（phrases.json，11 组），用"样本示范"而非"规则规定"来塑造说话方式
- **五路融合检索（V3）**：直播记忆 / 声音样本 / 行为触发 / 措辞指纹四路向量检索 + RRF 加权融合 + 预算截断，按用户消息情境只注入当前相关信息；用户长期/短期记忆两路确定性注入。全程只调 1 次 embedding
- **直播记忆 RAG（corpus）**：273 条场景化陈述，作为背景记忆（仅相关时提及，不参与风格）
- **精简人设提示词**：system_prompt 从 318 行/3.2 万字精简到 3500 字（原版备份 v9）
- **措辞指纹库**：同一意思 → 她真实说过的原话（被夸→"也没有啦"、被戳穿→"啊？没有吧"），全部从素材提取
- **分层人格引擎**：经典梗硬匹配 → 人设提示词 → 行为触发检索 → 直播记忆 RAG → 声音样本 few-shot
- **记忆系统**：短期（最近 5 轮 + 一致性规则）+ 长期（用户印象/事实/重要时刻/**承诺约定**，self_fact 停用防污染）
- **感知增强（V5）**：时间/农历/节气/天气感知（默认墨尔本=灰泽满所在地，各绿冻按记忆注入自己城市）；glm-4.6v 视觉理解（图片/表情包 → 描述注入对话）；B站开播/动态联动推送（结构化通知 + 配图 + 白名单）
- **真人节奏（V5）**：读秒窗口（连发消息攒批，静默 5~10s 统一回复）+ 分批发送（长回复一句句带打字延迟发出）+ 智能归纳（攒批后先理解整体再回）
- **离线工具箱**：`run_tool.py` 统一入口，13 个子命令覆盖从"直播音频"到"人格数据"的完整蒸馏流程（含 `bili-check` / `vision-test`）

## 技术栈

| 模块 | 技术 |
|:---|:---|
| 机器人框架 | NoneBot2 + OneBot V11 |
| QQ 接入 | NapCatQQ |
| 对话模型 | DeepSeek-V4-Flash |
| 视觉模型 | 智谱 glm-4.6v（关思考模式，秒级描述） |
| Embedding 模型 | 智谱 AI embedding-3 |
| 天气 | 和风天气免费版（专属 API Host） |
| 农历/节日 | lunar-python（本地计算） |
| B站联动 | bilibili-api-python + 直播公开接口 |
| 图片处理 | Pillow（统一转 JPEG，兼容 WEBP/透明图） |
| 语音转写 | faster-whisper (medium, GPU) |
| 数据存储 | JSON (记忆、向量、声音样本、人格规则) |

## 快速开始

### 前置要求
- Python 3.10+
- QQ 账号（NapCatQQ 登录）、DeepSeek API Key、智谱 AI API Key、本地 GPU（可选）

### 安装与配置
```bash
git clone https://github.com/MureasAm/Hzm-AI-Bot.git
cd Hzm-AI-Bot
pip install -r requirements.txt
```
创建 `.env.prod`：
```
OPENAI_API_KEY=你的DeepSeek_API_Key
OPENAI_API_BASE=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-v4-flash
ZHIPU_API_KEY=你的智谱AI_API_Key

# 感知增强（V5，均可留空，留空则对应功能跳过）
WEATHER_KEY=和风天气Key
WEATHER_BASE_URL=你的和风专属域名   # 控制台-设置页查看
WEATHER_CITY=墨尔本              # 默认天气城市（灰泽满所在地）
VISION_MODEL=glm-4.6v
BILI_UID=灰泽满的B站UID
BILI_SESSDATA=你的B站登录Cookie   # 动态监控必需，留空只监控开播
NOTIFY_FRIENDS_WHITELIST=        # 推送白名单QQ号（逗号分隔，留空=全部好友）
```
NapCat 登录机器人 → `python bot.py`

## 离线工具箱

统一入口：`python scripts/run_tool.py <工具> [参数]`

| 工具 | 干什么 |
|---|---|
| transcribe | 音频 → 原始转写 JSON |
| clean-transcript | 转写清洗（合并碎片/繁体转简体/固定错字表） |
| convert-to-chat | 直播 → QQ 聊天回复（V3：分离转述/回答 → 切分 → 压缩） |
| analyze-pace | 节奏地图（`--focus` 聚焦场景 + reasoning） |
| mine-phrases | 从素材批量挖措辞指纹（多维度，输出待审批） |
| generate-statements | 从素材生成场景化陈述（50-120字，供 corpus RAG） |
| generate-vectors | 场景化陈述 → 语料向量库 |
| generate-persona | 场景化陈述 → 人格三件套 |
| precompute | 预计算 trigger/声音样本/措辞向量 |
| pipeline | 四步人格蒸馏流水线 |
| regression | 回复灵性回归测试（A/B 对比） |

**典型工作流**
1. 转写：`run_tool transcribe assets/audio/live.mp3`
2. 清洗：`run_tool clean-transcript -i 转写.json`
3. 节奏地图：`run_tool analyze-pace -i cleaned.json --session 第一场 --focus 被调戏,被夸`
4. 转化聊天：`run_tool convert-to-chat -i 原文.json`
5. 挖措辞：`run_tool mine-phrases -i cleaned.json`
6. 生成背景记忆：`run_tool generate-statements -i cleaned.json` → `run_tool generate-vectors -i statements.json`
7. 向量化：`run_tool precompute --all`

**目录分工**：`assets/` 放素材 → `outputs/<工具>/` 放分析产物 → `data/`/`persona/` 是机器人读取的固定缓存。

## 项目结构

```
bot.py                           # 项目启动入口
src/plugins/chatbot/             # 核心对话逻辑
├── core.py                      # 消息处理主循环（组装 + 调模型 + 更新记忆）
├── constants.py                 # 路径/API/阈值/预算参数 ★
├── config.py                    # NoneBot 配置读取助手
├── context_probe.py             # 时间/农历/节气/天气感知 ★
├── vision.py                    # 视觉理解（glm-4.6v，转码+关思考）★
├── chat_window.py               # 读秒窗口（消息攒批 + 智能归纳 + 分批发送）★
├── bili_bridge.py               # B站开播/动态监听 + 私聊广播 ★
├── persona.py                   # 人格加载 + 行为匹配
├── memory.py                    # 短期记忆（带锁）+ 长期封装
├── rag.py                       # 直播记忆 RAG
└── retrieval.py                 # 四路检索 + RRF 融合 + 预算截断 ★
persona/                         # 人格数据
├── system_prompt.txt            # 人设提示词（V2 精简版）
├── persona_traits/styles/behaviors.json  # 人格三件套
├── voice_samples.json           # 声音样本库（56 条）★
└── phrases.json                 # 措辞指纹库（11 组）★
data/                            # 运行数据 + 向量缓存
├── corpus_vectors.json          # 直播记忆向量库（273 条）
├── memory.json                  # 短期记忆
├── long_term_memory.json        # 长期记忆（含承诺约定）
└── *_vectors.json               # trigger/样本/措辞向量缓存
scripts/                         # 离线工具箱（run_tool.py 统一入口）★
tests/                           # 单元测试
assets/                          # 原始素材
outputs/                         # 分析产物
```

## 人格蒸馏流水线

从直播素材到人格数据的完整流程（核心是"素材驱动"）：
1. **收集素材**：直播录播 → `assets/audio/`
2. **转写** → **清洗** → **节奏地图**（先判噪声，再聚焦场景）
3. **转化聊天**（convert-to-chat，分离转述/回答，切分压缩）
4. **挖措辞**（mine-phrases）→ **生成背景记忆**（generate-statements → generate-vectors）
5. **向量化**：`run_tool precompute --all`
6. 补进样本库/措辞库前**人工审批**（质量优先）

> 改过 `persona_behaviors.json` / `voice_samples.json` / `phrases.json` 后需重跑对应 precompute。

## 记忆系统设计

为每位用户维护独立的记忆卡片，存储：
- 印象标签（如"上班族"、"喜欢催播"）
- 用户事实（如"在准备考研"）
- **承诺约定**（她答应过用户的事，跨会话记住）
- 自我披露记录（self_fact 提取已停用，防 AI 自嗨污染真实人格）

> 短期记忆保留最近 5 轮 + 一致性规则（同一件事借口一致，新问题正常回答）。

## 版本演进

- **V1**：样本 few-shot + 停用记忆污染
- **V2**：提示词 318 行 → 3500 字
- **V3**：五路融合检索（按需注入）
- **V3.1**：措辞指纹库 + 样本分层 + 长度控制
- **V4**：样本库 14→56 全场景均衡、措辞库从素材重建、corpus 更新 273 条、承诺记忆、参数调优（temperature 0.85 + penalty 0.3）
- **V5（2026-08-10，感知增强 + 节奏系统）**：
  - 感知：时间/农历/节气/天气（和风专属域名，**按用户记城市**）；glm-4.6v 视觉（Pillow 转码 + 关思考加速）；B站开播/动态联动推送（结构化通知 + 配图 + 白名单）
  - 节奏：读秒窗口（攒批，静默 5~10s 随机）+ 分批发送（按长度随机延迟打字感）+ 智能归纳（攒批先理解整体）
  - 括号：素材层根治（样本去滥用括号、留情绪顶点示范）+ 输出清洗（clean_reply）

完整重构历程、方法论与踩坑记录见 `ROADMAP.md`。

## 说明

- 本项目仅供学习和交流使用
- 所有 API 密钥均通过 `.env.prod` 管理，已加入 .gitignore
- 人格数据来源于公开直播内容的二次创作
