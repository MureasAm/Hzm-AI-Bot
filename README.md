# 灰泽满（Hazel）· 基于素材驱动的 LLM 角色一致性对话框架

一个把「让 LLM 稳定还原一个真实人物的说话方式」这件事做到极致的工程实践。

表面看，它是一个虚拟主播"灰泽满"的 QQ 聊天机器人（NoneBot2 + OneBot V11 + NapCat）。但这不是一个套壳聊天 bot——**它的核心是一次系统性的角色一致性工程**：从真实直播语料蒸馏人格数据 → 多路检索按需注入 → 分层提示词组装 → 记忆系统 → 感知与真人节奏 → 评测闭环。

> 一句话：**不是靠模型的聪明，而是靠数据、检索、记忆、评测这套体系的构建，让一个 LLM 从"会聊天"变成"像某个人"。**

---

## 为什么做这个

给真实主播灰泽满做一个 QQ 聊天机器人，目标是"最好的展示"——不是通用问答，而是高度还原她的**语言风格、行为模式、情感表达**。

核心挑战很朴素：LLM 默认的"AI 腔"和真人说话差得很远。而让模型"像一个人"，**靠提示词堆规则是没用的**——这条路我走了大量弯路之后，确立了一套方法论，也是这个项目自认为最大的价值。

## 核心方法论（项目的最大价值）

这些不是教科书理论，是踩过坑后验证出来的：

1. **样本 > 规则**：真实对话示范"怎么说话"，比规则规定有效得多。few-shot 看样本学说话，而不是读规则学说话。
2. **素材层解决，别用提示词打补丁**：措辞、括号、重复问题从素材层根治；提示词硬约束是堆砌，无效且走老路。
3. **直播 ≠ 聊天**：直播语料必须经"转聊天"转化，且切分压缩（一个独立意思 = 一条 15-50 字短回复），不能整段保留。
4. **行为 > 标签**：人格标签（"乐观的悲观主义者"）是 tell，模型会当行为模板过度执行；要写"在什么情境怎么反应"。
5. **提示词不放具体台词**：带引号的原话放提示词 = "点名口癖 → 每条都加"；例句下沉到行为/措辞/样本层（条件注入 + 真人原话）。
6. **先筛选后分析**：从素材提炼前先判噪声（礼物/寒暄/转述），只提炼高质量话轮。
7. **确定性兜底，只在模型默认习惯压不住时才上**：括号、省略号、"她"自指，都是提示词管不住之后靠输出后处理（clean_reply）兜底。
8. **每步产物先展示审批**：不一次性跑完流水线，验证通过再往前。

> 完整方法论 + 踩坑记录见 [`ROADMAP.md`](ROADMAP.md)（新会话接续必读）。

## 架构演进：从"读规则"到"建体系"

这个项目是一步步进化来的——每一版都在解决一个真实暴露的问题，而不是为了加功能而加。

### V1 — 样本 few-shot
发现"读规则学说话"不行 → 改为"看样本学说话"。精选真实对话样本注入，替代提示词里的行为描述。

### V2 — 砍提示词
系统提示词从 318 行/3.2 万字砍到 40 行/3500 字。删掉数字配额与语气词详解，只保留核心人格。实测灵性明显提升——**提示词不是越多越好**。

### V3 — 多路融合检索
corpus（直播记忆）/声音样本/行为触发/措辞指纹四路向量检索 + RRF 加权融合 + 预算截断；后续版本加入偏好/核心记忆，现为六路检索。全程只调 1 次 embedding。这是"按需注入"的雏形。

### V3.1–V4 — 措辞指纹库 + 数据扩充 + 参数调优
"同一意思 → 她真实说过的原话"（被夸→"也没有啦"）；样本库 14→56 条全场景均衡；corpus 273 条；承诺记忆（跨会话记住答应过的事）。

### V5 — 感知增强 + 真人节奏
时间/农历/天气感知（按用户记城市）；glm-4.6v 看图（图片→描述注入）；B站开播/动态联动推送；读秒窗口 + 分批发送（真人打字感）。

### V5.1–V5.5 — 人格工程收敛
偏好第 5 路语义检索、核心记忆层、会话级记忆（话题追踪）、行为规则样板化（带真人示范）、表情消息处理、身份分层（人设 × 真实经历）、符号纪律、去标签化。测试从 150+ → 160+。

### 2026-08 — 从"调参数"到"建体系"
这一阶段不再是加数据，而是**承认 embedding 的边界，把判断权交给 LLM，并给调参建评测**：

- **行为 L3（LLM 意图分类）**：embedding 按句式聚团，把"灰泽满你唱歌好听"（夸）和"灰泽满你怎么又迟到"（质问）这类同句式消息挤在一起，余弦匹配把夸奖误判成质疑。治本：行为归属从 embedding 改为 LLM 判定 + 判别词兜底。
- **corpus 关键词门**：问句 vs 陈述式嵌入有天然鸿沟，纯 cosine 无法用单一阈值同时区分"相关 0.51"和"无关 0.62"。加区分性词重叠门，阈值才敢降。
- **记忆健壮性**：null 字符串污染根治、提取 JSON 修复 + 重试，记忆不再漏记/污染。
- **检索评测工具**：建标注集量化每路命中率，**把阈值调参从"人肉肉眼"变成可测量的回归**。

## 技术亮点

- **分层注入架构**：10 层条件注入（人设/行为/corpus/记忆/会话/措辞/样本/感知），每层管一件事，出问题能定位到具体层。
- **检索评测体系**：29 条标注集 + `retrieval_eval.py`，量化每路命中率，任何阈值/样本改动可回归验证。
- **人格一致性评测**：InCharacter 式大五人格开放题 + 匿名化防名字作弊，实测实名/匿名都不掉分。
- **记忆系统**：短期（5 轮）/长期（用户画像 + 承诺）/会话级（话题追踪）三层，提取带 JSON 修复 + 重试。
- **工程纪律**：207 个单测；方法论 + 踩坑记录持续沉淀进 ROADMAP。

## 技术栈

| 模块 | 技术 |
|:---|:---|
| 机器人框架 | NoneBot2 + OneBot V11 |
| QQ 接入 | NapCatQQ |
| 对话模型 | DeepSeek-V4-Flash |
| 视觉模型 | 智谱 glm-4.6v（关思考模式，秒级描述） |
| Embedding | 智谱 embedding-3 |
| 天气 | 和风天气（按用户记城市） |
| 农历/节日 | lunar-python（本地） |
| B站联动 | bilibili-api-python + 直播公开接口 |
| 语音转写 | faster-whisper（离线蒸馏工具链） |
| 数据存储 | JSON（记忆、向量、人格数据） |

## 快速开始

### 前置要求
- Python 3.10+、QQ 账号（NapCatQQ）、DeepSeek API Key、智谱 AI API Key、本地 GPU（可选）

### 安装与配置
```bash
git clone https://github.com/MureasAm/Hzm-AI-Bot.git
cd Hzm-AI-Bot
pip install -r requirements.txt
```
创建 `.env.prod`（DeepSeek / 智谱 / 和风 / B站 / 视觉等配置），NapCat 登录机器人 → `python bot.py`。

## 人格蒸馏流水线（离线工具箱）

从直播素材到人格数据的全链路：`python scripts/run_tool.py <工具>` 统一入口（16 子命令按阶段分组）。

```
【蒸馏】transcribe → clean-transcript → analyze-pace → convert-to-chat → mine-phrases
【生成】generate-statements → generate-vectors → generate-persona → extract-persona
【向量】precompute
【评测】regression / persona-eval / retrieval-eval
【工具】bili-check / vision-test / mine-theme
```

产物按阶段落盘到 `outputs/` 对应文件夹（transcribe/clean/pace/convert/mine/statements/eval），最终源数据 `persona/world/statement_final.json` → `persona/world/corpus_vectors.json`。

> 改过 `persona/behavior/behaviors.json` / `persona/speech/voice_samples.json` / `persona/speech/phrases.json` 等后需重跑对应 `precompute`。

## 项目结构

```
bot.py                       # 启动入口
src/plugins/chatbot/         # 运行时核心
├── core.py                  # 主循环（组装 + 生成 + 记忆更新）
├── reply_style.py           # 回复风格后处理（clean_reply / split_reply / 复读检测）
├── routing.py               # 硬匹配路由（梗库双路由 + 行为意图分类 L3）
├── retrieval.py             # 六路检索 + RRF 融合 + 预算截断 + 关键词门
├── config.py                # 配置读取 + API 客户端工厂
├── persona.py               # 人格数据加载 + trigger 向量缓存 + terms
├── memory.py                # 短期记忆 + 长期封装
├── session_memory.py        # 会话级记忆（话题追踪 + 指代性消息补全）
├── context_probe.py         # 时间/农历/天气感知
├── vision.py                # glm-4.6v 看图
├── chat_window.py           # 读秒窗口 + 分批发送
├── bili_bridge.py           # B站开播/动态联动推送
└── constants.py             # 所有可调参数 ★
persona/                     # 角色人格 + 她的记忆（按分层职责分子目录，源文件 + 向量放一起）
├── core/                    #   核心人设：system_prompt + traits + styles（她是谁）
├── behavior/                #   行为：behaviors + trigger_vectors（怎么反应）
├── speech/                  #   说话：voice_samples(83)+phrases(11) + 向量（怎么说）
└── world/                   #   世界：terms(25)+core_stories(5)+preferences(24)+corpus(315)（经历/懂什么）
user_memory/                 # 对用户的记忆
├── short_term.json          #   短期（最近 5 轮原文）
├── long_term.json           #   长期（用户画像 + 承诺）
└── session.json             #   会话级（当前话题 + 本场事件）
outputs/                     # 分析产物（按阶段分文件夹，gitignore）
assets/                      # 原始素材（音频）
data/                        # 运行时状态：bili_state.json（B站联动）
scripts/                     # 离线工具箱 + 评测（run_tool.py 统一入口）
tests/                       # 194 个单元测试
```

## 结语

这个项目想证明的是：**让 LLM 稳定还原一个真实人物的说话方式，是可以被系统性地工程化的。** 核心不在于模型多聪明，而在于数据蒸馏、检索、记忆、评测这套体系的构建——以及大量的、可复现的试错。

如果你在看这份 README 时想到"这不就是一个聊天机器人吗"——那它还没讲好。它更准确的身份是：**一个关于「LLM 角色一致性」这个开放问题的工程答卷**。

---

*完整重构历程、方法论、踩坑记录见 [`ROADMAP.md`](ROADMAP.md)。*
