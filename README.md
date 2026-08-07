# 灰泽满 AI 聊天机器人

基于 **NoneBot2** 框架构建的虚拟主播“灰泽满”AI 聊天机器人，部署于 QQ 平台。通过分层人格引擎、长期记忆系统和深度人格蒸馏，高度还原主播的语言风格、行为模式和情感表达。

## 核心特性

- **分层人格引擎**：经典梗硬匹配 → 基础人设提示词 → 人格规则检索 → 直播记忆 RAG → 用户记忆，五层优先级逐级兜底
- **长期记忆系统**：记住用户身份、聊天历史与关系深度，随互动次数自动升级关系等级并调整语气
- **人格蒸馏流水线**：从直播切片 → 语音转写 → 场景化陈述 → 向量化 + 人格 JSON，全流程半自动化
- **RAG 直播记忆**：基于智谱 Embedding 和余弦相似度，从 80+ 条场景化陈述中检索相关记忆
- **实时语气控制**：揉眼睛限制、心虚限制、括号频率控制，根据关系等级动态调整

## 技术栈

| 模块 | 技术 |
|:---|:---|
| 机器人框架 | NoneBot2 + OneBot V11 |
| QQ 接入 | NapCatQQ |
| 对话模型 | DeepSeek-V4-Flash |
| Embedding 模型 | 智谱 AI embedding-3 |
| 语音转写 | faster-whisper (medium, GPU) |
| 数据存储 | JSON (记忆、向量、人格规则) |

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

2.安装依赖
pip install -r requirements.txt

3.配置环境变量
在项目根目录创建 .env.prod 文件：
OPENAI_API_KEY=你的DeepSeek_API_Key
OPENAI_API_BASE=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-v4-flash
ZHIPU_API_KEY=你的智谱AI_API_Key

4.在NapCat里进行机器人登录

5.启动机器人
python bot.py

语音转写（可选）
如果你需要从直播录播生成转写文本：
scripts/run_whisper.bat

> 所有离线脚本统一从项目根目录执行，例如 `python scripts/generate_vectors.py`。

项目结构
├── bot.py                     # 项目启动入口
├── src/plugins/chatbot/       # 核心对话逻辑（模块化）
│   ├── __init__.py            # 插件入口（注册消息处理器）
│   ├── constants.py           # 路径 / API / 阈值常量集中
│   ├── persona.py             # 人格加载 + 语义行为匹配（trigger 向量缓存）
│   ├── memory.py              # 短期记忆（带锁）+ 长期记忆封装
│   ├── rag.py                 # 直播记忆检索（余弦相似度）
│   └── core.py                # 消息处理主循环
├── persona/                   # 人格数据
│   ├── system_prompt.txt      # 基础人设提示词
│   ├── system_prompt_upgraded.txt  # 蒸馏升级版提示词
│   ├── persona_traits.json    # 稳定性格特质
│   ├── persona_styles.json    # 语言风格规则
│   └── persona_behaviors.json # 触发式行为规则
├── data/                      # 运行数据与向量库
│   ├── memory.json            # 用户短期记忆
│   ├── long_term_memory.json  # 用户长期记忆
│   ├── corpus_vectors.json    # 直播记忆向量库
│   ├── trigger_vectors.json   # 行为 trigger 向量缓存（预计算生成）
│   ├── input_transcript.json  # 直播转写输入
│   ├── cleaned_corpus.json    # 清洗后语料
│   ├── qa_pairs.json          # 蒸馏 QA 对
│   └── personality_rules.json # 人格规则存档
├── outputs/                   # 分析产物
│   ├── persona_analysis.md    # 人格切面分析
│   └── system_prompt_suggestion.md  # 提示词草案
├── scripts/                   # 离线工具脚本（统一从项目根执行）
│   ├── generate_vectors.py    # 场景化陈述 → 向量库
│   ├── generate_persona.py    # 场景化陈述 → 人格 JSON
│   ├── precompute_trigger_vectors.py  # 预计算 trigger 向量
│   ├── four_step_pipeline.py  # 四步人格蒸馏流水线
│   ├── four_step_pipeline.txt # 流水线说明备份
│   ├── transcribe_whisper.py  # faster-whisper 转写
│   ├── transcribe_tencent.py  # 腾讯云语音转写
│   └── run_whisper.bat        # 一键 GPU 转写脚本
├── prompts/                   # 提示词模板
│   ├── prompt_step1.txt       # 蒸馏四步的提示词模板
│   ├── prompt_step2.txt
│   ├── prompt_step3.txt
│   ├── prompt_fusion.txt
│   └── 基于直播内容生成的描述.txt 等（场景化描述提示词）
├── materials/                 # 素材与备份
│   ├── 人格备份.txt
│   └── 灰泽满动态合集.txt
├── bin/                       # 二进制工具
│   ├── ffmpeg.exe / ffplay.exe / ffprobe.exe
├── memory_manager.py          # 长期记忆管理（带文件锁）
└── tests/                     # 核心函数单元测试（pytest）

人格蒸馏流水线
收集素材：直播录播/动态

语音转写：scripts/run_whisper.bat → faster-whisper GPU 加速

生成场景化陈述：通过 DeepSeek + 专用提示词

向量化入库：scripts/generate_vectors.py → data/corpus_vectors.json

人格分析：scripts/generate_persona.py → 更新 persona/ 下三个 JSON

预计算 trigger 向量（改过 persona_behaviors.json 后需重跑）：
scripts/precompute_trigger_vectors.py → data/trigger_vectors.json

长期记忆设计
为每位用户维护独立的记忆卡片，存储：

印象标签（如“上班族”、“喜欢催播”）

用户事实（如“在准备考研”）

自我披露记录（避免 AI 重复自曝）

关系等级（stranger → acquaintance → familiar → close）

关系等级根据互动次数自动升级，并动态调整 AI 的语气、括号使用频率和防御姿态

说明
本项目仅供学习和交流使用
所有 API 密钥均通过 .env.prod 管理，已加入 .gitignore
人格数据来源于公开直播内容的二次创作