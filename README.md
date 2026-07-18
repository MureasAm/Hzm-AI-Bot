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
| 对话模型 | DeepSeek-Chat |
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
OPENAI_MODEL=deepseek-chat
ZHIPU_API_KEY=你的智谱AI_API_Key

4.在NapCat里进行机器人登录

5.启动机器人
python bot.py

语音转写（可选）
如果你需要从直播录播生成转写文本：
run_whisper.bat

项目结构
├── bot.py                     # 项目启动入口
├── src/plugins/chatbot/       # 核心对话逻辑
│   └── __init__.py            # 分层人格引擎 + RAG + 记忆
├── persona/                   # 人格数据
│   ├── system_prompt.txt      # 基础人设提示词
│   ├── persona_traits.json    # 稳定性格特质
│   ├── persona_styles.json    # 语言风格规则
│   └── persona_behaviors.json # 触发式行为规则
├── memory_manager.py          # 长期记忆管理
├── generate_vectors.py        # 场景化陈述 → 向量库
├── generate_persona.py        # 场景化陈述 → 人格 JSON
├── transcribe_whisper.py      # faster-whisper 转写
├── run_whisper.bat            # 一键 GPU 转写脚本
├── memory.json                # 用户短期记忆
├── long_term_memory.json      # 用户长期记忆
└── corpus_vectors.json        # 直播记忆向量库

人格蒸馏流水线
收集素材：直播录播/动态

语音转写：run_whisper.bat → faster-whisper GPU 加速

生成场景化陈述：通过 DeepSeek + 专用提示词

向量化入库：generate_vectors.py → corpus_vectors.json

人格分析：generate_persona.py → 更新 persona/ 下三个 JSON

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