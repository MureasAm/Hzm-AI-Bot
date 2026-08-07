# 灰泽满 AI 机器人 · 开发路线图 & 会话交接

> 本文件由 Claude Code 在 2026-08 协作中生成。第一部分是此前会话的交接摘要（供新窗口续接），
> 第二部分是未来优化路线。新开会话时让 Claude 先读本文件即可无缝接上。

---

## 第一部分：会话交接摘要

### 这个项目是什么

基于 **NoneBot2 + OneBot V11 + NapCatQQ** 的虚拟主播"灰泽满"QQ 聊天机器人。核心目标不是"通用聊天"，而是**高度还原主播的语言风格、行为模式与情感表达**。用户从"对 AI 完全不了解"起步，在 DeepSeek 网页辅助 + 大量手动调试下独立完成——这是兴趣驱动的结晶，也是用户希望做成"最好的展示"的作品。

### 此前的相关会话做了什么（事件概括）

这段工作围绕 **paper-agent 项目**（`D:\paper_agent`，一个学术写作 Agent）展开，同时评估了本 QQ 项目。主要事件：

1. 从零构建学术 Agent：DeepSeek V4 + LangChain `create_agent` + Gradio 前端，两个工具（Semantic Scholar 检索 / python-docx 文档生成）
2. 实现**长期记忆**（用户画像 JSON + 规则式兜底 + LLM 提取 + 文件锁防并发竞态）
3. 实现**复杂任务状态机**（确认需求→大纲→逐节→整合），并发现"纯提示词无法强制分步"→ 改为代码级控制
4. 实现**两级意图路由**（强信号规则 + 弱信号 LLM + 否定守卫），配 20 条边界测试 + 52 条量化基准（两级路由 96.2% 准确率 / 83ms，纯 LLM 90.4% / 503ms）
5. 实现**引用审计**（检索白名单 + 正则提取 + 模糊匹配）
6. 拆模块（单文件 → 8 模块包）、补测试、README 叙事（V1→V6 演进）
7. 修复 Gradio 6 的 history 格式 bug（MessageDict vs tuple）
8. 解析了 ARS（academic-research-skills）skill 系统，理解了"同一 LLM 扮演多角色的角色协作"范式
9. 客观评估了 QQ 机器人项目，识别出它"像 Agent 但缺 ReAct"

### 沉淀的关键技术知识（对本项目直接可用）

- **DeepSeek V4**：模型名是 `deepseek-v4-flash` / `deepseek-v4-pro`（**没有** `deepseek-chat`/`deepseek-reasoner`——本项目代码里用的 `deepseek-chat` 需要更新）。思考模式默认开启；思考模式下 `temperature` 等参数不支持；禁用思考用 `extra_body={"thinking":{"type":"disabled"}}`
- **Gradio 6**：ChatInterface 没有 `theme` 参数；传给回调的 `history` 是 MessageDict（`[{"role":..,"content":..}]`），不是 `[[user,bot]]`
- **Agent 范式**：Agent = LLM + 记忆 + 工具 + 规划循环；ReAct 的本质是"工具调用结果回填上下文、继续推理"
- **可迁移的工程经验**（用户已在 paper-agent 中掌握）：
  - 记忆持久化的**文件锁**（多线程快照互相覆盖的坑）
  - **意图路由**思路（区分不同意图 → 路由到不同处理）——可用于区分"聊天"和"动作请求"
  - **量化评估**习惯（测试 + 基准数据证明设计，而非"我觉得"）
  - 工程化习惯（拆模块、补测试、魔法数字常量化）

---

## 第二部分：当前项目分析

### 架构现状

```
bot.py                           # NoneBot2 入口
src/plugins/chatbot/             # 核心对话逻辑（模块化）
├── __init__.py                  # 插件入口
├── constants.py                 # 路径 / API / 阈值常量
├── persona.py                   # 人格规则 + 语义匹配（trigger 向量缓存）
├── memory.py                    # 短期记忆（带锁）+ 长期记忆封装
├── rag.py                       # 直播记忆 RAG（余弦相似度）
└── core.py                      # 主循环
persona/                         # 人格数据（system_prompt.txt + persona_*.json）
data/                            # 运行数据（memory.json / corpus_vectors.json / trigger_vectors.json 等）
outputs/                         # 分析产物（persona_analysis.md / 提示词草案）
scripts/                         # 离线工具（蒸馏流水线 / 向量生成 / 转写）
prompts/                         # 提示词模板
materials/                       # 素材备份
bin/                             # ffmpeg 等二进制
memory_manager.py                # 长期记忆: 记忆卡/关系等级/LLM提取（带文件锁）
tests/                           # 核心函数单元测试（pytest）
```

### 已知问题（按优先级）

| # | 问题 | 影响 | 状态 |
|---|------|------|------|
| 1 | `match_behaviors_semantic` 每条消息对**每个 trigger 调一次 embedding API** | 每消息 N 次 API 调用，慢且贵 | ✅ 已解决：`precompute_trigger_vectors.py` 离线预计算 → `trigger_vectors.json`，运行时只对 query 调 1 次 |
| 2 | 记忆写入（memory.json / long_term_memory.json）**无文件锁** | 多用户并发可能竞态覆盖 | ✅ 已解决：`memory_manager.py` 与 `memory.py` 的读-改-写均加 `threading.Lock` |
| 3 | 模型名 `deepseek-chat` | DeepSeek 已无此模型名，可能失效 | ✅ 已解决：全部更新为 `deepseek-v4-flash`，聊天/记忆调用禁用思考模式保留 temperature |
| 4 | 核心逻辑 349 行集中在 `__init__.py` + 魔法数字 | 难维护 | ✅ 已解决：拆成 `constants/persona/memory/rag/core`，常量集中定义 |
| 5 | 无测试 | 不可验证 | ✅ 已解决：`tests/` 29 条核心函数单测（cosine / memory 合并 / RAG 过滤 / 行为匹配） |

---

## 第三部分：愿景与路线

### 终极愿景

让"灰泽满"从一个"会聊天的 bot"成长为**能行动、能感知、能联动的虚拟主播 Agent**——做到用户本人无法实现的效果。包括：联网了解近期事件、主动发消息、多模态看图、通过 B 站关联她发动态/直播并自动联动。

### 路线 A：工程化（已完成 ✅）

1. **embedding 缓存** ✅：`precompute_trigger_vectors.py` 离线预计算 trigger 向量 → `trigger_vectors.json`；`persona.match_behaviors_semantic` 运行时读缓存，query 向量在 core 中只算 1 次，供行为匹配与 RAG 共用
2. **记忆文件锁** ✅：`memory_manager.py` 读-改-写全程持 `threading.Lock`，并抽出纯函数 `merge_memory_card`（便于测试）；`memory.append_user_history` 同样持锁
3. **更新模型名** ✅：全部 `deepseek-chat` → `deepseek-v4-flash`（`.env.prod` / README / 各脚本）；聊天与记忆提取调用加 `extra_body={"thinking":{"type":"disabled"}}`，保留 temperature 参数
4. **拆模块** ✅：`src/plugins/chatbot/` → `constants.py`（路径/API/阈值集中）+ `persona.py`（人格加载与匹配）+ `memory.py`（短/长记忆）+ `rag.py`（检索）+ `core.py`（主循环，API 客户端惰性初始化）；`__init__.py` 只做插件入口
5. **常量集中 + 测试** ✅：`tests/` 29 条 pytest 单测覆盖 cosine、`merge_memory_card`（去重/关系升级/不改输入）、RAG 阈值过滤、行为匹配阈值

**使用提醒**：改了 `persona_behaviors.json` 后需重跑 `python precompute_trigger_vectors.py` 更新缓存。

### 路线 B：Agent 化（进阶，约 4-6 天）

**目标**：给机器人加"行动层"，从对话系统升级为 Agent。

1. **定义工具集**（NapCatQQ 均支持）：
   - `send_message`（主动发消息）
   - `search_web`（联网查事件）
   - `send_image`（发图）
   - `set_reminder`（定时提醒）
   - `check_bilibili`（查主播动态/直播状态）
2. **ReAct 循环**：手写或复用 LangChain。流程：识别用户请求 → 判断是否需要工具 → 调用 → 结果回填上下文 → 继续回复
3. **意图区分**：复用 paper-agent 的两级路由思路，区分"聊天"（直接生成）和"动作请求"（触发工具）
4. **主动性**：定时任务（APScheduler）+ 事件监听（B 站状态变化）→ 主动触发 `send_message`

### 扩展能力（可与 B 并行）

- **多模态看图**：QQ 收图 → 下载 → 传给视觉模型。**前置：确认 DeepSeek V4 是否支持视觉**；不支持则接 GLM-4V / Qwen-VL（需额外 key）
- **B 站联动**：轮询主播直播状态/最新动态，变化时自动发 QQ 消息。NapCatQQ 寄居 QQ 客户端内部、走客户端能力，**QQ 侧无独立风控**（QQ 群里主播开播自动联动是常见 bot 功能，已验证可行）。主要注意 B 站非官方 API 的稳定性：**低频轮询（如每 3-5 分钟）+ 失败静默降级**

---

## 第四部分：风险与注意

- **B 站 API**：非官方接口，签名规则可能变化 → 容错设计，轮询频率克制
- **QQ 消息频率**：虽无独立风控，但高频自动消息仍不友好 → 主动消息加节流
- **多模态**：DeepSeek V4 视觉能力是前置条件，需先验证
- **改动范围**：路线 B 动的是"消息处理主循环"，改前先备份，逐步替换

---

*由 Claude Code 生成于 2026-08 · 供新窗口续接参考*
