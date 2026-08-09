# 灰泽满 AI 机器人 · 开发路线图 & 会话交接

> 本文件由 Claude Code 在 2026-08 协作中生成，供新窗口续接。**新开会话时先读本文件**即可无缝接上。
> 关键：本文件记录了完整重构历程、方法论、工具链与踩坑记录，比代码本身更能帮你快速进入状态。

---

## 第一部分：这个项目是什么

基于 **NoneBot2 + OneBot V11 + NapCatQQ** 的虚拟主播"灰泽满"QQ 聊天机器人。核心目标不是"通用聊天"，而是**高度还原主播的语言风格、行为模式与情感表达**。用户希望做成"最好的展示"。

**技术栈**：DeepSeek-V4-Flash 对话 + 智谱 embedding-3 向量 + faster-whisper 转写 + JSON 存储。

---

## 第二部分：版本更迭史

### V1（2026-08）：样本 few-shot + 停用记忆污染
- 新建 `persona/voice_samples.json`（精选对话样本库），few-shot 注入，让模型"看样本学说话"而非"读规则学说话"
- 停用长期记忆 `new_self_fact` 提取——防止 AI 自嗨内容污染真实人格
- 建 `scripts/regression_test.py` 虚构弹幕 A/B 测试

### V2：砍提示词（318行 → 3500字）
- `persona/system_prompt.txt` 从 318 行/3.2万字 砍到 40 行/3500 字（原版备份 `system_prompt_v9_backup.txt`）
- 删掉全部数字配额与语气词详解，保留核心人格/说话节奏/自我称呼/行为反应/脆弱时刻
- 实测灵性明显提升；删除"按互动次数分关系等级"机制，默认当熟人

### V3：五路融合检索（按需注入）
- 新增 `retrieval.py`：corpus/voice_sample/behavior/phrase 四路向量检索 + RRF 加权融合 + 预算截断；长期/短期记忆两路确定性注入
- query 全程只调 1 次 embedding，四路共用；预计算向量缓存

### V3.1：措辞指纹库 + 样本分层 + 长度控制
- 新建 `persona/phrases.json` 措辞指纹库；voice_samples 分 `short/long` 档；新增 `analyze_pace.py` 节奏地图

### V4（2026-08-09~10）：补样本收官 + 措辞重建 + 参数调优 + 记忆增强
- **样本库 14 → 56 条**，8 场景全均衡（被戳穿10/日常8/被调戏8/失约7/被夸7/推进7/立Flag5/感性4）
- **措辞库重建为 11 组**：全部从素材提取（新工具 `mine-phrases`），废弃手工编的"被作业封印了"等
- **corpus 更新为 273 条**：5 场新素材场景化陈述 + 旧 73 条，修错字、剔乱码
- **记忆增强**：长期记忆加"承诺/约定"（`new_promise`）跨会话记忆；短期记忆一致性规则修复
- **参数定稿**（见第三部分）：`CHAT_TEMPERATURE=0.85`、`CHAT_FREQUENCY_PENALTY=0.3`、`CORPUS_TOP_N=2`、`SHORT_MEMORY_LINES=10`

---

## 第三部分：回复生成链路剖析（每一层在决定什么）

> 排查回复问题时，先定位是哪一层的锅：腔调（样本）？用词（措辞）？行为模式（behaviors）？还是人设定调（prompt）。**素材多样性问题从素材层解决，不要用提示词打补丁**（历史教训：提示词堆砌作用很差）。

### 消息组装顺序 = 注入优先级（build_message_list）

| 顺序 | 来源 | 注入标签 | 决定什么 |
|---|---|---|---|
| 1 | system_prompt.txt + traits/styles | 人设 | 她是谁、人格底线、节奏、自称、括号规则 |
| 2 | behaviors（trigger 命中） | 【当前情境下的行为指令】 | 该情境的明确反应模式（权重最高 1.5） |
| 3 | corpus（直播记忆） | 【她经历过的相关背景】 | 背景记忆，仅相关时提及，不参与风格 |
| 4 | 长期记忆 | 【关于这个绿冻的长期记忆】 | 这个用户是谁（印象/事实/承诺/时刻） |
| 5 | 短期记忆 | 【最近对话记录】 | 最近聊了什么（+一致性规则） |
| 6 | phrases（措辞指纹） | 【她的固定说法】 | 表达同类意思时的真实用词 |
| 7 | voice_samples（few-shot） | 【说话方式参考】 | 示范她的腔调（语气/断句/自称/节奏） |
| 8 | 长度提醒 | 【回复节奏】 | 一句话就停，30字内 |
| 9 | user_msg | — | 当前用户消息 |

### 各文件作用

**persona/**：`system_prompt.txt`（人设核心）/ `persona_traits.json`（性格基底）/ `persona_styles.json`（语言风格）/ `persona_behaviors.json`（触发行为，改后 precompute triggers）/ `voice_samples.json`（样本，改后 precompute voice-samples）/ `phrases.json`（措辞，改后 precompute phrases）

**data/**：`corpus_vectors.json`（直播记忆 RAG）/ `trigger_vectors.json` / `voice_sample_vectors.json` / `phrase_vectors.json` / `memory.json`（短期）/ `long_term_memory.json`（长期）

**src/plugins/chatbot/**：`core.py`（主循环组装）/ `persona.py` / `memory.py`（短期读写带锁）/ `rag.py`（embedding+余弦）/ `retrieval.py`（四路融合+预算）/ `constants.py`（所有可调参数）

**memory_manager.py**：长期记忆卡 merge + build_memory_context + 记忆提取 prompt

### 最终参数（constants.py）

```python
CHAT_TEMPERATURE = 0.85        # 0.7 会让模型走 RP 默认括号模板 → 括号爆炸（踩过坑）
CHAT_FREQUENCY_PENALTY = 0.3   # 0.5 无效还引入整句重复（踩过坑）
CORPUS_TOP_N = 2               # 长文本少占 few-shot 预算
SHORT_MEMORY_LINES = 10        # 5 轮，一致性
RETRIEVAL_BUDGET_CHARS = 1200
VOICE_SAMPLE_TOP_N = 3         # few-shot 甜蜜点 2-5
```

### 检索层
1 次 embedding → 四路检索 → RRF 融合（behavior 1.5 > phrase 1.2 > corpus/voice 1.0）→ topk → 预算截断。

---

## 第四部分：记忆系统设计

| 层 | 内容 | 注入 |
|---|---|---|
| **短期**（memory.json，5轮） | 最近对话原文 | 【最近对话记录】+ 一致性规则 |
| **长期**（long_term_memory.json） | 印象标签/用户事实/重要时刻/**承诺约定** | 【关于这个绿冻的长期记忆】 |

- **self_fact 停用**：V1 起不提取 AI 自嗨的自我披露，防污染真实人格
- **承诺记忆**（V4 加）：`new_promise` 跨会话记住她答应过用户的事（"明天一定""这周不鸽"）
- **一致性规则**（V4 修）：解释同一件事借口一致，但新问题正常回答（不绑架旧借口）
- 记忆提取异步、带文件锁

---

## 第五部分：离线工具箱（run_tool.py）

统一入口：`python scripts/run_tool.py <工具> [参数]`

| 子命令 | 干什么 |
|---|---|
| transcribe | 音频 → 原始转写（faster-whisper GPU） |
| clean-transcript | 转写清洗（繁体转简体 + 固定错字表） |
| convert-to-chat | 直播 → 聊天（V3 五步：分离转述/回答 → 提特征 → 切分 → 压缩） |
| analyze-pace | 节奏地图（`--focus` 聚焦 2-3 场景 + reasoning） |
| mine-phrases | 从素材批量挖措辞指纹（多维度，输出待审批） |
| generate-statements | 从素材生成场景化陈述（50-120字，供 corpus RAG） |
| generate-vectors | 场景化陈述 → corpus 向量库 |
| generate-persona | 场景化陈述 → 人格三件套 |
| precompute | 预计算 trigger/声音样本/措辞向量 |
| pipeline | 四步人格蒸馏流水线 |
| regression | 回复灵性 A/B 回归测试 |

---

## 第六部分：方法论（反复验证后确立）

1. **样本 > 规则**：真实对话示范"怎么说话"，比规则规定有效
2. **素材层解决 > 提示词打补丁**：措辞/括号/重复问题从素材解决；提示词硬约束是堆砌，无效且走老路
3. **直播 ≠ 聊天**：必须经"转聊天"转化，且**切分压缩**（一个独立意思 = 一条 15-50 字短回复），不能整段保留
4. **单人声音素材要分离"读弹幕 vs 回答"**：转述的粉丝话是 user，她的回答才是 reply
5. **先筛选后分析**：先判噪声（礼物/寒暄/转述/看二创），只提炼高质量话轮
6. **灰泽满永远自称"灰泽满/hzm"**，绝不用"我"（主语宾语都如此）
7. **括号只在情感顶点用**：是例外不是默认；温度靠语气词/自嘲/省略号，不靠括号
8. **措辞必须从素材提取**：不手工编造
9. **每步产物先展示审批**，验证通过再往前

---

## 第七部分：踩坑记录（新会话务必避免）

- ❌ 全量分析转写（55% 噪声）→ 先判噪声
- ❌ "被催播"命名错 → 应叫"失约被抓包"
- ❌ prompt 点名口癖 → 模型每条都加；应"原文有什么留什么"
- ❌ **temperature 降到 0.7 治"重复" → 反而括号爆炸**（低温度让模型走 RP 默认模板）。0.85 反而括号少
- ❌ **frequency_penalty 0.5 压括号 → 没压住还引入整句重复**。回退 0.3
- ❌ 提示词加"别老用封印" → 堆砌覆辙，撤销，从措辞库多样化解决
- ❌ "保持温度"提示词 → 模型理解成"用括号表达温度"；应明确"温度靠语气/自嘲/省略号"
- ❌ 手工编措辞（"被作业封印了"）→ 非她原话，废弃，从素材提取
- ❌ 一次性跑完流水线不给中间产物 → 每步先展示审批
- ❌ 短期记忆持久化污染 → 测试对话会积累进 memory.json，影响后续（测试后注意清空）
- ❌ "借口一致性强制规则"过度 → 旧借口绑架新问题（"泡面"）；应限定"同一件事一致，新问题正常答"
- ❌ 一次性把 statement 全部向量化 → 应先生成→筛选修错字→再向量化（"代属国""婚姻买""满居"等错译）

---

## 第八部分：终极愿景（路线 B：Agent 化）

让"灰泽满"成长为**能行动、能感知、能联动的虚拟主播 Agent**：
- **联网**：search_web 了解事件，动态回应时事
- **多模态**：send_image 看图、识别图片内容
- **主动性**：主动发消息、set_reminder 提醒、check_bilibili 联动她发动态/直播
- 架构：工具集（send_message/search_web/send_image/set_reminder/check_bilibili）+ ReAct 循环 + 意图区分 + 主动性
- 当前人格系统（素材驱动 + 融合检索）是 Agent 化的地基——**建议进入此阶段时开新会话**（新会话读本文件接上）

---

## 第九部分：风险与注意

- **DeepSeek V4**：`deepseek-v4-flash`；聊天调用需 `extra_body={"thinking":{"type":"disabled"}}` 保留 temperature
- 改 `persona_behaviors.json` / `voice_samples.json` / `phrases.json` 后需重跑对应 precompute
- 改主循环前先备份（core.py 有 backup_test1）
- `.env.prod` 含 API key，已在 .gitignore
- 素材质量决定样本上限——先快速判定场次价值

---

*由 Claude Code 生成于 2026-08-10 · 供新窗口续接参考*
