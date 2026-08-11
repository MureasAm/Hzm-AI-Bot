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

### V5（2026-08-10）：感知增强 + 节奏系统（路线B第一弹）
- **感知**：`context_probe.py` 时间/农历/节气/天气（lunar-python 本地 + 和风专属域名，默认墨尔本=灰泽满所在地，**按用户记城市**——对话提"我在广州"→ 长期记忆 `weather_city` → 注入该用户城市天气）
- **视觉**：`vision.py` glm-4.6v 图片理解（Pillow 统一转 JPEG 兼容 WEBP/透明图 + `thinking:disabled` 提速 ~9s→1.7s；QQ 图链带 UA 下载）
- **B站联动**：`bili_bridge.py` 开播翻转检测 + 动态 ID 去重 → **结构化通知推送**（开播带直播间网址、动态带原文+个人空间链接+配图），白名单过滤，状态存 `data/bili_state.json`
- **节奏**：`chat_window.py` 读秒窗口（攒批，静默 5~10s 随机）+ 分批发送（`split_reply` 按句拆 + `split_delay` 按长度随机延迟）+ 智能归纳（`summarize_batch` 攒批先理解整体）
- **括号根治**：样本去滥用括号（留 3 条情绪顶点示范）+ `clean_reply` 输出清洗（剥开头前缀、至多 1 个、拆段后再清）
- 测试：117 个单测全过

### V5.1（2026-08-10）：人格检索分层 + 数据扩充
- **偏好第 5 路**：`persona/preferences.json` 结构化条目（一条一话题，text 向量化）→ `retrieve_preferences` 语义检索（复用 query embedding，**不进 RRF 融合/不占预算**，阈值 0.55，命中才注入【灰泽满的偏好】+ 冲突裁决"与语料/记忆冲突以偏好为准"）
- **核心记忆层**：`persona/core_stories.json`（印象最深的结晶，独立于 corpus，阈值 0.42 低更容易浮现）→ 注入【她的核心记忆】
- **梗库双路由**：`LEGENDARY_REPLIES` 关键词粗筛 + LLM 语境确认（`LEGENDARY_CONFIRMS` 按梗区分确认模板，含带上下文的"爱/感情"确认）；跟进句锚定感情线（"爱不爱→没感觉出来"走感情梗，不再跑偏到声卡话题）
- **数据扩充**：偏好 22 条（食物/水果/果冻/饮料/雨天/看书/电话/动物/瓜类/运动/作家/番剧/游戏…一条一话题）+ 核心记忆 5 条（三好学生 38票 / 222名 / 虚拟深圳上学 / 作家梦《乌色月》含原文）
- **split_reply 重做**：逗号限制（每段 ≤1 逗号，超了从最后一个逗号拆）+ 省略号保护（"啊……这……"无语不切，停顿边界才切）
- **修复**：动态 OPUS 类型提取（文本在 major.opus.summary.text、图在 pics）、推送强制刷好友（当天新好友立即收到）、自动通过好友申请（NapCat 给单向好友发消息是已知 bug，只能让它变双向）、梗匹配记入短期记忆
- 测试：128 个单测全过

### V5.2（2026-08-11）：会话级记忆 + 短 query 扩充 + 样本层修复
- **会话级记忆（episodic memory）**：新建 `session_memory.py`，记录"当前话题 + 本场关键事件"，转话题时更替；注入【当前会话】让模型接得住会话调性（对齐 HEMA compact memory / plast-mem episodic 层）
- **对话前同步探测**：`probe_session` 在**组装消息前**同步探测话题延续/转换 + 短 query 扩充（一次 LLM 调用同时完成），避免对话后异步更新导致的"转话题滞后一轮"——新主题下一轮立即生效
- **短 query 上下文扩充**：用户消息 ≤4 字（如"咋这样"）先在会话语境里扩充成完整句再 embedding——短向量太"糊"会误命中不相关样本（实测触发过"好女孩"误回复）
- **样本层修复**：`flirt_6/flirt_7` 的 user 描述"发不绿色的内容"过泛，被"比爱心/撒娇"语境误命中 → 改为"发黄段子/擦边/低俗弹幕"，重算向量后「咋这样」不再命中
- **记忆清理**：清除线上 memory.json / long_term_memory.json（测试对话污染）
- 新增 `tests/test_session_memory.py`；测试 141 个全过

### V5.3（2026-08-11）：行为规则样板化 + 人格提取工具重建
- **behaviors 格式改造**：`persona_behaviors.json` 每条加 `samples` 字段（真人原话示范，few-shot），`_format_behavior_rule` 注入时带"她这么说过（照着学腔调）"；触发-响应从"指令式"升级为"指令 + 示范"（对齐 Codified Profiles / forge-persona 的"行为>标签"）
- **提取代码重建**：新增 `scripts/extract_persona.py`（替代被删的 four_step_pipeline），吃 **convert-to-chat 产物**（已分离 user/reply 的对话对）→ 提取触发-响应行为 → **validation 铁律**：samples 必须逐字匹配 convert-to-chat 产物，模型改写/自创直接剔除
- **流程修正**：行为提取必须先 convert-to-chat 分离"读弹幕 vs 回答"，再 extract_persona——直接吃 cleaned 会把弹幕当 user（踩过）
- **检索阈值调优（切断"日常聊时间→注入直播样本"伪关联）**：实测日常短句与任意样本相似度天然 0.55-0.62、真直播命中 0.62-0.70。取中间值：`VOICE_SAMPLE_THRESHOLD` 0.35→**0.60**、`RAG_THRESHOLD` 0.35→**0.55**、新增 `VOICE_SAMPLE_KEEPALIVE_MIN_SIM=0.60`（保底也设门槛，宁断档不错话题）。案例：用户"9点是你那边"曾被注入直播时间样本 → 模型编"明天还是9点开播"；修后日常不再注入、真直播仍命中
- 测试 139 个全过

### V5.4（2026-08-11）：表情消息处理 + behaviors/voice 样本扩充
- **表情消息识别与处理**：纯表情消息（emoji / QQ表情码 `[表情：xx]`）走独立路径——① probe 扩充时**按表情标准含义识别**（😅=无语/尴尬、😭=委屈/哭，不从对话历史臆测，曾误判😅为"傲娇调侃"）；② **跳过五路语义检索**（表情不表达话题，😭曾被命中"被夸"样本）；③ 注入专门提示：按情绪回应并体现态度（无语→攻击性反击"感觉你不是很服气？"、委屈→心软安慰但**不套当前话题模板**，曾硬接"夸你可爱"）
- **behaviors samples 补充**（走"挑话轮→初筛→精修→convert→再筛→写入"流程）：冷场补"群冷场大王/直播间没人说话尴尬"4 条、感性补"此生无憾/打破网络世界妖魔化"2 条、被夸补"生日/反响"2 条、被质疑补"周表"2 条、立Flag补"早起"2 条；触发-响应从指令式升级为"指令+samples 真人示范"
- **voice_samples 扩充**：0807 生日场新增 birth_1~8 / peer_1~7 共 15 条（生日祝福反应 + 同行/前辈关系，填补原有缺口）
- **素材复用规则**：behaviors samples 尽量不复用 voice_samples 已有样本，避免同话题双注入（H1 同期祝福因已在 voice 而剔除）
- **B站动态配图下载修复**：根因是 bot 环境有失效代理 `HTTP_PROXY=127.0.0.1:50454`，httpx 默认 trust_env=True 走代理 → 图链下载 ConnectError。修法：`vision._read_image_bytes` 加 `trust_env=False` 强制直连 + Referer 改 B站（只影响图片下载，不影响 DeepSeek/智谱 API）
- **开播通知带直播封面**：`_fetch_live_status` 增加 `cover`（取 `cover_from_user` 直播封面，空回退 `keyframe` 关键帧）；开播时下载封面附在文字后推送（下载失败不阻塞仍发文字）
- 测试 141 个全过

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
| 5 | 会话级记忆（V5.2） | 【当前会话】 | 这场对话的话题线 + 本场发生的事（调性锚点） |
| 6 | 短期记忆 | 【最近对话记录】 | 最近聊了什么（+一致性规则） |
| 7 | phrases（措辞指纹） | 【她的固定说法】 | 表达同类意思时的真实用词 |
| 8 | voice_samples（few-shot） | 【说话方式参考】 | 示范她的腔调（语气/断句/自称/节奏） |
| 9 | 长度提醒 | 【回复节奏】 | 一句话就停，30字内 |
| 10 | user_msg | — | 当前用户消息 |

> **检索 query（V5.2）**：用户消息 ≤4 字时，先在会话语境里扩充成完整句再 embedding——短向量太"糊"会误命中不相关样本（踩过"好女孩"误回复）。见 `session_memory.expand_short_query`。

### 各文件作用

**persona/**：`system_prompt.txt`（人设核心）/ `persona_traits.json`（性格基底）/ `persona_styles.json`（语言风格）/ `persona_behaviors.json`（触发行为，改后 precompute triggers）/ `voice_samples.json`（样本，改后 precompute voice-samples）/ `phrases.json`（措辞，改后 precompute phrases）

**data/**：`corpus_vectors.json`（直播记忆 RAG）/ `trigger_vectors.json` / `voice_sample_vectors.json` / `phrase_vectors.json` / `memory.json`（短期）/ `long_term_memory.json`（长期）/ `session_memory.json`（会话级记忆：当前话题+本场事件，V5.2）

**src/plugins/chatbot/**：`core.py`（主循环组装）/ `persona.py` / `memory.py`（短期读写带锁）/ `session_memory.py`（会话级记忆：话题追踪+短query扩充，V5.2）/ `rag.py`（embedding+余弦）/ `retrieval.py`（四路融合+预算 + 偏好第5路 + 核心记忆检索）/ `constants.py`（所有可调参数）/ `config.py`（配置读取）/ `context_probe.py`（时间/天气感知）/ `vision.py`（glm-4.6v 视觉）/ `chat_window.py`（读秒窗口+分批发送+智能归纳）/ `bili_bridge.py`（B站联动推送 + 自动通过好友）

**persona/** 新增：`preferences.json`（偏好条目，一条一话题，第5路语义检索）/ `core_stories.json`（核心记忆，印象最深的结晶，低阈值高浮现）

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
- ❌ **样本 user 描述过泛 → 误命中**：`flirt_6`"发不绿色的内容"被"比爱心/咋这样"撒娇语境误命中，模型抄了"好女孩不能看这个"措辞答非所问 → user 描述要精确限定触发语境（"发黄段子/擦边"）
- ❌ **短 query 向量"糊"**：≤4 字消息 embedding 后与多个无关话题相似度都偏高，阈值 0.35 全放行 → 短 query 先在会话语境扩充成完整句再检索（V5.2 session_memory.expand_short_query）

---

## 第七部分B：生成 statement 的标准流水线（新窗口必读）

**目标**：把新直播场次的 `assets/audio/场次.mp3` 处理成 corpus 的 statement，合并进 `outputs/transcribe/statement_final.json`（corpus 源），再向量化。

**标准流水线**（全部用 `python scripts/run_tool.py`）：
```
1. transcribe        assets/audio/场次.mp3            → 场次_transcribed.json（原始转写）
2. clean-transcript  -i 场次_transcribed.json          → cleaned_场次.json（清洗：繁体→简体/错字）
3. analyze-pace      -i cleaned_场次.json --focus <场景> → 节奏地图（先判噪声，只留高质量话轮）
4. convert-to-chat   -i cleaned_场次.json              → 转化聊天（分离转述/回答、切分压缩）
5. mine-phrases      -i cleaned_场次.json              → 措辞候选（人工审批）
6. generate-statements -i cleaned_场次.json            → statement 候选
7. 人工审批 → 合并进 outputs/transcribe/statement_final.json
8. generate-vectors  -i outputs/transcribe/statement_final.json → data/corpus_vectors.json
9. 重启 bot 生效（corpus 向量缓存模块级加载）
```

**方法论文法（务必遵守）**：
- **先判噪声**：礼物/寒暄/转述粉丝话/看二创 全过滤，只提炼高质量话轮
- **直播 ≠ 聊天**：必须 convert-to-chat 切分压缩（一个独立意思 = 一条 15-50 字短陈述），不能整段保留
- **statement 要具体锚定真实事件**：让模型能"直接引用"，不给它"由头自己编"的空间（"客厅能说话"就是反例）
- **措辞从素材提取，不手工编**
- **每步产物先展示审批**，验证通过再往前
- 改 statement_final.json 后：先 `python -c "import json; json.load(open('outputs/transcribe/statement_final.json',encoding='utf-8'))"` 验证 JSON 合法，再 generate-vectors

---

## 第八部分：终极愿景（路线 B：Agent 化）

> **V5 已落地"感知"部分**：时间/天气（context_probe）、看图（vision）、B站开播/动态联动（bili_bridge）+ 真人节奏（读秒窗口/分批发送/智能归纳）。剩余的是"行动/主动性"。

让"灰泽满"成长为**能行动、能感知、能联动的虚拟主播 Agent**：
- **联网**：search_web 了解事件，动态回应时事（轻量探针已有：天气/时间/B站状态，缺通用搜索）
- **主动性**：主动发消息、set_reminder 提醒、check_bilibili 联动她发动态/直播（B站感知已有，缺"主动发送"与"触发判断"）
- **多模态生成**：send_image 发图（现有识图，缺生成）
- 架构：工具集 + ReAct 循环 + 意图区分 + 主动性 + 节制（防骚扰）
- 当前人格系统（素材驱动 + 融合检索 + 感知增强 + 真人节奏）是 Agent 化的地基——**建议进入此阶段时开新会话**（新会话读本文件接上）

---

## 第九部分：风险与注意

- **DeepSeek V4**：`deepseek-v4-flash`；聊天调用需 `extra_body={"thinking":{"type":"disabled"}}` 保留 temperature
- 改 `persona_behaviors.json` / `voice_samples.json` / `phrases.json` 后需重跑对应 precompute
- 改主循环前先备份（core.py 有 backup_test1）
- `.env.prod` 含 API key，已在 .gitignore
- 素材质量决定样本上限——先快速判定场次价值

---

## 第十部分：未来优化方向（2026-08 调研业界后记档）

> 以下为对照业界（AMADEUS/CharacterRAG、plast-mem/HEMA、InCharacter/CharacterEval 等）调研出的优化项。
> ✅ ①会话级记忆、④人格评测已于 V5.2 落地；以下 ②③ 仍未做，量上来后再考虑。

### ① 会话级 episodic 记忆【✅ V5.2 已实现】
- 落地：`session_memory.py`（当前话题 + 本场关键事件，转话题更替，注入【当前会话】）+ 短 query（≤4字）会话语境扩充后再检索。
- 见版本史 V5.2。

### ② 记忆时间维度（承诺过期 + 印象衰减）【需求不大，规划未来】
- **现状**：印象标签 confidence 只升不降（最高 1.0），promises 只保留最近 5 条但无"应兑现日期/过期"概念。
- **业界参考**：Memoria 的指数衰减（`w=exp(-a·x)` 按分钟数）；Zenodo 论文的"记忆有效性期限"。
- **做法草案**：给承诺加应兑现日期，过期自动标注失效；印象标签长时间不出现则降 confidence。

### ③ 轻量情绪状态（neurostate-lite）【有兴趣但怕动摇回复根基，需详聊后定】
- **设想**：给灰泽满一个轻量的当日心情状态（如"今天被夸多了→心情+1"），注入影响语气，提升连续感。
- **风险**：可能动摇"温度靠语气不靠括号、人格来自素材"的回复根基，需谨慎设计（见会话讨论）。
- **业界参考**：EmiliaLab neurostate-engine（六神经递质 0-100，事件驱动更新）；AiVtuberProject 情绪→TTS 映射。

### ④ 人格一致性评测体系【✅ V5.2 已实现】
- 落地：`scripts/persona_eval.py`（项目外独立脚本）。大五人格 14 道开放题 + `--anonymous` 匿名化防名字作弊。
- 实测：实名 13/14（93%）、匿名 14/14（100%）——匿名不掉分，说明人格来自设定而非名字记忆。
- 运行：`python scripts/persona_eval.py [--anonymous]`

---

*由 Claude Code 生成于 2026-08-10 · 供新窗口续接参考*
