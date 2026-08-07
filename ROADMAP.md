# 灰泽满 AI 机器人 · 开发路线图 & 会话交接

> 本文件由 Claude Code 在 2026-08 协作中生成，供新窗口续接。**新开会话时先读本文件**即可无缝接上。
> 关键：本文件记录了完整的重构历程、方法论与工具链，比代码本身更能帮你快速进入状态。

---

## 第一部分：会话交接摘要（本次会话做了什么）

### 这个项目是什么

基于 **NoneBot2 + OneBot V11 + NapCatQQ** 的虚拟主播"灰泽满"QQ 聊天机器人。核心目标不是"通用聊天"，而是**高度还原主播的语言风格、行为模式与情感表达**。用户希望做成"最好的展示"。

### 本次会话的核心：人格系统三层重构（V1 → V2 → V3）

这一轮从"工程化"升级到"人格系统重构"，核心是用**真实素材驱动人格**，而非靠规则堆砌。演进脉络：

#### V1：样本 few-shot + 停用记忆污染
- 新建 `persona/voice_samples.json`（精选对话样本库），core.py 注入 few-shot，让模型"看样本学说话"而非"读规则学说话"
- 停用长期记忆 `new_self_fact` 提取——防止 AI 自嗨内容污染真实人格
- 建 `scripts/regression_test.py` 虚构弹幕 A/B 测试

#### V2：砍提示词（318行 → 3500字）
- `persona/system_prompt.txt` 从 318 行/3.2万字 砍到 40 行/3500 字，**原版备份在 `persona/system_prompt_v9_backup.txt`**
- 删掉全部数字配额（揉眼睛次数/括号频率/心虚频率）与几十个语气词详解
- 保留：核心人格 / 说话节奏 / 自我称呼 / 行为反应 / 脆弱时刻
- **实测灵性明显提升**（V1 vs V2 对比在 `outputs/v1_full.txt`、`outputs/v2_full.txt`）
- 顺带删除了"按对话次数决定关系等级"的机制（`RELATION_LEVELS` + 自动升级），默认一律当熟人

#### V3：五路融合检索（按需注入）
- **背景**：V2 暴露句式重复（"你是不是又在蹲我唱歌"出现 3 次）——因为样本全量注入
- **解法**：新增 `src/plugins/chatbot/retrieval.py`，三路向量检索（直播记忆 corpus / 风格样本 voice_sample / 行为触发 trigger）+ 加权 RRF 融合 + 字符预算截断
- 两路确定性记忆（用户长期记忆卡 / 短期对话）不检索永远给
- query 全程只调 1 次 embedding，三路共用
- 新增 `scripts/precompute_voice_sample_vectors.py`、`scripts/precompute_phrase_vectors.py` 预计算向量

#### V3.1：措辞指纹库 + 样本分层 + 长度控制
- 新建 `persona/phrases.json`（措辞指纹库）：同一意思 → 她的真实原话（被夸→"也没有啦"、被戳穿→"啊？没有吧"等 8 组）
- voice_samples 分层 `short/long`，短句样本优先注入控制长度
- 新增 `scripts/analyze_pace.py` 节奏地图工具

### 本轮最核心的方法论（反复验证后确立）

1. **样本 > 规则**：用真实对话示范"怎么说话"，比用规则规定怎么说话有效得多
2. **直播 ≠ 聊天**：直播是叙述体，QQ 聊天是对话体，必须经"转聊天"转化
3. **先筛选后分析**：不能全量分析转写，要先过滤噪声（礼物名单/寒暄/转述），只提炼高质量对话
4. **灰泽满永远自称"灰泽满/hzm"**，绝对不用"我"（这是她的人格标志，主语宾语都如此）
5. **括号是"心里话标注"**，只在情感顶点用（小声/心虚/警觉），不是默认配置

### 关键踩坑记录（新会话务必避免）

- ❌ **全量分析转写**：把 493 个话轮全喂模型标注场景 → 55% 是噪声（礼物名单/寒暄/转述），结论不可信。正确：先判定噪声再分析
- ❌ **"被催播"场景命名错误**：直播中不会催播，应叫"**失约被抓包**"（鸽了/迟到时）
- ❌ **转化时把"我"改成"灰泽满"当错误**：恰恰相反，这是她的正确风格（"请灰泽满吃饭"没错）
- ❌ **prompt 里点名口癖**：说"必须保留好吧" → 模型每条都加"好吧……"。应该"原文有什么就保留什么，不刻意添加"
- ❌ **把"灰泽满住到了"等梗串味**：批量转化时模型互相传染，应逐条独立转化

---

## 第二部分：当前项目架构

```
bot.py                           # NoneBot2 入口
src/plugins/chatbot/             # 核心对话逻辑
├── __init__.py                  # 插件入口
├── constants.py                 # 路径/API/阈值常量（含 V3 检索/融合/预算参数）
├── persona.py                   # 人格规则加载 + 行为匹配（trigger 向量缓存）
├── memory.py                    # 短期记忆（带锁）+ 长期记忆封装
├── rag.py                       # 直播记忆 RAG（embed_query / cosine）
├── retrieval.py                 # V3 三路检索 + RRF 融合 + 预算截断
└── core.py                      # 主循环（handle_chat / build_message_list）
persona/                         # 人格数据
├── system_prompt.txt            # 线上人设提示词（V2，3500字）
├── system_prompt_v9_backup.txt  # 原 318 行完整版备份
├── persona_traits/styles/behaviors.json  # 人格三件套
├── voice_samples.json           # 声音样本库（当前 14 条）
├── phrases.json                 # 措辞指纹库（8 组）
data/                            # 运行数据 + 向量缓存（机器人启动要读）
├── corpus_vectors.json / trigger_vectors.json / voice_sample_vectors.json / phrase_vectors.json
├── memory.json / long_term_memory.json
assets/                          # 用户放原始素材（audio/ 音频、transcripts/ 转写）
outputs/                         # 分析产物（pace/ 节奏地图、transcribe/、regression/ 等）
scripts/                         # 离线工具箱（统一入口 run_tool.py）
prompts/                         # 提示词模板（蒸馏/清洗/描述）
materials/                       # 素材备份
memory_manager.py                # 长期记忆（带文件锁，已移除关系等级机制）
tests/                           # 46 条 pytest 单测
bin/                             # ffmpeg 等
```

---

## 第三部分：离线工具箱（run_tool.py）

**统一入口**：`python scripts/run_tool.py <工具> [参数]`（旧脚本仍可直接运行）

| 子命令 | 干什么 | 常用示例 |
|---|---|---|
| transcribe | 音频 → 原始转写 | `run_tool transcribe assets/audio/live1.m4a` |
| clean-transcript | 转写 → 清洗（合并碎片/修错字/加标点） | `run_tool clean-transcript -i 转写.json` |
| convert-to-chat | 原文 → QQ 聊天回复（灰泽满自称+保留括号） | `run_tool convert-to-chat -i 原文.json` |
| analyze-pace | 清洗后 → 节奏地图（可多场合并） | `run_tool analyze-pace -i cleaned.json --session 第一场 --merge` |
| generate-vectors | 场景化陈述 → 语料向量库 | `run_tool generate-vectors` |
| generate-persona | 场景化陈述 → 人格三件套 | `run_tool generate-persona` |
| precompute | 预计算 trigger/声音样本/措辞向量 | `run_tool precompute --all` |
| pipeline | 四步人格蒸馏流水线 | `run_tool pipeline -i 转写.json` |
| regression | 回复灵性回归测试（A/B 对比） | `run_tool regression --ab` |

**完整素材→样本流程**：
```
转写 → 清洗 → 分析(节奏地图) → 筛选高质量原文 → convert-to-chat 转聊天 → 进 voice_samples.json
```

**目录分工**：`assets/` 放素材 → `outputs/<工具>/` 放分析产物 → `data/`/`persona/` 是机器人读取的固定缓存。

---

## 第四部分：样本库规划（当前 14 条 → 目标 35-40 条）

### 场景规划

| 场景 | 现状 | 目标 | 优先级 |
|---|---|---|---|
| 日常闲聊（游戏/饮食/健康/倒霉） | 9 | 8-10 | ✅ 已够 |
| **失约被抓包**（鸽了/迟到/临时不播） | 1 | 4-5 | 🔴 最高 |
| **被越界/被调戏** | 0 | 3-4 | 🔴 最高 |
| **被夸时**（嘴硬否认） | 0 | 3-4 | 🔴 高 |
| **被戳穿/被质疑** | 1 | 3-4 | 🔴 高 |
| 摆烂/拖延 | 3 | 4-5 | 🟡 |
| 感性流露/孤独 | 2 | 3-4 | 🟡 |
| 分享倒霉事 | 2 | 3-4 | 🟡 |
| 立Flag/承诺 | 0 | 2-3 | 🟢 |
| 害羞/回忆 | 0 | 2-3 | 🟢 |

### 补样本原则（本轮确立）

1. **"被催播"改名"失约被抓包"**——直播中不会催播，只在"该播没播"时发生
2. 样本形态必须是**聊天体**（user=弹幕问，reply=她的回答，两者严格分离，别混）
3. **弹幕问的话和她的回答必须分开**——user 放触发，reply 只放她的回答
4. 质量 > 数量——删掉辨识度低的，只留真有灰泽满味道的
5. **补样本按"缺什么找什么"**：找对应场次

### 找素材指引表（缺什么场景 → 找什么场次的直播）

| 缺失/不足场景 | 该找什么类型的直播场次 | 判断信号 |
|---|---|---|
| **被越界/被调戏** | 弹幕整活多、调戏她的场次 | 她说"你不对劲""请把这份感情留给更值得的人""？？？" |
| **被夸时** | 唱歌/可爱被夸多的场次 | 她说"也没有啦""一般般吧""滤镜太重了" |
| **失约被抓包** | 迟到开播、鸽了、临时有事不播的场次 | 开场在解释"为什么迟到""定闹钟又睡过头" |
| **被戳穿/被质疑** | 她状态被弹幕看穿、被追问的场次 | "啊？没有吧""好吧，可能有一点点" |
| **立Flag/承诺** | 她立目标（早起/更新/直播计划）的场次 | "这周一定""第0天打卡""如果...就绝对不可能" |
| 感性流露/孤独 | 深夜场、聊孤独/独居/家人的场次 | 语气变缓、说"一个人""其实有点" |
| 分享倒霉事 | 聊日常糗事的场次 | "又没带伞""被...了"、有画面感的倒霉故事 |
| 害羞/回忆 | 被提往事、被叫宝宝、被说可爱 | "谢谢宝宝（小声）""别说了" |

### 用户素材情况

- 用户已提供 2 场直播：`assets/audio/` 下（8月5日电话场 + 7月26日突击场）
- 均已转写+清洗+分析，节奏地图在 `outputs/pace/pace_map.md`
- 这两场里挖出的样本：失约被抓包（miss_1）、被戳穿（caught_1）、审丑（daily_short_8）
- **缺被越界/被夸/立Flag**——这两场没料，需找专门场次

---

## 第五部分：愿景与下一步

### 当前主线（进行中）：补样本到目标规模

1. 用户按"缺什么找什么"找素材（优先：被调戏/被夸/失约被抓包场次）
2. 拿到新素材 → 跑工具链（转写→清洗→分析→筛选→转聊天）→ 补进 voice_samples.json
3. 每补一批重新 `precompute voice-samples`

### 后续优化方向（本轮遗留）

- **措辞指纹库扩展**：phrases.json 目前 8 组，可从更多素材里挖她的真实措辞补充
- **清理残留**：`long_term_memory.json` 里已有用户的 `relationship_level` 字段是死数据（机制已删），可清理
- **长度控制验证**：V3.1 加的长度控制平均 74→46 字，但"日常闲聊本来就长"是她的真实节奏，勿过度压短

### 终极愿景（未动）

让"灰泽满"成长为**能行动、能感知、能联动的虚拟主播 Agent**：联网了解事件、主动发消息、多模态看图、B 站联动她发动态/直播。即路线 B（Agent 化）：工具集（send_message/search_web/send_image/set_reminder/check_bilibili）+ ReAct 循环 + 意图区分 + 主动性。

---

## 第六部分：风险与注意

- **DeepSeek V4**：模型名 `deepseek-v4-flash`；思考模式默认开启，聊天调用需 `extra_body={"thinking":{"type":"disabled"}}` 保留 temperature
- **改 `persona_behaviors.json` / `voice_samples.json` / `phrases.json` 后需重跑对应 precompute**
- **素材质量决定样本上限**——不是所有直播都值得提炼，先快速判定场次价值
- **改动主循环前先备份**（V2 的 318 行原版备份在 `persona/system_prompt_v9_backup.txt`）

---

*由 Claude Code 生成于 2026-08 · 供新窗口续接参考*
