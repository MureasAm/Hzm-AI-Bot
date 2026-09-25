# 文件地图（谁是什么 · 一看就懂）

> 给使用者/接手的开发者看。按"入口 → 运行时 → 人格数据 → 工具 → 测试 → 产物"组织。
> 版本史见 `ROADMAP.md`；**还没做的问题**见 `待办清单.md`（从 2026-09 全项目通读蒸馏）。

## 一、入口（怎么跑起来）

| 文件 | 作用 |
|---|---|
| `bot.py` | 启动入口：初始化 NoneBot、注册 OneBotV11、加载 `src/plugins` 插件 |
| `memory_manager.py` | **长期记忆**实现（user_memory/long_term.json）：用户画像/承诺/印象卡读写与 LLM 提取 |
| `启动.bat` | 一键开两个窗口：GPT-SoVITS api + bot（NapCat 自己开） |
| `更新周表.bat` | **把周表图拖上去**即识图并写入 `persona/world/schedule.json`（只覆盖 weekly）。双击=处理 `data/schedule_inbox/` 里最新一张。改完要重启 bot |
| `记录.txt` | 用户手放的**真实翻车对话**（不入库），供分析用 |

## 二、运行时核心（`src/plugins/chatbot/`）

| 文件 | 作用 | 关键点 |
|---|---|---|
| `__init__.py` | 事件入口 | 收消息→进读秒窗口；心跳/在线信号(data/heartbeat\|qq_alive\|qq_offline)；自动通过好友 |
| `core.py` | 主循环 | 十层提示注入、六路检索融合、生成、记忆提取、防复读、梗/行为路由落地 |
| `chat_window.py` | 读秒攒批窗口 | 群=整群一窗；回复前读图+归纳；**群聊先过接话门**（私聊不过）；语音优先；`split_reply` 分批发（打字感） |
| `group_gate.py` | **群聊接话门** | 攒批安静下来时判"这批要不要开口"（判据=用户标定：情绪/经历/点名才接，纯事务附和收尾不接）。失败按"接"放行。`GROUP_GATE=0` 可关 |
| `proactive.py` | **主动发言** | 抓到动态/微博 → 转成「她会主动说的那句话」（走人设 + 按内容检索的风格样本）；转不了/失败返回 None，桥上退回原通知模板。`PROACTIVE=0` 可关 |
| `stickers.py` | **表情包** | 判「她这句话配哪张表情」，十分对应才发（失败**不发**，与接话门相反）。标注在 `persona/media/stickers.json`，图在 `assets/stickers/`。`STICKER=0` 可关 |
| `reply_style.py` | 纯函数后处理 | 拆句/分批延迟/`clean_reply`（换行归一、去括号、省略号纪律、自指兜底）/防复读检测 |
| `retrieval.py` | 检索融合 | corpus/voice/phrase 向量 + behavior(L3) 走 RRF；preference/core_story 命中才带；关键词门/预算 |
| `routing.py` | 硬路由 | legendary 梗库（含 LLM 语境确认）+ 行为意图分类 L3 |
| `persona.py` | 人格加载 | traits/styles/behaviors、terms、schedule 的读取与拼装 |
| `rag.py` | 向量工具 | embedding 客户端封装（`embed_query` 等），检索层的底座 |
| `qq_faces.py` | QQ 表情表 | face id → 中文名（248 条）。NapCat 拿不到 `faceText` 时兜底，免得消息变成"QQ表情6"。**别手改**，来源与提取规则见文件头 |
| `memory.py` | 短期记忆 | short_term.json 带锁读写；并加载根 memory_manager 供长期 |
| `session_memory.py` | 会话记忆 | session.json（话题/事件/指代补全） |
| `group_memory.py` | 群记忆 | groups.json：成员 id→昵称 + 群近况 events |
| `context_probe.py` | 感知 | 时间/农历/天气/直播状态 →【当前时间】（天气异步预热，不卡主循环） |
| `vision.py` | 看图 | glm-4.6v 把图片描述成文字注入 |
| `bili_bridge.py` | B站联动 | 开播/动态变化 → 私聊广播（含表情本地化/压小） |
| `weibo_bridge.py` | 微博联动 | 新微博 → 私聊广播（cookie jar 会话续期） |
| `voice.py` | 语音 | GPT-SoVITS 合成→QQ 语音（静音裁剪、达标重试、截断兜底） |
| `_bridge_common.py` | 公共小件 | bili/weibo 共用的状态读写 + 图片下载 |
| `config.py` | 配置 | NoneBot 配置读取 + DeepSeek/智谱客户端工厂 |
| `constants.py` | 常量 ★ | 所有路径/API/阈值集中，阈值旁都写了实验理由 |

## 三、人格数据（`persona/` —— 她是谁/懂什么/怎么说）

按职责分层，**源文件（人肉维护）+ 向量缓存**放一起；改了源要重跑对应 `precompute`/`generate-vectors`。

| 目录/文件 | 装什么 | 谁读 |
|---|---|---|
| `core/system_prompt.txt` | 核心人格提示词（骨架：身份框架/自我称呼/说话节奏/括号语义） | 每轮 base system |
| `core/traits.json` · `styles.json` | 性格基底 / 语言风格（name+desc+evidence） | →【性格基底】/【语言风格】 |
| `behavior/behaviors.json` | 情境→反应示范（11 条，含真人 samples） | L3 分类命中才注入 |
| `behavior/behavior_keywords.json` | 行为判别词（确定性兜底，跳 LLM） | retrieval |
| `speech/voice_samples.json(+_vectors)` | 她说话原话 few-shot（86 条，风格来源） | RRF 注入"说话方式参考" |
| `speech/phrases.json(+_vectors)` | 措辞指纹（同意思→她真实原话） | 命中注入"固定说法" |
| `world/terms.json` | 名词库 lorebook（人物/梗/黑话 + meaning/reaction/aliases/pattern） | 命中注入"世界" |
| `world/preferences.json(+_vectors)` | 偏好档案 | 命中注入"偏好" |
| `world/core_stories.json(+_vectors)` | 印象最深的经历 | 低阈值浮现"核心记忆" |
| `world/legendary.json` | 经典梗固定应答（含确认） | 硬路由直回 |
| `world/schedule.json` | 周表（根目录 `更新周表.bat` 拖图 OCR 更新） | 每轮注入"周表" |
| `world/statement_final.json → corpus_vectors.json` | 直播记忆语料源(322) → 向量 | 机器人只读向量 |

## 四、离线工具（`scripts/`）

统一入口 `python scripts/run_tool.py <工具>`。分组：

- **蒸馏**：transcribe → clean-transcript → analyze-pace → convert-to-chat
- **生成**：extract-persona（→behaviors 人工审批）、mine-phrases、mine-theme；`generate-statements` 已冷落
- **向量**：`generate-vectors -i persona/world/statement_final.json`；`precompute voice-samples|phrases|preferences|core-stories`
- **评测**：regression / persona-eval / retrieval-eval
- **工具**：bili-check / bili-login / vision-test
- **独立运行（不在 run_tool）**：`watchdog.py`（假死自愈进程）、`notifier.py`（SMTP，被 watchdog 用）、`update_schedule.py`（周表识图；日常用根目录 `更新周表.bat` 拖图，也可丢 `data/schedule_inbox/` 后无参跑）、`label_stickers.py`（给 `assets/stickers/` 打标，**加了新表情就重跑它**，幂等）
- **判据回归集**：`scripts/regression_cases.json`（真实翻车固化成确定性判据）→ `run_tool.py regression --check`

> ⚠️ `generate-persona` 会覆盖人格三件套，需 `--danger`；`generate-vectors` 缺省指向 statement_final，别靠内置 RAW_CORPUS。

## 五、测试（`tests/`，495 个）
`conftest.py` 初始化 NoneBot 并加载插件。核心逻辑（reply_style/retrieval/session/voice/chat_window/bili/group_memory/weibo/short_memory 纯函数）覆盖较全；weibo 推送、config、`__init__` 心跳、watchdog/notifier 覆盖少。

## 六、运行产物/状态（不入库）
- `data/`：bili_state / weibo_state / weibo_cookies(会话jar) / heartbeat / qq_alive|offline / voice_cache / schedule_inbox
- `user_memory/`：short_term / long_term / session / groups（可清空换新开聊）
- `outputs/`：蒸馏与分析中间产物
