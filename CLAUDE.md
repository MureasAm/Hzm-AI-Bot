# CLAUDE.md — 灰泽满 QQ bot（常驻速查）

> 给每个新会话打地基的短文件。**只放"说错会踩坑的稳定真理"**；细节别写这，看下面的文档指针。

## 这是什么
NoneBot2 + OneBot v11(NapCat) 的"灰泽满"人格聊天机器人。核心是**角色一致性工程**：
真人素材→人格数据→多路检索按需注入→分层提示→记忆→感知→评测。目标不是"会聊天"，是"像她"。

## 黄金律（违反=返工）
1. **样本 > 规则；素材层解决，别用提示词打补丁**。台词/例句放数据层（behaviors/phrases/legendary/terms），不写进 system_prompt。
2. **行为 > 标签**：写"什么情境怎么反应"，别贴性格标签。
3. **确定性后处理只兜底**（clean_reply/复读/措辞固化），能数据解决就别加代码规则。
4. **单一真值**：同一件事只在一个文件维护（历史上 terms/behaviors/提示词三处打架）。
5. **改数据要重算向量**：改 voice_samples/phrases/preferences/core_stories/statement_final/behaviors 后跑对应 `precompute`/`generate-vectors`，否则检索用旧文本。

## 怎么跑 / 怎么测
- 启动：NapCat 自己开 → `env\python.exe api_v2.py`(GPT-SoVITS,端口9880) → `python bot.py`（或双击 `启动.bat`）。**改代码/数据后要重启 bot**（有进程内缓存）。
- 测试：`.venv\Scripts\python.exe -m pytest -q`（**280 个**，全绿才算完；数字变了说明你增删了用例，顺手改这里）。
- 推送：代理常抽风。先试 `git push origin main`（直连有时通），不行再用 `-c http.proxy=http://127.0.0.1:7897/7898`。

## 别碰 / 危险
- `generate-persona` 会覆盖人工人格三件套 → 已加 `--danger`，别乱跑。
- `generate-vectors` 别省略 `-i`（缺省指向 statement_final；内置 RAW_CORPUS 是过期副本会覆盖线上322条）。
- `outputs/`、`data/`、`.env.prod` 不入库；`data/weibo_cookies.json` 是敏感会话。
- 语音参考音频必须"无尾静音"（否则合成退化成"几个字+长空尾"）。

## 文件地图（详见 FILE_MAP.md）
- **一条消息的链路**（改之前先认路，跳过这步最容易改错地方）：
  `__init__._handle_chat` → `chat_window.enqueue` → 读秒窗口 → `_flush`(读图+归纳) → `core.handle_chat` → `build_message_list`(十层注入) → `generate_reply` → `clean_reply` → `split_reply` → `_send`
  （语音优先：`clean_reply` 后先 `should_voice`→`send_voice`，成功就不走文字分段）
- 运行时核心：`src/plugins/chatbot/`（core 主循环 / chat_window 读秒窗口 / retrieval 六路RRF / reply_style 纯函数 / routing 梗库+行为分类 / persona 加载 / memory|session_memory|group_memory 记忆 / voice 语音 / bili_bridge|weibo_bridge 联动 / context_probe 感知 / vision 看图 / rag 向量 / config 客户端 / constants 常量★）
- 人格数据：`persona/`（core/behavior/speech/world，源文件+向量）
- 长期记忆：根 `memory_manager.py`
- 离线工具：`scripts/run_tool.py <工具>`（见 FILE_MAP 分组）

## 文档指针（按需懒读，别开头全灌）
- `FILE_MAP.md` —— 每个文件干什么（给使用者）★ 想知道"该改哪"先看这个
- `待办清单.md` —— 从 2026-09 全项目通读蒸馏的**未完成**问题（已做掉的不在里面）
- `ROADMAP.md` —— 版本史/踩坑/链路剖析（部分停在 2026-08）

## 当前状态 / 下一步（更新时改这里）
- 已完成：群聊整场+群记忆、GPT-SoVITS 语音、记忆清理、corpus 清洗、P3 提示词下沉、P4 重构(E5天气异步/E6记忆归位/E7桥公共件)、同期双行为、输出**换行归一**(clean_reply)、记忆**时间戳**(长期注入 last_seen 间隔 + 短期每轮存时间、生成前注入一行"距上一轮多久")。
- 下一步：① **主动发消息**（初步设想：结合 B站动态等主动找话题）；② 往期 cohort 词表（`VR27期/277/同期` 与 `前辈/后辈` 的等价写法，方案见 `待办清单.md`）。
