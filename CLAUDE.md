# CLAUDE.md — 灰泽满 QQ bot（常驻速查）

> 给每个新会话打地基的短文件。**只放"说错会踩坑的稳定真理"**；细节别写这，看下面的文档指针。

## 这是什么
NoneBot2 + OneBot v11(NapCat) 的"灰泽满"人格聊天机器人。核心是**角色一致性工程**：
真人素材→人格数据→多路检索按需注入→分层提示→记忆→感知→评测。目标不是"会聊天"，是"像她"。

## 黄金律（违反=返工）
1. **样本 > 规则；素材层解决，别用提示词打补丁**。台词/例句放数据层（behaviors/legendary/terms），不写进 system_prompt。
   - **但"样本"要本土化，不是照搬原话**：素材来自**直播弹幕**语境，这里是**私聊**——语境变了，同义改写是适配不是失真，**别把改写当"素材掺假"回退掉**（详见 `docs/待办清单.md` 的「本土化」条）。
   - 判据：样本携带的情境**越具体越危险**——"某一回发生的事"（明天要搬家/现在感冒）会被模型当**现在**复用；应改写成"她一贯如何"。详见下条。
   - ⚠️ **样本必须挂在"情景"上，不能只按话题索引**（2026-10-04 实测）：按话题检索必然捞到"同话题的完整回答"→ 把事实灌进对话 → 自相矛盾/编事实/逐字搬。**所以 voice_samples 整条通道已删，该留的原句并进了 behaviors。**
2. **行为 > 标签**：写"什么情境怎么反应"，别贴性格标签。
3. **确定性后处理只兜底**（clean_reply/防复读/换行归一），能数据解决就别加代码规则。**别让它伤到人设**——曾有个"防措辞固化"因为 75% 的触发都在拦她自己的自称"灰泽满"而被删掉。
4. **单一真值**：同一件事只在一个文件维护（历史上 terms/behaviors/提示词三处打架）。
5. **改数据要重算向量**：改 preferences/core_stories/statement_final 后跑对应 `precompute`/`generate-vectors`，否则检索用旧文本。（**behaviors 不用**——它不走向量，走判别词 + LLM 分类。）
   - **voice_samples / phrases / traits / styles 四个文件已于 2026-10-04 拆掉**（内容分别并进 behaviors 与 `core/system_prompt.txt`，原因见 `docs/情景条目设计模板.md` 六之二 / 六之三）。

## 怎么跑 / 怎么测
- 启动：NapCat 自己开 → `env\python.exe api_v2.py`(GPT-SoVITS,端口9880) → `python bot.py`（或双击 `启动.bat`）。**改代码/数据后要重启 bot**（有进程内缓存）。
- 测试：`.venv\Scripts\python.exe -m pytest -q`（**800 个**，全绿才算完；数字变了说明你增删了用例，顺手改这里）。
- 改**检索/判据/素材**后必跑这两个（unit test 测不出"检索该不该命中"）：
  `run_tool.py retrieval-eval`（query→各路期望，**改语料会把候选排名洗牌，改完必须重跑**）、
  `run_tool.py regression --check`（真实翻车固化成的确定性判据）。
  看到答得不对 → `run_tool.py problems --query "那句话"` 先看检索实况，再决定改哪个文件。
- 新增直播素材：`run_tool.py persona-pipeline <corpus|behaviors>`
  先产候选，人工看 review，确认后加 `--apply`；corpus 会自动重建向量。
  corpus 默认补 `corpus_keywords` 钩子并审计自指代词；存量语料用
  `--sanitize-existing --from-index N` 定向清理。
  （`voice-samples` / `phrases` 两个子命令**已于 2026-10-04 下线**——见「人格数据」条。）
- 推送：代理常抽风。先试 `git push origin main`（直连有时通），不行再用 `-c http.proxy=http://127.0.0.1:7897/7898`。

## 别碰 / 危险
- `generate-persona` 会覆盖人工人格三件套 → 已加 `--danger`，别乱跑。
- `generate-vectors` 别省略 `-i`（缺省指向 statement_final；内置 RAW_CORPUS 是过期副本会覆盖线上421条）。
- `outputs/`、`data/`、`.env.prod` 不入库；`data/weibo_cookies.json` 是敏感会话。
- 语音参考音频必须"无尾静音"（否则合成退化成"几个字+长空尾"）。

## 文件地图（详见 docs/FILE_MAP.md）
- **一条消息的链路**（改之前先认路，跳过这步最容易改错地方）：
  `__init__._handle_chat` → `chat_window.enqueue` → 读秒窗口 → `_flush`(读图+归纳) → `core.handle_chat` → `build_message_list`(17 段注入) → `generate_reply` → `clean_reply` → `split_reply` → `_send`
  （语音优先：`clean_reply` 后先 `should_voice`→`send_voice`，成功就不走文字分段）
- 运行时核心：`src/plugins/chatbot/`（core 主循环 / chat_window 读秒窗口 / retrieval 两路 RRF+两路直达 / reply_style 纯函数 / routing 梗库+行为分类 / persona 加载 / memory|session_memory|group_memory 记忆 / voice 语音 / bili_bridge|weibo_bridge 联动 / context_probe 感知 / vision 看图 / rag 向量 / config 客户端 / constants 常量★）
- 人格数据：`persona/`（core=骨架 / behavior=16 条情景 / world=事实层 / speech=仅剩孤儿向量缓存）
- 长期记忆：根 `memory_manager.py`
- 离线工具：`scripts/run_tool.py <工具>`；**52 个脚本各干什么见 `docs/脚本清单.md`**

## 文档指针（按需懒读，别开头全灌）
- **`docs/` —— 全部说明文档都在这里面，索引是 `docs/README.md`**（按"你现在要干什么"找）
- `docs/交接文档.md` —— **下个会话先读这份**（当前状态 + 下一步 + 卡住的东西）
- `docs/注入设计·人话版.md` —— **最终 messages 的注入文件清单与顺序**
- `docs/FILE_MAP.md` —— 每个文件干什么 ★ 想知道"该改哪"先看这个
- `docs/注入链路.md` —— 素材层 + 注入层的完整解剖（**现在是什么样**）
- `docs/注入设计原理.md` —— 判据怎么选、素材该长什么形式、每改一次是变好还是变坏（**带实验数据**）
- `docs/脚本清单.md` —— 52 个脚本各干什么；**改链路先看"实验/诊断工具"那节**
- `docs/人格素材流水线.md` —— **新直播素材 → corpus / behaviors 的安全流水线**
- `video/README.md` —— **注入链路说明视频**：Remotion 工程、旁白时间轴、Speak TTS 接入和渲染命令
- `docs/包装语审计.md` —— A/C 类已删、B 类保留；它是历史审计记录，不是新的待办。
- `docs/待办清单.md` —— 还没做的问题 + "已判定不是问题、别再改回去"的登记
- `docs/版本史.md` —— V1.0→V7.9 每个版本改了什么、为什么、踩过哪些坑
- `docs/抽查清单.md` —— 抽查她回复时看什么（**推荐用 `scripts/spot_check.py` 自动挑**）

## 当前状态 / 下一步（更新时改这里）
- **当前状态/下一步不在本文件维护**（会跟仓库脱节）——看 `docs/交接文档.md`（下个会话先读那份）。
- 本文件只放**说错会踩坑的稳定真理**：黄金律、怎么跑测、危险清单、文档在哪。
