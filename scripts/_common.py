"""离线工具箱公共模块：路径约定 + API key 读取 + 输出工具。

目录分工：
- assets/      用户放原始素材（audio/ 音频、transcripts/ 转写）
- outputs/<tool>/  人读分析产物
- data/        运行时数据 + 向量缓存（路径固定，机器人读，勿搬默认值）
- persona/     人格数据（路径固定，机器人读）
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

# ==================== 输入侧：素材 ====================
ASSETS_DIR = PROJECT_ROOT / "assets"
AUDIO_DIR = ASSETS_DIR / "audio"          # 原始音频（拖入处）
TRANSCRIPTS_DIR = ASSETS_DIR / "transcripts"  # 手动放置的转写 JSON

# ==================== 输出侧：分析产物（按流水线阶段单开文件夹） ====================
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUT_TRANSCRIBE = OUTPUTS_DIR / "transcribe"    # 转写产物
OUT_CLEAN = OUTPUTS_DIR / "clean"              # 清洗产物
OUT_PACE = OUTPUTS_DIR / "pace"                # 节奏地图
OUT_CONVERT = OUTPUTS_DIR / "convert"          # 直播→聊天转化
OUT_MINE = OUTPUTS_DIR / "mine"                # 挖措辞/挖主题
OUT_STATEMENTS = OUTPUTS_DIR / "statements"    # 场景化陈述
OUT_EVAL = OUTPUTS_DIR / "eval"                    # 评测（persona/retrieval/regression 子目录）
OUT_REGRESSION = OUT_EVAL / "regression"

# ==================== 运行时固定契约 ====================
DATA_DIR = PROJECT_ROOT / "data"
ENV_FILE = PROJECT_ROOT / ".env.prod"

# ⚠️ persona 的路径**唯一真值在 `src/plugins/chatbot/constants.py`**（2026-10-16 收口）。
#    这里以前抄了一份，两份各自漂移——抄的那份还留着 `speech/` 这个 2026-10-04 就删掉的目录。
#    constants.py 只 import pathlib，脚本引用它不会拉起 nonebot（别的脚本早就这么用了）。
# ⚠️ persona 的路径**唯一真值在 `src/plugins/chatbot/constants.py`**（2026-10-16 收口）。
#    这里以前抄了一份，两份各自漂移——抄的那份还留着 `speech/` 这个 2026-10-04 就删掉的目录。
#
# ⚠️ 但**不能写 `from src.plugins.chatbot.constants import ...`**：那会先执行包的
#    `__init__.py`，而它第 52 行就是 `@get_driver().on_bot_connect` → 脚本里直接
#    `ValueError: NoneBot has not been initialized`。所以**按文件路径单独加载这一个模块**
#    （constants.py 只 import pathlib，没有别的依赖）。
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "_hzm_constants", PROJECT_ROOT / "src" / "plugins" / "chatbot" / "constants.py")
_c = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_c)

PERSONA_DIR = _c.PERSONA_DIR
INJECTION_DIR = _c.INJECTION_DIR
SYSTEM_PROMPT_FILE = _c.SYSTEM_PROMPT_FILE
SCHEDULE_FILE = _c.SCHEDULE_FILE
PREFERENCES_FILE = _c.PREFERENCES_FILE
CORE_STORIES_FILE = _c.CORE_STORIES_FILE
TERMS_FILE = _c.TERMS_FILE
BEHAVIORS_FILE = _c.BEHAVIORS_FILE
BEHAVIOR_KEYWORDS_FILE = _c.BEHAVIOR_KEYWORDS_FILE
STATEMENT_FILE = _c.STATEMENT_FILE
VECTOR_FILE = _c.VECTOR_FILE
CORPUS_KEYWORDS_FILE = _c.CORPUS_KEYWORDS_FILE
CORPUS_ASKS_FILE = _c.CORPUS_ASKS_FILE
CORE_STORY_VECTOR_FILE = _c.CORE_STORY_VECTOR_FILE
DECISION_DIR = _c.DECISION_DIR
LEGENDARY_FILE = _c.LEGENDARY_FILE
STICKER_FILE = _c.STICKER_FILE


def get_api_key(name: str) -> str | None:
    """读取配置值：优先进程环境，其次 .env.prod（OPENAI_MODEL 也用这个）。"""
    if os.environ.get(name):
        return os.environ[name]
    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(name):
                    return line.split("=", 1)[1].replace('"', '').strip()
    return None


def get_model_name(default: str = "deepseek-flash") -> str:
    """读取对话模型名；优先环境变量，其次 .env.prod。"""
    return get_api_key("OPENAI_MODEL") or default


def get_openai_base_url(default: str = "https://api.deepseek.com/v1") -> str:
    """读取 OpenAI 兼容接口地址。"""
    return get_api_key("OPENAI_API_BASE") or default


def ensure_utf8_stdout():
    """Windows 控制台默认 GBK，强制 UTF-8 避免 emoji 打印报错。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def report_saved(*paths):
    """统一打印『输出已保存到 X』。"""
    print("\n✅ 输出已保存：")
    for p in paths:
        print(f"   - {p}")


def warn_fixed_path(path) -> None:
    """提示某个路径是机器人启动要读的固定位置。"""
    print(f"   ⚠️ {path} 是机器人启动要读的固定位置，如需改输出请用 --out 并手动迁移")


def isolate_user_stores(root=None, *, load_plugins: bool = False):
    """把**用户数据**（短期记忆 / 会话记忆 / 长期记忆 v2）全部指到临时目录。

    ## 为什么要有这个函数（2026-10-16）

    离线实验会直接跑 `core.handle_chat`，它一路上会往 `user_memory/` 里写东西。
    之前**短期记忆和 v2 长期记忆各自隔离了、会话记忆漏了**，实测后果：

        user_memory/session/ 共 585 个文件，其中 337 个是实验造的假会话
        （regr_* 78 ／ abl_* 36 ／ sect_* 120 ／ skel_* 40 ／ sit_* 32 …）

    占 58%。**隔离这件事以前散在 8 个脚本里各写各的，所以漏一层没人发现** ——
    现在收口到这一处，以后再加存储只改这个函数。

    用法（`nonebot.load_plugins` 之后、跑第一条消息之前）：

        import _common
        tmp = _common.isolate_user_stores()          # 自己建临时目录
        tmp = _common.isolate_user_stores("out/x")   # 或者指定目录

    ⚠️ 会替换 `core._MEMORY_V2_STORE`（它是模块级实例，改不了路径）。
    """
    import tempfile
    from pathlib import Path

    import src.plugins.chatbot.core as core
    import src.plugins.chatbot.memory as mem
    import src.plugins.chatbot.memory_v2 as mv2
    import src.plugins.chatbot.session_memory as sm

    root = Path(root) if root else Path(tempfile.mkdtemp(prefix="hzm_offline_"))
    root.mkdir(parents=True, exist_ok=True)
    mem.use_storage(root)                    # 短期记忆（按会话分文件）
    sm.use_storage(root)                     # 会话记忆（话题/事件/情绪）← 之前一直漏的就是它
    core._MEMORY_V2_STORE = mv2.ShadowMemoryStore(   # 长期记忆 v2（影子库 + 审计）
        state_file=root / "memory_v2_shadow.json",
        audit_file=root / "memory_v2_audit.jsonl",
    )
    return root


def chatlog_lines():
    """跨所有会话产出聊天记录行（JSON 文本）—— 分析脚本**统一走这个**。

    聊天落盘 2026-10-08 起**按会话分文件**（`data/chat_log/<QQ号>.jsonl`），
    原来那个 `chat_log/chat.jsonl` 单文件已经拆开并改名为 `.migrated`。
    所以脚本里**不要再写死 `chat_log/chat.jsonl`**（那个路径已经不存在了，
    写死就再也读不到东西，而且不报错——静默空结果）。

    放这里而不是复用 `chatlog.iter_lines`：插件包一导入就会拉起 NoneBot
    （这些分析脚本刻意不启动它）。
    """
    log_dir = DATA_DIR / "chat_log"
    if not log_dir.exists():
        return
    for f in sorted(log_dir.glob("*.jsonl")):
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in content.splitlines():
            if line.strip():
                yield line
