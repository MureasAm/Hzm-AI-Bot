"""离线工具箱公共模块：路径约定 + API key 读取 + 输出工具。

目录分工：
- assets/      用户放原始素材（audio/ 音频、transcripts/ 转写）
- outputs/<tool>/  人读分析产物
- data/        运行时数据 + 向量缓存（路径固定，机器人读，勿搬默认值）
- persona/     人格数据（路径固定，机器人读）
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

# ==================== 输入侧：素材 ====================
ASSETS_DIR = PROJECT_ROOT / "assets"
AUDIO_DIR = ASSETS_DIR / "audio"          # 原始音频（拖入处）
TRANSCRIPTS_DIR = ASSETS_DIR / "transcripts"  # 手动放置的转写 JSON

# ==================== 输出侧：分析产物 ====================
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUT_TRANSCRIBE = OUTPUTS_DIR / "transcribe"
OUT_PACE = OUTPUTS_DIR / "pace"
OUT_PIPELINE = OUTPUTS_DIR / "pipeline"
OUT_REGRESSION = OUTPUTS_DIR / "regression"

# ==================== 运行时固定契约（勿改默认值）====================
DATA_DIR = PROJECT_ROOT / "data"
PERSONA_DIR = PROJECT_ROOT / "persona"
# persona/ 按分层职责分子目录：core 核心人设 / behavior 行为 / speech 说话 / world 世界
CORE_DIR = PERSONA_DIR / "core"
BEHAVIOR_DIR = PERSONA_DIR / "behavior"
SPEECH_DIR = PERSONA_DIR / "speech"
WORLD_DIR = PERSONA_DIR / "world"
ENV_FILE = PROJECT_ROOT / ".env.prod"

# 机器人启动要读的固定文件（与 src/plugins/chatbot/constants.py 对齐）
# 向量缓存跟源文件放一起（persona/*/），只有 corpus 向量留 data/（源在 outputs/）
VECTOR_FILE = DATA_DIR / "corpus_vectors.json"
TRIGGER_VECTOR_FILE = BEHAVIOR_DIR / "trigger_vectors.json"
VOICE_SAMPLE_VECTOR_FILE = SPEECH_DIR / "voice_sample_vectors.json"
PHRASE_VECTOR_FILE = SPEECH_DIR / "phrase_vectors.json"
PREFERENCE_VECTOR_FILE = WORLD_DIR / "preference_vectors.json"
CORE_STORY_VECTOR_FILE = WORLD_DIR / "core_story_vectors.json"
TRAITS_FILE = CORE_DIR / "traits.json"
STYLES_FILE = CORE_DIR / "styles.json"
BEHAVIORS_FILE = BEHAVIOR_DIR / "behaviors.json"
VOICE_SAMPLES_FILE = SPEECH_DIR / "voice_samples.json"
PHRASES_FILE = SPEECH_DIR / "phrases.json"
PREFERENCES_FILE = WORLD_DIR / "preferences.json"
CORE_STORIES_FILE = WORLD_DIR / "core_stories.json"


def get_api_key(name: str) -> str | None:
    """从 .env.prod 读取 API key（OPENAI_API_KEY / ZHIPU_API_KEY）。"""
    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(name):
                    return line.split("=", 1)[1].replace('"', '').strip()
    return None


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
