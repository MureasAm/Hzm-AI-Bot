"""集中管理路径 / API / 阈值的常量。

项目根目录：src/plugins/chatbot/ 往上数四级。
"""
from pathlib import Path

# ==================== 路径 ====================
# constants.py -> chatbot -> plugins -> src -> 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[3]

SYSTEM_PROMPT_FILE = PROJECT_ROOT / "persona" / "system_prompt.txt"
TRAITS_FILE = PROJECT_ROOT / "persona" / "persona_traits.json"
STYLES_FILE = PROJECT_ROOT / "persona" / "persona_styles.json"
BEHAVIORS_FILE = PROJECT_ROOT / "persona" / "persona_behaviors.json"

MEMORY_FILE = PROJECT_ROOT / "data" / "memory.json"                 # 用户短期记忆
VECTOR_FILE = PROJECT_ROOT / "data" / "corpus_vectors.json"         # 直播记忆向量库
TRIGGER_VECTOR_FILE = PROJECT_ROOT / "data" / "trigger_vectors.json"  # trigger 向量缓存

# ==================== API ====================
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
DEFAULT_MODEL = "deepseek-v4-flash"
EMBEDDING_MODEL = "embedding-3"

# DeepSeek V4 默认开启思考模式；思考模式下 temperature 等参数不被支持，
# 故对聊天 / 记忆提取调用显式禁用思考。
THINKING_DISABLED = {"extra_body": {"thinking": {"type": "disabled"}}}

# ==================== 对话参数 ====================
CHAT_TEMPERATURE = 0.85
CHAT_MAX_TOKENS = 150
MEMORY_EXTRACT_TEMPERATURE = 0.1
MEMORY_EXTRACT_MAX_TOKENS = 100

# ==================== 检索阈值 ====================
BEHAVIOR_MATCH_THRESHOLD = 0.65
RAG_THRESHOLD = 0.35
RAG_TOP_K = 2

# ==================== 短期记忆 ====================
SHORT_MEMORY_LINES = 6  # 最近 3 轮，每轮 2 条（用户 + AI）
