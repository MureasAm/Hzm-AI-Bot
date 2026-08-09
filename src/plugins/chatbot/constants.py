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
VOICE_SAMPLES_FILE = PROJECT_ROOT / "persona" / "voice_samples.json"  # 声音样本库(few-shot)

MEMORY_FILE = PROJECT_ROOT / "data" / "memory.json"                 # 用户短期记忆
VECTOR_FILE = PROJECT_ROOT / "data" / "corpus_vectors.json"         # 直播记忆向量库
TRIGGER_VECTOR_FILE = PROJECT_ROOT / "data" / "trigger_vectors.json"  # trigger 向量缓存
VOICE_SAMPLE_VECTOR_FILE = PROJECT_ROOT / "data" / "voice_sample_vectors.json"  # 声音样本向量缓存
PHRASE_VECTOR_FILE = PROJECT_ROOT / "data" / "phrase_vectors.json"   # 措辞指纹向量缓存

# ==================== API ====================
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
DEFAULT_MODEL = "deepseek-v4-flash"
EMBEDDING_MODEL = "embedding-3"

# DeepSeek V4 默认开启思考模式；思考模式下 temperature 等参数不被支持，
# 故对聊天 / 记忆提取调用显式禁用思考。
THINKING_DISABLED = {"extra_body": {"thinking": {"type": "disabled"}}}

# ==================== 对话参数 ====================
CHAT_TEMPERATURE = 0.85       # 实验：回高温度（低温度让模型更走 RP 默认括号模板）
CHAT_FREQUENCY_PENALTY = 0.3  # 实验：penalty 降低（0.5 未压括号还引入整句重复）
CHAT_MAX_TOKENS = 150
MEMORY_EXTRACT_TEMPERATURE = 0.1
MEMORY_EXTRACT_MAX_TOKENS = 100

# ==================== 检索阈值 ====================
BEHAVIOR_MATCH_THRESHOLD = 0.65
RAG_THRESHOLD = 0.35
RAG_TOP_K = 2

# ==================== V3 三路检索 ====================
CORPUS_TOP_N = 2              # 直播记忆每路取 top（少取，防止长文本挤占 few-shot 预算）
VOICE_SAMPLE_TOP_N = 3        # 风格样本每路取 top
VOICE_SAMPLE_THRESHOLD = 0.35 # 风格样本阈值（与 RAG 同）
BEHAVIOR_TOP_N = 1            # 行为指令只取 1 条（保持 V2 语义）
VOICE_SAMPLE_KEEPALIVE = True # 样本全低于阈值时保底注入 1 条（保住口癖）
VOICE_SAMPLE_MIN_K = 1        # 保底注入条数
VOICE_SAMPLE_PREFER_SHORT = True  # 注入时优先 short 档样本（控制回复长度）

# ==================== V3 措辞指纹检索 ====================
PHRASE_TOP_N = 2              # 措辞组每路取 top
PHRASE_THRESHOLD = 0.40       # 措辞组阈值（略高于样本，避免误命中）
PHRASE_PHASES_MAX = 3         # 每个措辞组注入的短语条数上限

# ==================== V3 RRF 融合 ====================
RRF_K = 60                    # RRF 平滑常数
SOURCE_WEIGHTS = {"behavior": 1.5, "corpus": 1.0, "voice_sample": 1.0, "phrase": 1.2}
RETRIEVAL_TOPK = 6            # 融合后条数硬上限

# ==================== V3 预算控制 ====================
RETRIEVAL_BUDGET_CHARS = 1200      # 融合检索注入字符预算（corpus+samples）
MAX_RETRIEVAL_ITEM_CHARS = 300     # 单条检索结果字符上限
VOICE_SAMPLE_REPLY_TRIM_CHARS = 60   # 长样本回复裁剪到该字数（引导短句）
MAX_CONTEXT_CHARS = 8000           # 整体安全上限兜底

# ==================== 短期记忆 ====================
SHORT_MEMORY_LINES = 10  # 最近 5 轮，每轮 2 条（用户 + AI）；>3 轮可缓解承诺/借口遗忘
