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
RAG_THRESHOLD = 0.55
RAG_TOP_K = 2

# ==================== V3 三路检索 ====================
CORPUS_TOP_N = 3              # 直播记忆每路取 top（315 条库后 2→3，相关背景更容易命中）
VOICE_SAMPLE_TOP_N = 3        # 风格样本每路取 top
VOICE_SAMPLE_THRESHOLD = 0.60 # 风格样本阈值（0.60：真直播命中≈0.70，日常"聊时间"伪关联≈0.575，取中间值切断伪关联）
BEHAVIOR_TOP_N = 1            # 行为指令只取 1 条（保持 V2 语义）
VOICE_SAMPLE_KEEPALIVE = True # 样本全低于阈值时保底注入 1 条（保住口癖）
VOICE_SAMPLE_KEEPALIVE_MIN_SIM = 0.60  # 保底注入的最低相关度：与主阈值一致，低于则不注入（宁断档不错话题，防"日常聊时间"被塞直播样本）
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

# ==================== 读秒窗口（方案B：消息攒批） ====================
# 静默窗口随机区间（真人打字回复普遍 5-10s；随机避免固定等长=机械感）
READ_WINDOW_MIN_SECONDS = 5.0
READ_WINDOW_MAX_SECONDS = 10.0

# ==================== 分批发送（打字感） ====================
SPLIT_REPLY_ENABLED = True     # 长回复拆成几句分开发送
SPLIT_MIN_LEN = 10             # 回复短于该长度不拆（灰泽满短句多，阈值别太高）
SPLIT_MAX_PARTS = 4            # 最多拆成几条消息，超出并入最后一条（防刷屏）
# 句间延迟按内容长度模拟打字（基础 + 每字，夹 MIN~MAX，±15% 抖动）
# 用户反馈 0.6~2.6s 太快"像涌出"，调大到 ~1.8~5s 还原打字过程
SPLIT_DELAY_BASE_MS = 1500
SPLIT_DELAY_PER_CHAR_MS = 100
SPLIT_DELAY_MIN_MS = 1800
SPLIT_DELAY_MAX_MS = 5000
SPLIT_DELAY_JITTER = 0.15

# ==================== 感知增强（路线B） ====================
# 方向1 感知
WEATHER_BASE_URL = "https://devapi.qweather.com"  # 和风 API Host 根；每项目专属域名，需在控制台查并配 WEATHER_BASE_URL
WEATHER_CACHE_SECONDS = 3600      # 天气进程内缓存 1 小时，省免费额度
WEATHER_GEO_CACHE_SECONDS = 86400  # 城市名→LocationID 解析缓存 1 天
# 方向2 视觉
VISION_MODEL = "glm-4.6v"         # 视觉理解模型（智谱 OpenAI 兼容；用户有免费 token）
VISION_MAX_TOKENS = 512           # 视觉描述输出上限
# 视觉调用关掉思考模式：glm-4.6v 默认会先推理（可拖到 9s+），关掉后秒出，消除"图片迟到"感
VISION_THINKING_DISABLED = {"extra_body": {"thinking": {"type": "disabled"}}}
# 方向3 B站联动
PUSH_INTERVAL = 45                # B站轮询间隔（秒）
BILI_STATE_FILE = PROJECT_ROOT / "data" / "bili_state.json"   # 开播/动态去重状态持久化

# 好友申请自动通过（否则新加的绿冻是单向好友，推送发不到，见 NapCat issue #72）
AUTO_ACCEPT_FRIEND = True

# 偏好档案（第 5 路语义检索）
PREFERENCE_VECTOR_FILE = PROJECT_ROOT / "data" / "preference_vectors.json"
PREFERENCE_THRESHOLD = 0.55       # 偏好命中阈值（实测：真实命中 0.58+，短句偏好向量泛化过头会误命中，取 0.55 压误命中）
PREFERENCE_TOP_N = 2              # 最多注入几条偏好条目

# 核心记忆（印象最深的结晶，独立于 corpus 单独检索，低阈值高浮现）
CORE_STORY_VECTOR_FILE = PROJECT_ROOT / "data" / "core_story_vectors.json"
CORE_STORY_THRESHOLD = 0.42       # 比偏好/行为低，核心故事更容易浮出（但低于此不注入，防乱触发）
CORE_STORY_TOP_N = 2              # 最多注入几条核心记忆
