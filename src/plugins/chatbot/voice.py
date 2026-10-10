"""语音回复：GPT-SoVITS 合成 → QQ 语音消息（服务地址/参考音频/长度闸门见 constants.py 的
SOVITS_URL / SOVITS_REF_DIR / VOICE_MIN_LEN；总开关是 .env 的 VOICE_ENABLED）。

原则：成句(≥30字)回复朗读成语音条；短敷衍词/含内心戏括号 → 打字文字（互斥，不双发）。
灰泽满实际回复多为 15~45 字一句（样本最长 60），语音覆盖"完整成句"档，朗读模型擅成句。
语音在回复路径里优先尝试，**超预算就放弃**（`_TTS_BUDGET_SEC`），失败自动回退文字分段
——绝不影响收到回复，也绝不让她卡在 TTS 上不说话。

合成结果的质量判据（`_is_acceptable`）：空洞（总长被撑爆）与截断（有效语音不够）都要拦。
不达标就换温度重合成；重试耗尽的取"最不烂"的一次发出去。缓存读回**照样复检**。
"""
import array
import asyncio
import hashlib
import math
import os
import re
import wave
from pathlib import Path

import httpx

from nonebot.adapters.onebot.v11 import MessageSegment

from .constants import (
    SOVITS_URL, SOVITS_REF_DIR, VOICE_CACHE_DIR, VOICE_MIN_LEN, VOICE_MAX_LEN,
)
from .config import get_voice_enabled

# 不可语音化的"内心戏/动作"括号（TTS 表达不了 → 这些回复走文字）
_INNER_VOICE_PARENS = ("小声", "心虚", "捂脸", "揉眼睛", "叹气", "皱眉", "低头", "脸红")

# 短寒暄/高情绪词（晚安/早安/辛苦啦…）：命中可突破 VOICE_MIN_LEN 下限也朗读语音。
# 纯数据可随意增删——想让她"说出口"的短句就加这（如"晚安""好想你""记得想我"）。
# 只放宽长度闸门，内心戏括号/数字/链接这些内容铁则仍然最高优先（如"晚安（心虚）"仍文字）。
_GREETING_PHRASES = (
    # 问候
    "晚安", "早安", "早上好", "中午好", "下午好", "晚上好",
    "你好", "你好呀", "你好啊", "嗨", "hi", "hello", "欢迎",
    # 道别
    "拜拜", "再见", "明天见", "下次见", "先睡啦", "先去忙啦",
    # 高情绪应答/关切
    "辛苦啦", "谢谢你", "谢谢呀", "抱歉", "对不起",
    "在呀", "睡了吗", "还没睡", "到家了吗", "好久不见",
    "好想你", "想你啦", "记得想我", "想我了吗", "我也想你",
)

# 寒暄放宽只救"真·短应酬句"（≤12字）。别让 13~29 字的普通长句因为顺带提到
# "还没睡/你好/想你"这类词就突破下限被语音（那句就不是寒暄，是陈述）。
_GREETING_MAX_LEN = 12

_tts_lock = asyncio.Lock()   # 同时只合成一条（TTS 占 GPU，并发互相拖慢）


def _is_greeting(reply: str) -> bool:
    """是否命中短寒暄/高情绪词表（只判断词，长度由调用方管）。"""
    r = reply.strip(" 　～~。！!？?,，").lower()
    return any(g in r for g in _GREETING_PHRASES)


def should_voice(reply: str, min_len: int = VOICE_MIN_LEN,
                 max_len: int = VOICE_MAX_LEN, force: bool = False) -> bool:
    """判断这条回复是否适合朗读成语音条。

    长度/寒暄判断用**实际会读出来的文本**（_tts_text 剥括号后）——避免"原文带括号尾巴够 20 字、
    读出来只剩短句"的误触发。内心戏括号/数字/链接铁则仍针对原文（括号是文字专属表达）。
    语音 = 剥括号后成句(≥min_len) 或 短寒暄命中；短敷衍词("在呢")/含内心戏括号/链接仍打字。

    `force=True`：用户**点名要了语音**（消息里带"语音"，见 `voice_requested`）→
    放宽**长度**闸门，短句也读。内容铁则（内心戏括号/数字/链接）仍然最高优先，
    超长（>max_len）也仍走文字——这两类是"读出来就没意义/发不出去"，
    不是风格偏好（用户 2026-10-05 要的是"她肯开口"，不是"什么都念"）。
    """
    if not reply:
        return False
    # 内容铁则（看原文）：内心戏括号 TTS 表达不了 → 文字；数字/链接语音没法回看 → 文字
    if any(p in reply for p in _INNER_VOICE_PARENS):
        return False
    if re.search(r"\d{2,}|http|https|网址|邮箱", reply):
        return False
    # 长度看实际朗读文本：剥掉（笑）转换、其余括号剥除、去空白
    spoken = _tts_text(reply)
    if not spoken:
        return False
    n = len(spoken)
    if force and n <= max_len:
        return True
    in_range = min_len <= n <= max_len
    # 短寒暄命中 → 突破长度下限。只救"真·短应酬句"(≤_GREETING_MAX_LEN)，
    # 长句顺带提到词表词不算寒暄 → 按长度规则走文字。超长仍走文字分段
    greeting_ok = (not in_range) and n <= _GREETING_MAX_LEN and _is_greeting(spoken)
    return in_range or greeting_ok


# 用户**直接要语音**的说法（"能发语音吗""来条语音听听"）：命中 → 下一句回复直接朗读。
# 纯数据可随意增删；只放宽长度闸门，内容铁则（链接/数字/内心戏括号）不受它影响。
_VOICE_REQUEST_WORDS = ("语音", "录音", "发个声音")


def voice_requested(user_msg: str) -> bool:
    """用户这条消息是不是在**点名要语音**。

    为什么要它（用户 2026-10-05）：不少人发"能发语音吗"，而她的应答往往是
    "行啊""这就说给你听"这类**短句**——够不着 30 字下限、也不在寒暄表里，
    于是照样回文字，用户觉得"没听劝"。
    """
    return any(w in (user_msg or "") for w in _VOICE_REQUEST_WORDS)


def _tts_text(reply: str) -> str:
    """清洗：笑声括号→笑声文字，其余括号/动作剥掉（括号是文字专属表达）。

    顺带把自称 "hzm" 换成"灰泽满"：TTS 念不了这三个字母（会读成"诶吃贼诶姆"
    或干脆吞掉）。字数不变（都是三个字），所以 `should_voice` 的长度判断不受影响。
    """
    t = reply
    # 换自称要在别处之前做：括号里的 "（hzm心虚）" 反正会被剥掉，不影响
    t = re.sub(r"hzm", "灰泽满", t, flags=re.IGNORECASE)
    for pat, rep in (("（笑）", "哈哈哈"), ("（笑死）", "哈哈哈哈"),
                     ("（偷笑）", "嘿嘿嘿"), ("（尬笑）", "哈"), ("（苦笑）", "呵")):
        t = t.replace(pat, rep)
    t = re.sub(r"[（【\[][^）】\]]*[）】\]]", "", t)
    t = re.sub(r"\s+", "", t)
    return t.strip()


# 参考音频：情绪档（可选，粗粒度；细粒度如"小声/心虚"做不到走文字）。
# 只放 ref_voice.wav（+同名 .txt 转写）就能跑——缺情绪档时自动回落默认/任一 .wav。
def _pick_tier(reply: str) -> str:
    if any(p in reply for p in ("笑", "哈哈", "！", "～")):
        return "ref_happy"
    if any(p in reply for p in ("困", "累", "唉", "……")):
        return "ref_lazy"
    return "ref_serious"


def _resolve_ref(reply: str) -> Path | None:
    """按情绪档选参考音频；档位文件缺失 → 回落 ref_voice.wav → 任一 .wav。一个参考也能跑。"""
    d = Path(SOVITS_REF_DIR)
    if not d.exists():
        return None
    tier = d / f"{_pick_tier(reply)}.wav"
    if tier.exists():
        return tier
    default = d / "ref_voice.wav"
    if default.exists():
        return default
    for wav in sorted(d.glob("*.wav")):
        return wav
    return None


def _read_prompt_text(ref_path: Path) -> str:
    """读参考音频同名 .txt 的转写（灰泽满在这段里说的话），GPT-SoVITS 靠它对齐。
    缺失则空串（仍能合成但对齐/音质弱）——放音频时建议带一份同名 .txt。"""
    txt = ref_path.with_suffix(".txt")
    if txt.exists():
        return txt.read_text(encoding="utf-8").strip()
    return ""


# ── 静音 / 有效语音判定 ────────────────────────────────────────────────
# 判决线**必须相对本文件峰值取**，不能写死（2026-10-11 拿 3438 个缓存文件实测）：
# 线上 wav 是 32kHz，底噪常在 RMS 300~500 —— 旧代码写死 _SILENCE_RMS = 300，
# 恰好骑在底噪上，于是"整条只剩底噪"的坏条子被判成"满是人声"：60 秒的条子里
# 只有 2 秒人声，裁静音**一个字都没裁掉**（空操作）。正常语音窗 RMS 在数千，
# 取峰值的 5%（下限 800）能把语音和底噪干净分开。
_ACTIVE_RMS_FLOOR = 800.0    # 绝对下限（整体音量都很小时，别把判决线压进底噪里）
_ACTIVE_RMS_REL = 0.05       # 相对本文件峰值
_SILENCE_WIN_SEC = 0.05      # 能量统计窗秒数
_SILENCE_MARGIN_SEC = 0.05   # 裁完留的边距，防咬掉字头字尾
# 句中"空"超过 _GAP_SEC 就是合成故障（不是她停顿），压到 _GAP_KEEP_SEC。
# 正常换气不到 1 秒、省略号停顿也就 1 秒出头，3 秒以上只有坏合成才会出现。
_GAP_SEC = 3.0
_GAP_KEEP_SEC = 0.6


def _read_mono_wav(wav_path: Path) -> tuple[array.array, int] | tuple[None, int]:
    """读 16-bit 单声道 wav → (样本数组, 采样率)。格式不符 / 读不动 → (None, 0)。"""
    try:
        with wave.open(str(wav_path), "rb") as w:
            if w.getsampwidth() != 2 or w.getnchannels() != 1:
                return None, 0
            rate = w.getframerate()
            a = array.array("h")
            a.frombytes(w.readframes(w.getnframes()))
        return a, rate
    except Exception:
        return None, 0


def _rms(seg: array.array) -> float:
    return math.sqrt(sum(x * x for x in seg) / len(seg)) if len(seg) else 0.0


def _windows(a: array.array, rate: int) -> list[tuple[int, int]]:
    """样本区间列表，每窗 _SILENCE_WIN_SEC（最后一窗可能不足）。"""
    hop = max(1, int(rate * _SILENCE_WIN_SEC))
    return [(s, min(s + hop, len(a))) for s in range(0, len(a), hop)]


def _active_threshold(vals: list[float]) -> float:
    """判决线：本文件峰值的 _ACTIVE_RMS_REL，但不低于 _ACTIVE_RMS_FLOOR。"""
    peak = max(vals) if vals else 0.0
    return max(_ACTIVE_RMS_FLOOR, peak * _ACTIVE_RMS_REL)


def _speech_seconds(wav_path: Path) -> float:
    """wav 里有效语音(能量超判决线)的秒数——量的是她**真的说了多久**。读不动 → 0.0。"""
    a, rate = _read_mono_wav(wav_path)
    if a is None or not len(a):
        return 0.0
    vals = [_rms(a[lo:hi]) for lo, hi in _windows(a, rate)]
    thr = _active_threshold(vals)
    return sum(1 for v in vals if v > thr) * _SILENCE_WIN_SEC


def _wav_duration(wav_path: Path) -> float:
    """wav 总时长（秒）。"""
    try:
        with wave.open(str(wav_path), "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


def _clean_wav(wav_path: Path) -> tuple[float, float]:
    """就地清理刚合成的 wav，并回报质量：**裁掉首尾静音 + 压掉句中超长空洞**。

    返回清理后 (有效语音秒数, 总秒数)。格式不符 / 整段无声 → 原样不动（宁可发原件，
    也不能把语音裁没）；解析失败返回 (0.0, 原总长)。

    坏合成有两种形态，这里一起治：
    - "说两句 + 十几秒空尾"（首尾裁剪治它）；
    - "说两句 + 中间几十秒空"（首尾裁剪完全治不了，要切句中）。
    """
    try:
        a, rate = _read_mono_wav(wav_path)
        if a is None or not len(a):
            return 0.0, 0.0
        wins = _windows(a, rate)
        vals = [_rms(a[lo:hi]) for lo, hi in wins]
        loud = [v > _active_threshold(vals) for v in vals]
        if not any(loud):
            return 0.0, len(a) / rate      # 整段无声：不裁
        first = loud.index(True)
        last = len(loud) - 1 - loud[::-1].index(True)
        margin = int(rate * _SILENCE_MARGIN_SEC)
        # 保留区间：先取 [首个有声窗-边距, 末个有声窗+边距]，再把句中超长空洞切成两半
        cur = max(0, wins[first][0] - margin)
        keeps: list[tuple[int, int]] = []
        i = first
        while i <= last:
            if loud[i]:
                i += 1
                continue
            j = i
            while j <= last and not loud[j]:
                j += 1
            gap_lo, gap_hi = wins[i][0], wins[j - 1][1]
            half = int(rate * _GAP_KEEP_SEC / 2)
            if gap_hi - gap_lo >= rate * _GAP_SEC and gap_lo + half < gap_hi - half:
                keeps.append((cur, gap_lo + half))
                cur = gap_hi - half
            i = j
        keeps.append((cur, min(len(a), wins[last][1] + margin)))
        cleaned = array.array("h")
        for lo, hi in keeps:
            cleaned += a[lo:hi]
        # 临时文件 + 原子替换：写的正是缓存文件本身，写坏一半会让缓存里躺着个坏 wav
        tmp = Path(f"{wav_path}.tmp")
        with wave.open(str(tmp), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(cleaned.tobytes())
        os.replace(tmp, wav_path)
        return sum(1 for v in loud if v) * _SILENCE_WIN_SEC, len(cleaned) / rate
    except Exception:
        return 0.0, _wav_duration(wav_path)   # 裁不动就发原件，别把语音裁没了


# 期望时长口径（2026-10-11 标定：2340 条真实缓存回归 + 4 条纯英文实跑）。
# 旧的单一 4.2 字/秒把中文估长了 40%（真实语速中位 5.75 字/秒），于是判定线恰好压在
# 正常分布的 p25 上——"说得稍快"的正常条子会被误判成截断。
#
# 拉丁串要分两种，别用一个数：
# - 夹在中文里的（"P5R""VirtuaReal""QQ"）多是被**逐字母念**出来的，慢 → 回归得 10.5 字符/秒；
# - 整句英文走英语前端，连读，实测 16~19 字符/秒（旧口径按中文字速估，能把期望撑到 3 倍，
#   于是每条英文回复都被判"截断"、白白重合成三次）。
_CJK_PER_SEC = 5.9
_LATIN_PER_SEC = 10.5
_ENGLISH_PER_SEC = 16.0
_MIN_SPEECH_RATIO = 0.6   # 有效语音 ≥ 期望的 60% 才算"读完整"
_MAX_TOTAL_RATIO = 1.6    # 总长 ≤ 期望的 160%（再长就是被空洞撑出来的）
_MIN_TOTAL_SEC = 5.0      # 短句只查"总长不超过这个绝对上限"，不按比例（期望太小没意义）
_CHECK_MIN_CHARS = 8      # 短于这么多字不查"读完整"（"晚安"本来就短）

_CJK_RE = re.compile(r"[一-鿿]")
_LATIN_RE = re.compile(r"[A-Za-z]")

# 重试：同一句话换温度重合成。GPT-SoVITS 偶发截断/空洞，换温度能救回一部分。
_TTS_TEMPERATURES = (0.8, 0.95, 0.85)
_TTS_ATTEMPT_TIMEOUT_SEC = 30.0   # 单次 HTTP 上限（旧值 120s，卡住时把"她一个字都不发"拖到 6 分钟）
_TTS_MIN_ATTEMPT_SEC = 6.0        # 剩余预算不够再来一次就别试了
_TTS_BUDGET_SEC = 45.0            # 一次语音从开始到必须出结果的总预算（含等锁），超了走文字


def _expect_seconds(text: str) -> float:
    """估这条文本读出来该有多长（秒）。标点不计（标定里被常数项吸收了）。"""
    cjk = len(_CJK_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    if not cjk:
        return latin / _ENGLISH_PER_SEC
    return cjk / _CJK_PER_SEC + latin / _LATIN_PER_SEC


def _bloat_limit(expect: float) -> float:
    """总长的上限：超过它 = 被空洞撑爆了。"""
    return max(expect * _MAX_TOTAL_RATIO, _MIN_TOTAL_SEC)


def _is_acceptable(sec: float, total: float, expect: float, n_chars: int) -> bool:
    """这条合成结果能不能用：**没被撑爆（空洞）且没被砍掉半句（截断）**。"""
    if total <= 0:
        return False
    if total > _bloat_limit(expect):
        return False
    return n_chars < _CHECK_MIN_CHARS or sec >= expect * _MIN_SPEECH_RATIO


def _quality_score(sec: float, total: float, expect: float) -> float:
    """都不达标时"谁更不烂"：先比有效语音长短，被空洞撑爆的按撑爆比例打折。"""
    limit = _bloat_limit(expect)
    return sec if total <= limit else sec * (limit / total)


def _text_lang(text: str) -> str:
    """按文本选 text_lang：**纯拉丁（英文）给 "en"，其余给 "zh"**。

    原来写死 "zh"（2026-10-11 修）：中文前端虽然能念夹带的英文词，但整句英文
    交给它会失真。参考音频是中文的，prompt_lang 保持 "zh" 不变（跨语言合成是支持的）。
    """
    if _CJK_RE.search(text):
        return "zh"
    return "en" if _LATIN_RE.search(text) else "zh"


async def _synthesize(reply_text: str, deadline: float | None = None) -> str | None:
    """GPT-SoVITS 合成 → wav 路径（带缓存）。未启用 / 失败 / 超预算返回 None。

    `deadline`：整个语音通道的截止时刻（`loop.time()` 口径，调用方在**进锁之前**算好）。
    卡住时"她一个字都不发"是最糟的结果——所以预算到点就放弃语音，让调用方回退文字。
    """
    if not get_voice_enabled():
        return None
    text = _tts_text(reply_text)
    if not text:
        return None
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    cached = VOICE_CACHE_DIR / f"voice_{h}.wav"
    expect = _expect_seconds(text)
    if cached.exists():
        # 缓存读回**必须复检**：不然历史坏条子（空洞/截断，3400+ 个文件里的那些）
        # 会被永久复用。复检顺手清一遍（幂等），不合格就重合成、覆盖它。
        sec, total = _clean_wav(cached)
        if _is_acceptable(sec, total, expect, len(text)):
            return str(cached)
        print(f"[语音] 缓存不合格（{cached.name}）：有效{sec:.1f}s/总{total:.1f}s vs "
              f"期望{expect:.1f}s，重新合成")
    ref_path = _resolve_ref(reply_text)
    if ref_path is None:
        print(f"⚠️ 参考音频缺失: {SOVITS_REF_DIR} 下没有 .wav（放 ref_voice.wav 即可起步）")
        return None
    prompt_text = _read_prompt_text(ref_path)
    text_lang = _text_lang(text)
    loop = asyncio.get_running_loop()
    best: tuple[float, bytes, float, float] | None = None   # (质量分, 字节, 有效秒, 总秒)
    for attempt, temperature in enumerate(_TTS_TEMPERATURES):
        left = None if deadline is None else deadline - loop.time()
        if left is not None and left < _TTS_MIN_ATTEMPT_SEC:
            # 预算不够：第一次就别开口，重试则不再试（等锁也可能把预算耗光）
            print(f"[语音] {'预算不足，跳过合成' if not attempt else '预算用完，不再重试'}"
                  f"（剩 {left:.0f}s）")
            break
        timeout = (_TTS_ATTEMPT_TIMEOUT_SEC if left is None
                   else max(3.0, min(_TTS_ATTEMPT_TIMEOUT_SEC, left)))
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(SOVITS_URL, json={
                    "text": text, "text_lang": text_lang,
                    "ref_audio_path": str(ref_path),
                    "prompt_text": prompt_text, "prompt_lang": "zh",
                    "top_k": 5, "top_p": 0.85, "temperature": temperature,
                    "speed_factor": 1.0, "media_type": "wav",
                    # 整句合成不分段：api 默认 cut5 分段会吞句子（复现：28字句丢"就放纵了一下"）
                    "text_split_method": "cut0",
                })
            if resp.status_code == 200 and resp.content:
                VOICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cached.write_bytes(resp.content)
                sec, total = _clean_wav(cached)   # 裁首尾静音 + 压句中空洞，并回报质量
                score = _quality_score(sec, total, expect)
                if best is None or score > best[0]:
                    best = (score, cached.read_bytes(), sec, total)
                if _is_acceptable(sec, total, expect, len(text)):
                    break
                # ⚠️ 这里必须看**两半**：只看 sec 会让"总长被空洞撑爆"的结果提前收工，
                # 重试从来没真正发生过（2026-10-11 修的就是这个）。
                print(f"[语音] 第{attempt+1}次 有效{sec:.1f}s/总{total:.1f}s vs 期望{expect:.1f}s，"
                      f"疑似截断或空洞，重试")
            else:
                print(f"⚠️ TTS 合成非200: {resp.status_code}")
        except Exception as e:
            print(f"⚠️ TTS 合成失败(第{attempt+1}次): {e}")
    if best is None:
        return None
    _, best_bytes, best_sec, best_total = best
    if not _is_acceptable(best_sec, best_total, expect, len(text)):
        print(f"⚠️ 多次合成仍不达标（有效{best_sec:.1f}s/总{best_total:.1f}s vs 期望{expect:.1f}s），"
              f"发最不烂的一次")
    cached.write_bytes(best_bytes)
    return str(cached)


def _to_qq_voice(wav_path: str) -> str | None:
    """wav → QQ 语音文件 URL。方案 A：NapCat 较新版本支持本地文件路径自动转。

    ⚠️ **必须是正斜杠**。`Path.resolve()` 在 Windows 上给出反斜杠路径，直接拼进
    `file:///` 会得到 `file:///D:\\a\\b.wav`——这**不是合法的 file URL**：
    NapCat 是 Node 写的，URL 解析会把反斜杠当成路径里的普通字符（不是分隔符），
    于是去找一个叫 `D:\a\b.wav` 的文件，找不到 → 语音发不出去。
    踩坑：接入方案「三.4」当初试通的是 `file:///D:/...`（正斜杠），实现时用了
    `.resolve()` 忘了转 `.as_posix()`，于是一直在发反斜杠版本。

    若 NapCat 不认 wav，需要装 silk 编码器转（方案 B，见接入方案「三.4」）。
    """
    if not wav_path:
        return None
    return "file:///" + Path(wav_path).resolve().as_posix()


async def send_voice(bot, target_id: str, is_private: bool, reply_text: str) -> bool:
    """朗读语音：合成 → 转 QQ 语音 → 发送。

    返回是否发送成功。成功 True（调用方不再发文字）；未启用/合成失败/**超预算**返回 False，
    由调用方回退文字分段——语音是优先通道不是唯一通道，绝不影响收到回复。

    预算是硬约束（2026-10-11 加）：卡住的 TTS 曾经能让她 **6 分钟一个字都不发**
    （timeout 120s × 最多 3 次重试，而文字兜底排在语音后面）。deadline 在**进锁之前**
    算好，所以"排队等锁"也算在预算里；到点就放弃语音、交给文字兜底。
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _TTS_BUDGET_SEC
    voice_file = None
    try:
        async with _tts_lock:
            left = deadline - loop.time()
            if left <= 0:
                print(f"[语音] 预算已被前一条合成占满（{_TTS_BUDGET_SEC:.0f}s），走文字")
                return False
            try:
                # 外层 wait_for 是**兜底**：就算 httpx 的超时没生效（或 _synthesize
                # 哪天被改坏），这一层也保证到点返回，不会一直挂着她不说话。
                wav_path = await asyncio.wait_for(
                    _synthesize(reply_text, deadline=deadline), left)
            except asyncio.TimeoutError:
                print(f"⚠️ 语音合成超预算（{_TTS_BUDGET_SEC:.0f}s），改走文字")
                return False
        if not wav_path:
            return False
        voice_file = _to_qq_voice(wav_path)
        if not voice_file:
            return False
        if is_private:
            await bot.send_private_msg(user_id=target_id,
                                       message=MessageSegment.record(file=voice_file))
        else:
            await bot.send_group_msg(group_id=target_id,
                                     message=MessageSegment.record(file=voice_file))
        print(f"[语音] 已发送语音 {len(_tts_text(reply_text))}字 file={voice_file}")
        return True
    except Exception as e:
        # 把 file= 一起打出来：语音发不出去时，第一个要确认的就是"发出去的到底是什么路径"
        # （曾经发的是反斜杠版 file:///D:\...，非法 URL，见 _to_qq_voice 的注释）
        print(f"⚠️ 语音发送失败（忽略，文字已回）: {e}")
        print(f"    发出去的 file={voice_file!r}")
        return False
