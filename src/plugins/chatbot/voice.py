"""语音回复：GPT-SoVITS 合成 → QQ 语音消息（接入方案见 D:\\my_qq_bot\\语音接入方案.md）。

原则：成句(≥20字)回复朗读成语音条；短敷衍词/含内心戏括号 → 打字文字（互斥，不双发）。
灰泽满实际回复多为 15~45 字一句（样本最长 60），语音覆盖"完整成句"档，朗读模型擅成句。
语音在回复路径里优先尝试，失败自动回退文字分段——绝不影响收到回复。
"""
import array
import asyncio
import hashlib
import math
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

_tts_lock = asyncio.Lock()   # 同时只合成一条（TTS 占 GPU，并发互相拖慢）


def _is_greeting(reply: str) -> bool:
    """是否命中短寒暄/高情绪词表（只判断词，长度由调用方管）。"""
    r = reply.strip(" 　～~。！!？?,，").lower()
    return any(g in r for g in _GREETING_PHRASES)


def should_voice(reply: str, min_len: int = VOICE_MIN_LEN,
                 max_len: int = VOICE_MAX_LEN) -> bool:
    """判断这条回复是否适合朗读成语音条。

    长度/寒暄判断用**实际会读出来的文本**（_tts_text 剥括号后）——避免"原文带括号尾巴够 20 字、
    读出来只剩短句"的误触发。内心戏括号/数字/链接铁则仍针对原文（括号是文字专属表达）。
    语音 = 剥括号后成句(≥min_len) 或 短寒暄命中；短敷衍词("在呢")/含内心戏括号/链接仍打字。
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
    in_range = min_len <= n <= max_len
    # 短寒暄命中 → 突破长度下限（≤min_len 的也语音）；超长仍走文字分段
    greeting_ok = (not in_range) and n < min_len and _is_greeting(spoken)
    return in_range or greeting_ok


def _tts_text(reply: str) -> str:
    """清洗：笑声括号→笑声文字，其余括号/动作剥掉（括号是文字专属表达）。"""
    t = reply
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


# 判定静音的能量下限：语音窗远高于此（实测语音≈数千，GPT-SoVITS 静音尾≈150）
_SILENCE_RMS = 300
_SILENCE_WIN_SEC = 0.05     # 能量统计窗秒数
_SILENCE_MARGIN_SEC = 0.05  # 裁完留的边距，防咬掉字头字尾


def _trim_silence(wav_path: Path) -> None:
    """裁掉 wav 首尾的静音段。防御：个别坏合成（尤其参考音频带尾静音时）会输出
    一段"就几个字 + 十几秒空尾"，发到 QQ 很难受。仅处理 16-bit PCM 单声道，
    其他格式或解析失败就原样跳过，绝不影响发送。
    """
    try:
        with wave.open(str(wav_path), "rb") as w:
            if w.getsampwidth() != 2 or w.getnchannels() != 1:
                return
            rate = w.getframerate()
            data = w.readframes(w.getnframes())
        a = array.array("h")
        a.frombytes(data)
        if not len(a):
            return
        hop = max(1, int(rate * _SILENCE_WIN_SEC))

        def _loud(lo: int, hi: int) -> bool:
            seg = a[lo:hi]
            if not seg:
                return False
            return math.sqrt(sum(x * x for x in seg) / len(seg)) > _SILENCE_RMS

        first = last = None
        for start in range(0, len(a), hop):
            if _loud(start, min(start + hop, len(a))):
                if first is None:
                    first = start
                last = start
        if first is None or last is None:
            return  # 整段无声，别乱裁
        margin = int(rate * _SILENCE_MARGIN_SEC)
        lo = max(0, first - margin)
        hi = min(len(a), last + hop + margin)
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(a[lo:hi].tobytes())
    except Exception:
        pass  # 裁不掉就发原件，别把语音裁没了


async def _synthesize(reply_text: str) -> str | None:
    """GPT-SoVITS 合成 → wav 路径（带缓存）。未启用/失败返回 None。"""
    if not get_voice_enabled():
        return None
    text = _tts_text(reply_text)
    if not text:
        return None
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    cached = VOICE_CACHE_DIR / f"voice_{h}.wav"
    if cached.exists():
        return str(cached)
    ref_path = _resolve_ref(reply_text)
    if ref_path is None:
        print(f"⚠️ 参考音频缺失: {SOVITS_REF_DIR} 下没有 .wav（放 ref_voice.wav 即可起步）")
        return None
    prompt_text = _read_prompt_text(ref_path)
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(SOVITS_URL, json={
                "text": text, "text_lang": "zh",
                "ref_audio_path": str(ref_path),
                "prompt_text": prompt_text, "prompt_lang": "zh",
                "top_k": 5, "top_p": 0.85, "temperature": 0.8,
                "speed_factor": 1.0, "media_type": "wav",
            })
        if resp.status_code == 200 and resp.content:
            VOICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(resp.content)
            _trim_silence(cached)   # 裁掉首尾静音（防"几个字+长空尾"的坏合成）
            return str(cached)
        print(f"⚠️ TTS 合成非200: {resp.status_code}")
    except Exception as e:
        print(f"⚠️ TTS 合成失败: {e}")
    return None


def _to_qq_voice(wav_path: str) -> str | None:
    """wav → QQ 语音文件。方案 A：NapCat 较新版本支持本地文件路径自动转。

    若 NapCat 不认 wav，需要装 silk 编码器转（方案 B，见接入方案「三.4」）。
    """
    return f"file:///{Path(wav_path).resolve()}"


async def send_voice(bot, target_id: str, is_private: bool, reply_text: str) -> bool:
    """朗读语音：合成 → 转 QQ 语音 → 发送。

    返回是否发送成功。成功 True（调用方不再发文字）；未启用/合成失败返回 False，
    由调用方回退文字分段——语音是优先通道不是唯一通道，绝不影响收到回复。
    """
    try:
        async with _tts_lock:
            wav_path = await _synthesize(reply_text)
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
        print(f"[语音] 已发送语音 {len(_tts_text(reply_text))}字: {reply_text}")
        return True
    except Exception as e:
        print(f"⚠️ 语音发送失败（忽略，文字已回）: {e}")
        return False
