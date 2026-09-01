"""语音回复：GPT-SoVITS 合成 → QQ 语音消息（接入方案见 D:\\my_qq_bot\\语音接入方案.md）。

原则：短 + 语气型情绪回复 → 语音条；长回复/含内心戏括号 → 文字（互斥，不双发）。
TTS 在文字发出后的后台任务里跑，失败只跳过，绝不影响文字回复。
"""
import asyncio
import hashlib
import re
from pathlib import Path

import httpx

from nonebot.adapters.onebot.v11 import MessageSegment

from .constants import SOVITS_URL, SOVITS_REF_DIR, VOICE_CACHE_DIR
from .config import get_voice_enabled

# 不可语音化的"内心戏/动作"括号（TTS 表达不了 → 这些回复走文字）
_INNER_VOICE_PARENS = ("小声", "心虚", "捂脸", "揉眼睛", "叹气", "皱眉", "低头", "脸红")

_tts_lock = asyncio.Lock()   # 同时只合成一条（TTS 占 GPU，并发互相拖慢）
_voice_cache: dict[str, str] = {}


def should_voice(reply: str, max_len: int = 20) -> bool:
    """判断这条回复是否适合发语音。

    语音 = 短 + 无"内心戏括号" + 非数字/链接/长解释内容。
    长回复走文字分段，语音只发短句——天然互斥，不双发。
    """
    if not reply:
        return False
    if len(reply) > max_len:
        return False
    if any(p in reply for p in _INNER_VOICE_PARENS):
        return False
    if re.search(r"\d{2,}|http|https|网址|邮箱", reply):
        return False
    return True


def _tts_text(reply: str) -> str:
    """清洗：笑声括号→笑声文字，其余括号/动作剥掉（括号是文字专属表达）。"""
    t = reply
    for pat, rep in (("（笑）", "哈哈哈"), ("（笑死）", "哈哈哈哈"),
                     ("（偷笑）", "嘿嘿嘿"), ("（尬笑）", "哈"), ("（苦笑）", "呵")):
        t = t.replace(pat, rep)
    t = re.sub(r"[（【\[][^）】\]]*[）】\]]", "", t)
    t = re.sub(r"\s+", "", t)
    return t.strip()


# 参考音频情绪映射（粗粒度；细粒度如"小声/心虚"做不到，走文字）
EMOTION_REFS = {
    "开心": ("ref_happy.wav", "开心的参考文本"),
    "慵懒": ("ref_lazy.wav", "慵懒的参考文本"),
    "认真": ("ref_serious.wav", "认真的参考文本"),
}


def _pick_ref(reply: str):
    if any(p in reply for p in ("笑", "哈哈", "！", "～")):
        return EMOTION_REFS["开心"]
    if any(p in reply for p in ("困", "累", "唉", "……")):
        return EMOTION_REFS["慵懒"]
    return EMOTION_REFS["认真"]


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
    ref_audio, prompt_text = _pick_ref(reply_text)
    ref_path = Path(SOVITS_REF_DIR) / ref_audio
    if not ref_path.exists():
        print(f"⚠️ 参考音频缺失: {ref_path}（放 ref_happy/ref_lazy/ref_serious.wav）")
        return None
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


async def send_voice(bot, target_id: str, is_private: bool, reply_text: str) -> None:
    """后台语音任务：合成 → 转 QQ 语音 → 发送。失败只跳过，不影响文字。"""
    try:
        async with _tts_lock:
            wav_path = await _synthesize(reply_text)
            if not wav_path:
                return
        voice_file = _to_qq_voice(wav_path)
        if not voice_file:
            return
        if is_private:
            await bot.send_private_msg(user_id=target_id,
                                       message=MessageSegment.record(file=voice_file))
        else:
            await bot.send_group_msg(group_id=target_id,
                                     message=MessageSegment.record(file=voice_file))
        print(f"[语音] 已发送语音: {reply_text[:20]}")
    except Exception as e:
        print(f"⚠️ 语音发送失败（忽略，文字已回）: {e}")
