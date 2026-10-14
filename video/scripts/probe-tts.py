"""最小 TTS 探针：确认某个 GPT-SoVITS 端口是否真的在合成。

用法:
    python scripts/probe-tts.py [port] [--ref PATH] [--out FILE]

只做一次 /tts 请求并报告：HTTP 状态、响应字节数、WAV 头参数、时长、耗时。
不写缓存、不碰项目数据。
"""
import argparse
import json
import struct
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_REF = REPO / "assets" / "voice_refs" / "ref_voice.wav"
DEFAULT_PROMPT = "有没有可能居委会小区居委会也说会啊，虽然没说开的是小区居委会"
TEXT = "测试一下这条语音能不能正常合成出来，灰泽满在这里跟你说话。"


def read_wav_info(data: bytes) -> dict:
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return {"ok": False, "reason": "not a RIFF/WAVE payload", "head": data[:16].hex()}
    pos, fmt, frames = 12, None, 0
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", body, 0)
        elif cid == b"data":
            frames = size
        pos += 8 + size + (size & 1)
    if fmt is None:
        return {"ok": False, "reason": "no fmt chunk"}
    _, channels, rate, _, _, bits = fmt
    seconds = frames / (rate * channels * bits // 8) if rate else 0.0
    return {"ok": True, "channels": channels, "rate": rate, "bits": bits,
            "dataBytes": frames, "seconds": round(seconds, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", type=int, default=9880)
    ap.add_argument("--ref", default=str(DEFAULT_REF))
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref = Path(args.ref)
    if not ref.exists():
        print(f"参考音频不存在: {ref}")
        return 2

    payload = {
        "text": TEXT,
        "text_lang": "zh",
        "ref_audio_path": str(ref).replace("\\", "/"),
        "prompt_text": args.prompt,
        "prompt_lang": "zh",
        "top_k": 5,
        "top_p": 0.85,
        "temperature": 1.0,
        "speed_factor": 1.0,
        "media_type": "wav",
        "text_split_method": "cut0",
        "batch_size": 1,
        "parallel_infer": True,
        "split_bucket": True,
        "streaming_mode": False,
        "repetition_penalty": 1.35,
        "seed": 42,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = f"http://127.0.0.1:{args.port}/tts"
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json; charset=utf-8"})

    print(f"POST {url}  payload={len(body)}B  ref={ref.name}")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
            status = resp.status
    except Exception as exc:  # noqa: BLE001 - probe reports whatever happened
        print(f"请求失败: {type(exc).__name__}: {exc}")
        return 1
    elapsed = time.time() - t0

    print(f"HTTP {status}  {len(data)}B  耗时 {elapsed:.2f}s")
    info = read_wav_info(data)
    print(f"音频: {info}")
    if args.out:
        Path(args.out).write_bytes(data)
        print(f"已写入 {args.out}")

    if not info.get("ok"):
        return 1
    real = elapsed > 1.0 and info["seconds"] > 0.5
    print("判定: " + ("看起来在真实合成" if real else "疑似固定桩/空响应（太快或太短）"))
    return 0 if real else 1


if __name__ == "__main__":
    sys.exit(main())
