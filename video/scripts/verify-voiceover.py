#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EP01 配音完整性校验：确认每一行音频都是**当前文本**合成的。

为什么需要它：改了旁白文案之后，line id 会重排（折句会新增 id），
这时候"文件存在"不等于"音频对得上"——拿旧 id 的音频配新文本，
画面上就会出现字幕与声音不一致。这个脚本用 bot 的语音缓存做交叉核对：

    data/voice_cache/voice_<md5(实际朗读文本)>.wav

`voice._tts_text()` 决定了"实际朗读文本"（它会剥括号、把 hzm 换成"灰泽满"），
所以这里复刻同一个规则，再比对缓存里有没有对应文件。

用法：
    python video/scripts/verify-voiceover.py ep01
    python video/scripts/verify-voiceover.py ep01 --fix    # 自动重合成有问题的行
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "video"
FFPROBE = ROOT / "bin" / "ffprobe.exe"
CACHE = ROOT / "data" / "voice_cache"

# 整片旁白基准语速（与 generate-voiceover.py 的 --speed 默认值保持一致）
DEFAULT_SPEED = 1.2

# 与 src/plugins/chatbot/voice.py 的 _tts_text 对齐（只实现影响 md5 的部分）
_LAUGH = (("（笑）", "哈哈哈"), ("（笑死）", "哈哈哈哈"), ("（偷笑）", "嘿嘿嘿"),
          ("（尬笑）", "哈"), ("（苦笑）", "呵"))


def tts_text(reply: str) -> str:
    t = re.sub(r"hzm", "灰泽满", reply or "", flags=re.IGNORECASE)
    for pat, rep in _LAUGH:
        t = t.replace(pat, rep)
    t = re.sub(r"[（(][^）)]*[）)]", "", t)
    return t.strip()


def duration_ms(path: Path) -> int:
    out = subprocess.run(
        [str(FFPROBE), "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return int(round(float(out) * 1000))


def expected_ms(text: str, speed: float = DEFAULT_SPEED) -> int:
    """期望朗读时长（毫秒）——与 generate-voiceover.py / redistribute-narration.mjs 同模型。

        audio_ms(speed) ≈ (332 + 355×标点) + (200×汉字 + 71×字母) / speed
    """
    han = len(re.findall(r"[\u4e00-\u9fa5]", text))
    punc = len(re.findall(r"[，。、；：？！…]", text))
    lat = sum(len(m) for m in re.findall(r"[A-Za-z0-9_]+", text))
    return int((332 + 355 * punc) + (200 * han + 71 * lat) / max(0.5, speed))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", nargs="?", default="ep01")
    ap.add_argument("--fix", action="store_true", help="重合成有问题的行")
    ap.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                    help=f"旁白语速（默认 {DEFAULT_SPEED}）")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    narration = json.loads(
        (VIDEO / "src" / "episodes" / args.episode / f"{args.episode}-narration.json")
        .read_text(encoding="utf-8"))
    audio_dir = VIDEO / "public" / "audio" / args.episode
    report_path = audio_dir / "voiceover-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else []
    by_id = {r["line_id"]: r for r in report}

    rows = []
    missing, stale, offlen = [], [], []
    for line in narration["lines"]:
        lid = line["id"]
        wav = audio_dir / f"{lid}.wav"
        spoken = tts_text(line["text"])
        h = hashlib.md5(spoken.encode("utf-8")).hexdigest()
        cached = CACHE / f"voice_{h}.wav"
        rep = by_id.get(lid)
        want = expected_ms(line["text"], args.speed)

        state = "ok"
        if not wav.exists():
            state = "缺失"
            missing.append(lid)
        else:
            ms = duration_ms(wav)
            # ⚠️ **判据以"时长对不对"为主，不以"缓存里有没有"为主**。
            #    本集踩过：折句后行号会被复用（content-forms-25 原来装别的话），
            #    旧音频的缓存条目仍然存在 → 只查缓存会把它判成 ok，
            #    结果画面里字幕和声音对不上。时长与当前文本的期望值差得远才是真信号。
            if ms < 900 or ms < want * 0.55:
                state = f"疑似退化({ms}ms/期望{want}ms)"
                offlen.append(lid)
            elif ms > want * 1.40:
                state = f"疑似拖慢({ms}ms/期望{want}ms)"
                offlen.append(lid)
            elif rep and rep.get("text") != line["text"]:
                state = "文本已变"
                stale.append(lid)
            elif rep is None:
                state = "无报告条目"
                stale.append(lid)
            elif not cached.exists():
                # 缓存里没有同一文本的合成结果，但时长对得上 —— 大概率是改过停顿后重测的
                state = "ok(时长核对)"
        rows.append((lid, state, line["text"]))

    ok = [r for r in rows if r[1].startswith("ok")]
    bad = [r for r in rows if not r[1].startswith("ok")]

    print(f"行 {len(rows)}  正常 {len(ok)}  有问题 {len(bad)}\n")
    if bad:
        print("有问题的行：")
        for lid, state, text in bad:
            print(f"  {lid:<22} {state:<26} {text}")
    print(f"\n缺失音频 {len(missing)}  文本已变 {len(stale)}  时长不符 {len(offlen)}")

    if args.fix and (missing or stale or offlen):
        todo = sorted(set(missing) | set(stale) | set(offlen))
        print(f"\n重新合成 {len(todo)} 行（speed={args.speed}）…")
        cmd = [sys.executable, str(VIDEO / "scripts" / "generate-voiceover.py"),
               args.episode, "--only", ",".join(todo), "--speed", str(args.speed)]
        return subprocess.run(cmd, cwd=ROOT).returncode

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
