#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EP01 旁白合成：用灰泽满 GPT-SoVITS（HZM-SPEAK）逐行生成语音。

规格 §13.1：
  - 全片使用灰泽满声音，不切换中性配音；
  - 旁白拆成短行（单行 2.5~6 秒）；
  - 不在 Remotion 里用 toneFrequency 伪造情绪。

规格 §13.2：脚本必须支持 episode 参数，不能只读写根级 narration.csv / public/audio。

⚠️ 为什么用 Python 而不是 PowerShell 的 Invoke-WebRequest：
   PowerShell 会把请求体里的中文按本地代码页编码，服务端收到的是乱码
   （实测：返回一个固定 0.34 秒的桩音频）。这里显式 UTF-8 编码 JSON 并带 charset。

用法：
    python video/scripts/generate-voiceover.py ep01
    python video/scripts/generate-voiceover.py ep01 --port 9880 --out-dir public/audio/ep01
"""
import argparse
import csv
import json
import random
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "video"
FFPROBE = ROOT / "bin" / "ffprobe.exe"

DEFAULT_REF = ROOT / "assets" / "voice_refs" / "ref_voice.wav"
DEFAULT_PROMPT_FILE = ROOT / "assets" / "voice_refs" / "ref_voice.txt"


def probe_duration_ms(path: Path) -> int:
    out = subprocess.run(
        [str(FFPROBE), "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return int(round(float(out) * 1000))


def synthesize(port: int, text: str, ref: Path, prompt_text: str, out: Path,
               seed: int = -1, speed: float = 1.0, timeout: int = 300) -> None:
    payload = {
        "text": text,
        "text_lang": "zh",
        "ref_audio_path": str(ref).replace("\\", "/"),
        "prompt_text": prompt_text,
        "prompt_lang": "zh",
        "top_k": 5,
        "top_p": 0.85,
        "temperature": 1.0,
        # speed_factor：整片旁白基准提速。参考音频的语速偏慢，实测
        # 1.2 能在各条上稳定快约 15%（多 seed 中位数：3380→2880ms 等），
        # 听感更接近"private chat 里正常说话"，也不再出现明显的拖字。
        "speed_factor": speed,
        "media_type": "wav",
        # 整句合成不分段（和 bot 的 voice.py 一致）：cut5 会吞句子
        "text_split_method": "cut0",
        "batch_size": 1,
        "parallel_infer": True,
        "split_bucket": True,
        "streaming_mode": False,
        "repetition_penalty": 1.35,
        # ⚠️ seed 默认 -1（随机）。
        #    固定 seed=42 时，**某些句子会稳定合成出 0.42 秒的垃圾音频**
        #    （本集实测多句中招）。改成"不固定 seed + 合成后检查时长、不合格换 seed 重试"。
        "seed": seed,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/tts",
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if not data:
        raise RuntimeError("TTS 返回空响应")
    out.write_bytes(data)


def expected_ms(text: str, speed: float = 1.0) -> int:
    """期望朗读时长（毫秒）——与 redistribute-narration.mjs 用同一个模型。

    模型（固定开销 + 汉字 + 标点 + 英文字母）来自 123 行真实合成结果的拟合：
        audio_ms ≈ 332 + 200×汉字 + 355×标点 + 71×英文字母        （speed=1.0）

    其中**固定开销与标点停顿不随 speed_factor 缩放**（它们是静音与韵律停顿），
    只有汉字/字母的发音段会按 speed 缩短。所以要分开算：

        audio_ms(speed) ≈ (332 + 355×标点) + (200×汉字 + 71×字母) / speed
    """
    han = len(re.findall(r"[\u4e00-\u9fa5]", text))
    punc = len(re.findall(r"[，。、；：？！…]", text))
    lat = sum(len(m) for m in re.findall(r"[A-Za-z0-9_]+", text))
    fixed = 332 + 355 * punc
    spoken = 200 * han + 71 * lat
    return int(fixed + spoken / max(0.5, speed))


def synthesize_checked(port: int, text: str, ref: Path, prompt_text: str, out: Path,
                       speed: float = 1.0, attempts: int = 6) -> tuple[int, list[str]]:
    """合成并检查质量，不合格就换 seed 重试。

    两个方向都要拦（本集实测两种都出现过）：
      · **退化**（太短）：seed 抽坏 → 0.34~0.5 秒的垃圾音频，或明显缺字。判据 < 期望的 55%。
      · **拖慢**（太长）：seed 抽到拖字 → 比期望长 40% 以上（实测有行 7.1 秒而期望 3.4 秒）。
        这类听起来就是"语速太慢不自然"，必须重抽。

    ⚠️ **不合格就是不合格，绝不"取最不烂的一次"发出去**（本集踩过）：
       原来是"多次都不合格 → 取最接近期望的那一次"，结果 closing-3
       （期望 5300ms）被写成了 2140ms 的退化音频——字幕、时间轴全对，
       只有声音是坏的，而且**校验脚本也放它过了**（因为它是"最接近的"）。
       现在改成：全部不合格就抛错，让调用方显式看到并决定（重跑 / 降速 / 拆句），
       宁可留下上一版音频也不要写进一条坏音。
    """
    want = expected_ms(text, speed)
    lo, hi = want * 0.55, want * 1.40
    notes: list[str] = []
    for i in range(attempts):
        seed = -1 if i == 0 else random.randint(1, 10**6)
        synthesize(port, text, ref, prompt_text, out, seed=seed, speed=speed)
        ms = probe_duration_ms(out)
        if lo <= ms <= hi:
            if notes:
                notes.append(f"第{i + 1}次(seed={seed})合格 {ms}ms")
            return ms, notes
        why = "退化" if ms < lo else "拖慢"
        notes.append(f"第{i + 1}次(seed={seed}){why} {ms}ms（期望{want}ms）")
    raise RuntimeError(
        f"「{text}」连续 {attempts} 次都没合成到合格区间"
        f"（期望 {want}ms，容差 {int(lo)}~{int(hi)}ms）：" + "；".join(notes)
        + "。请重跑，或把该句再折短，或降低 --speed。"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", nargs="?", default="ep01")
    ap.add_argument("--port", type=int, default=9880,
                    help="GPT-SoVITS api_v2.py 端口（bot 用的是 9880）")
    ap.add_argument("--ref", default=str(DEFAULT_REF))
    ap.add_argument("--prompt-text", default=None,
                    help="参考音频转写；默认读 ref_voice.txt")
    ap.add_argument("--only", default=None, help="只合成这些 line id（逗号分隔）")
    ap.add_argument("--speed", type=float, default=1.2,
                    help="整片旁白语速（GPT-SoVITS speed_factor）；默认 1.2")
    ap.add_argument("--measure-only", action="store_true",
                    help="不合成，只重新测量已有 wav 并重建报告（改完文案核对时用）")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    csv_path = VIDEO / "narration" / f"{args.episode}.csv"
    if not csv_path.exists():
        print(f"❌ 找不到 {csv_path}；先跑 npm run {args.episode}:captions")
        return 2

    audio_dir = VIDEO / "public" / "audio" / args.episode
    audio_dir.mkdir(parents=True, exist_ok=True)
    report_path = audio_dir / "voiceover-report.json"

    ref = Path(args.ref)
    if not ref.exists():
        print(f"❌ 参考音频不存在: {ref}")
        return 2
    prompt_text = args.prompt_text
    if prompt_text is None:
        prompt_text = (DEFAULT_PROMPT_FILE.read_text(encoding="utf-8").strip()
                       if DEFAULT_PROMPT_FILE.exists() else "")

    import csv
    with csv_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    only = set(args.only.split(",")) if args.only else None
    if only:
        rows = [r for r in rows if r["line_id"] in only]

    # ⚠️ --only 时**合并**进已有报告，不能覆盖：覆盖会把没重跑的那些行从报告里删掉，
    #    下游 fit-voice 就看不到它们的实测时长（本集踩过：修 54 行后报告只剩 54 条）。
    prior: dict[str, dict] = {}
    if only and report_path.exists():
        try:
            prior = {r["line_id"]: r for r in json.loads(report_path.read_text(encoding="utf-8"))}
        except (json.JSONDecodeError, OSError):
            prior = {}

    print(f"参考音频: {ref.name}（{ref.stat().st_size} B）")
    print(f"参考转写: {prompt_text}")
    print(f"待合成: {len(rows)} 行 → {audio_dir.relative_to(VIDEO)}\n")

    report = []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        lid = row["line_id"]
        text = row["text"]
        out = audio_dir / f"{lid}.wav"
        slot = int(row["duration_ms"])
        try:
            if args.measure_only:
                if not out.exists():
                    raise FileNotFoundError(out)
                ms = probe_duration_ms(out)
                notes = []
            else:
                ms, notes = synthesize_checked(args.port, text, ref, prompt_text, out,
                                               speed=args.speed)
            over = ms - slot
            flag = "" if over <= 0 else f"  ⚠ 超出槽位 {over}ms"
            note = ("  ← " + "；".join(notes)) if notes else ""
            print(f"  [{i:>2}/{len(rows)}] {lid:<11} {ms:>5}ms / 槽 {slot:>5}ms  {text[:26]}{flag}{note}")
            report.append({
                "line_id": lid, "scene_id": row["scene_id"],
                "start_ms": int(row["start_ms"]), "end_ms": int(row["end_ms"]),
                "slot_ms": slot, "audio_ms": ms, "overrun_ms": over,
                "expected_ms": expected_ms(text, args.speed),
                "speed": args.speed,
                "retry_notes": notes,
                "text": text, "audio_file": f"public/audio/{args.episode}/{lid}.wav",
            })
        except Exception as exc:  # noqa: BLE001 - 报告哪一行坏了，继续跑
            print(f"  [{i:>2}/{len(rows)}] {lid:<11} ❌ {type(exc).__name__}: {exc}")
            report.append({
                "line_id": lid, "scene_id": row["scene_id"],
                "start_ms": int(row["start_ms"]), "end_ms": int(row["end_ms"]),
                "slot_ms": slot, "audio_ms": None, "overrun_ms": None,
                "text": text, "error": str(exc),
            })

    merged = dict(prior)
    for r in report:
        merged[r["line_id"]] = r
    out_report = sorted(merged.values(), key=lambda r: (r["scene_id"], r["line_id"]))
    report_path.write_text(json.dumps(out_report, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = [r for r in report if r.get("audio_ms")]
    total = sum(r["audio_ms"] for r in ok)
    over = [r for r in ok if r["overrun_ms"] > 0]
    print(f"\n合成完成 {len(ok)}/{len(report)} 行，用时 {time.time() - t0:.0f}s")
    print(f"音频总长 {total / 1000:.1f}s（旁白纯时长）")
    print(f"报告: {report_path.relative_to(VIDEO)}")
    if over:
        print(f"⚠️ {len(over)} 行超出原槽位；下一步跑 fit-voice 会按真实音频顺延时间轴：")
        for r in over[:10]:
            print(f"   {r['line_id']} 槽{r['slot_ms']}ms → 实际{r['audio_ms']}ms (+{r['overrun_ms']})")
    if len(ok) < len(report):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
