#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EP01：把逐行旁白拼成**一条**与时间轴严格对齐的整轨 WAV。

## 为什么要拼成整轨

Remotion 里用 151 个 `<Sequence><Audio/></Sequence>` 播放 151 个 wav，
在最终 ffmpeg 拼接阶段容易失败（本集实测：1080p 渲染到最后
`Conversion failed!`，1080p 不产出文件；0.5 倍预览也偶发同样错误）。

改成**一条整轨**之后：
  - Remotion 侧只剩 1 个 `<Audio>`，拼接阶段不再有 151 路输入；
  - 静音间隔由这里按 `ep01-narration.json` 的时间轴算，与画面帧严格对齐；
  - 渲染不再依赖 Remotion 的音频调度，产物可控、可复现。

## 对齐规则

对每一行：在它 `startMs` 处放音频，音频长度 = 实测时长；两条音频之间按时间轴
补静音。如果实测音频比它的槽位长，**不裁切**（宁可与下一句轻微叠一点），
因为裁切会咬掉字尾。

用法：
    python video/scripts/build-voiceover-track.py ep01
"""
import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "video"
FFMPEG = ROOT / "bin" / "ffmpeg.exe"
FFPROBE = ROOT / "bin" / "ffprobe.exe"


def probe_ms(path: Path) -> int:
    out = subprocess.run(
        [str(FFPROBE), "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return int(round(float(out) * 1000))


def wav_params(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as w:
        return w.getframerate(), w.getnchannels(), w.getsampwidth()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", nargs="?", default="ep01")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-parts", action="store_true",
                    help="保留中间的分段 wav（调试用）")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    narr = json.loads(
        (VIDEO / "src" / "episodes" / args.episode / f"{args.episode}-narration.json")
        .read_text(encoding="utf-8"))
    audio_dir = VIDEO / "public" / "audio" / args.episode
    parts_dir = audio_dir / "_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    total_ms = narr["durationMs"]
    lines = sorted(narr["lines"], key=lambda l: l["startMs"])

    # 参考参数取第一行音频（GPT-SoVITS 输出统一是 32kHz / 单声道 / 16bit）
    first = audio_dir / f"{lines[0]['id']}.wav"
    if not first.exists():
        print(f"❌ 缺少 {first}")
        return 2
    rate, channels, width = wav_params(first)
    print(f"轨道参数: {rate}Hz / {channels}ch / {width * 8}bit  总长 {total_ms / 1000:.1f}s")

    # ---- 逐行生成"静音 + 音频 + 静音"的定长分段，再整体拼接 ----
    # 每段严格等于该行的槽位长度：这样把所有段按顺序拼起来，就天然对齐时间轴，
    # 而且不需要计算 151 个 gap（少一层出错的地方）。
    pcm_total = total_ms * rate // 1000
    cursor = 0
    segments: list[Path] = []
    missing = 0
    late = 0

    for i, line in enumerate(lines):
        lid = line["id"]
        wav = audio_dir / f"{lid}.wav"
        slot = line["endMs"] - line["startMs"]
        if not wav.exists():
            missing += 1
            wav = None
        seg = parts_dir / f"{i:04d}.wav"

        # 该行相对轨道的起点（用帧粒度对齐，避免逐段累计误差）
        target = line["startMs"] * rate // 1000
        gap_before = max(0, target - cursor)

        if wav is None:
            # 缺音频：整段留静音，保持时间轴不断
            subprocess.run(
                [str(FFMPEG), "-y", "-loglevel", "error", "-f", "lavfi",
                 "-t", f"{slot / 1000:.3f}", "-i", f"anullsrc=r={rate}:cl=mono",
                 "-c:a", "pcm_s16le", str(seg)], check=True)
            segments.append(seg)
            cursor += slot * rate // 1000
            continue

        ms = probe_ms(wav)
        if ms > slot + 40:
            late += 1
        # ⚠️ 三个输入各自用 `-t` 限长、再用 concat 滤镜接起来。
        #    曾经把 `-t slot` 放在 `-i` 之后当**输出**选项——ffmpeg 把它当成
        #    第一个输入的限长，于是每段只剩 0.2 秒（整轨 608s 变 33s）。
        #    正确做法：`-t` 必须紧跟在它要限的那个 `-i` **之前**。
        #
        # ⚠️ 另一个坑：`-t 0.000` 会让 anullsrc 产出**零帧**输入，concat 直接失败
        #    （`CalledProcessError`，-28）。所以零长度的一侧要**整段略掉**，
        #    只在真的需要静音时才加那一路输入。
        gap_ms = gap_before * 1000 / rate
        tail_ms = max(0, slot - ms)
        inputs: list[str] = []
        labels: list[str] = []
        if gap_ms >= 5:
            inputs += ["-f", "lavfi", "-t", f"{gap_ms / 1000:.3f}", "-i", f"anullsrc=r={rate}:cl=mono"]
            labels.append(f"[{len(labels)}:a]")
        inputs += ["-i", str(wav)]
        labels.append(f"[{len(labels)}:a]")
        if tail_ms >= 5:
            inputs += ["-f", "lavfi", "-t", f"{tail_ms / 1000:.3f}", "-i", f"anullsrc=r={rate}:cl=mono"]
            labels.append(f"[{len(labels)}:a]")

        if len(labels) == 1:
            # 音频长度正好等于槽位：不需要拼接，直接重编码
            subprocess.run(
                [str(FFMPEG), "-y", "-loglevel", "error", "-i", str(wav),
                 "-ar", str(rate), "-ac", str(channels), "-c:a", "pcm_s16le", str(seg)],
                check=True)
        else:
            fc = "".join(labels) + f"concat=n={len(labels)}:v=0:a=1[out]"
            subprocess.run(
                [str(FFMPEG), "-y", "-loglevel", "error", *inputs,
                 "-filter_complex", fc, "-map", "[out]",
                 "-ar", str(rate), "-ac", str(channels), "-c:a", "pcm_s16le", str(seg)],
                check=True)
        segments.append(seg)
        cursor = target + slot * rate // 1000

    # 若最后一行之后还有余量（场景尾部呼吸），补一段静音到总长
    tail = max(0, pcm_total - cursor)
    if tail:
        seg = parts_dir / "9999_tail.wav"
        subprocess.run(
            [str(FFMPEG), "-y", "-loglevel", "error", "-f", "lavfi",
             "-t", f"{tail / rate:.3f}", "-i", f"anullsrc=r={rate}:cl=mono",
             "-c:a", "pcm_s16le", str(seg)], check=True)
        segments.append(seg)

    out = Path(args.out) if args.out else (audio_dir / f"{args.episode}-voiceover-track.wav")
    lst = parts_dir / "concat.txt"
    lst.write_text("".join(f"file '{s.as_posix()}'\n" for s in segments), encoding="utf-8")
    # ⚠️ 不能用 `-c copy`：各分段是独立编码的 wav，时间戳不连续，
    #    concat demuxer 会在第一个不连续处停住（本集实测：608s 的轨道输出只有 33s）。
    #    重编码成 PCM 并让 ffmpeg 重新生成时间戳，才能得到完整整轨。
    subprocess.run(
        [str(FFMPEG), "-y", "-loglevel", "error",
         "-fflags", "+genpts", "-f", "concat", "-safe", "0",
         "-i", str(lst), "-ar", str(rate), "-ac", str(channels),
         "-c:a", "pcm_s16le", str(out)], check=True)

    got = probe_ms(out)
    print(f"\n分段 {len(segments)} 个（缺音频 {missing} 行，音频超槽 {late} 行）")
    print(f"整轨时长 {got / 1000:.2f}s（时间轴 {total_ms / 1000:.2f}s，差 {got - total_ms}ms）")
    print(f"✅ {out.relative_to(VIDEO)}  ({out.stat().st_size / 1024 / 1024:.1f} MB)")

    # 再转一份 AAC：Remotion 渲染时用 .m4a（更小、拼接更稳），wav 留作母版。
    m4a = out.with_suffix(".m4a")
    subprocess.run(
        [str(FFMPEG), "-y", "-loglevel", "error", "-i", str(out),
         "-c:a", "aac", "-b:a", "160k", "-ar", str(rate), "-ac", str(channels), str(m4a)],
        check=True)
    m4a_ms = probe_ms(m4a)
    print(f"✅ {m4a.relative_to(VIDEO)}  ({m4a.stat().st_size / 1024 / 1024:.1f} MB, "
          f"{m4a_ms / 1000:.3f}s)")
    if abs(m4a_ms - got) > 50:
        print(f"⚠️ AAC 与 wav 时长差 {m4a_ms - got}ms（>50ms）")

    if not args.keep_parts:
        for s in parts_dir.glob("*.wav"):
            s.unlink()
        try:
            (parts_dir / "concat.txt").unlink()
            parts_dir.rmdir()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
