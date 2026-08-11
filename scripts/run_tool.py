#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线工具箱统一入口。

把所有离线脚本统一成一个命令行入口：
    python scripts/run_tool.py <工具> [参数]
    python scripts/run_tool.py --help
    python scripts/run_tool.py <工具> --help

7 个子命令：transcribe / analyze-pace / generate-vectors / generate-persona
           / precompute / pipeline / regression

旧脚本仍可直接运行（向后兼容），本入口为推荐用法。
"""
import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import _common  # noqa: E402


def _flatten_inputs(args_input):
    """把 -i 解析结果拉平，兼容 `-i a b` 与 `-i a -i b` 两种写法。

    子命令的 -i 是 nargs="+" + action="append"，多次出现会得到嵌套列表
    如 [["a"], ["b"]]，此处拍平成 ["a", "b"]。
    """
    flat = []
    for group in args_input:
        if isinstance(group, list):
            flat.extend(group)
        else:
            flat.append(group)
    return flat


# ==================== 子命令：transcribe ====================

def _add_transcribe(sub):
    p = sub.add_parser("transcribe", help="faster-whisper 语音转写（音频/文件夹 → 转写 JSON）")
    p.add_argument("input", nargs="?", default=None,
                   help="音频文件或文件夹；缺省扫描 assets/audio/")
    p.add_argument("-o", "--out-dir", default=None,
                   help="输出目录（默认 outputs/transcribe/）")
    p.add_argument("--language", default="zh", help="转写语言，默认 zh")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--model", default="medium")
    p.set_defaults(func=_run_transcribe)


def _run_transcribe(args):
    import importlib.util
    import os
    import subprocess

    # 自愈：当前解释器没有 faster-whisper 时，尝试用 whisper 环境重跑
    if importlib.util.find_spec("faster_whisper") is None:
        whisper_py = os.environ.get("WHISPER_PYTHON")
        if whisper_py:
            print("↳ 检测到 faster-whisper 缺失，改用 WHISPER_PYTHON 重跑…")
            sys.exit(subprocess.run([whisper_py, sys.argv[0], *sys.argv[1:]]).returncode)
        print("❌ 当前解释器未安装 faster-whisper。")
        print("   请用 whisper 虚拟环境运行，例如：")
        print("   D:\\whisper_env\\Scripts\\python.exe scripts/run_tool.py transcribe …")
        print("   或设置环境变量 WHISPER_PYTHON=D:\\whisper_env\\Scripts\\python.exe 后重试。")
        sys.exit(1)

    import transcribe_whisper
    src = Path(args.input) if args.input else _common.AUDIO_DIR
    out_dir = Path(args.out_dir) if args.out_dir else _common.OUT_TRANSCRIBE
    out_dir.mkdir(parents=True, exist_ok=True)
    transcribe_whisper.run(str(src), out_dir=str(out_dir),
                           language=args.language, device=args.device, model=args.model)


# ==================== 子命令：clean-transcript ====================

def _add_clean_transcript(sub):
    p = sub.add_parser("clean-transcript", help="转写清洗（合并碎片/修错字/加标点）")
    p.add_argument("-i", "--input", nargs="+", required=True, action="append",
                   help="原始转写 JSON，可传多个（-i a b 或 -i a -i b 均可）")
    p.add_argument("-o", "--output", default=None,
                   help="输出路径（默认 outputs/transcribe/cleaned.json）")
    p.add_argument("--turn-gap", type=float, default=2.0,
                   help="话轮聚合间隔阈值（秒），默认 2.0")
    p.set_defaults(func=_run_clean_transcript)


def _run_clean_transcript(args):
    import asyncio
    import clean_transcript
    out = Path(args.output) if args.output else _common.OUT_TRANSCRIBE / "cleaned.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(clean_transcript.run(_flatten_inputs(args.input), out, turn_gap=args.turn_gap))
    _common.report_saved(out)


# ==================== 子命令：convert-to-chat ====================

def _add_convert_to_chat(sub):
    p = sub.add_parser("convert-to-chat", help="直播原文 → QQ 聊天回复转化")
    p.add_argument("-i", "--input", nargs="+", required=True, action="append",
                   help="原文 JSON（可传多个，-i a b 或 -i a -i b 均可）")
    p.add_argument("-o", "--output", default=None,
                   help="输出路径（默认 outputs/transcribe/converted_chat.json）")
    p.set_defaults(func=_run_convert_to_chat)


def _run_convert_to_chat(args):
    import asyncio
    import convert_to_chat
    out = Path(args.output) if args.output else _common.OUT_TRANSCRIBE / "converted_chat.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(convert_to_chat.run(_flatten_inputs(args.input), out))
    _common.report_saved(out)


# ==================== 子命令：analyze-pace ====================

def _add_analyze_pace(sub):
    p = sub.add_parser("analyze-pace", help="直播节奏地图（支持多场合并、增量累积）")
    p.add_argument("-i", "--input", nargs="+", required=True, action="append",
                   help="转写 JSON，可传多个（每场一个，-i a b 或 -i a -i b 均可）")
    p.add_argument("-o", "--out-prefix", default=None,
                   help="输出前缀（默认 outputs/pace/pace_map，生成 .json + .md）")
    p.add_argument("--session", action="append", default=None,
                   help="每场标签，如 --session 第一场；个数与 --input 对齐")
    p.add_argument("--merge", action="store_true",
                   help="并入既有 pace_map 增量累积；不加则覆盖为单场结果")
    p.add_argument("--focus", default=None,
                   help="只精标这几个场景（逗号分隔，如 --focus 被调戏,被夸），其余话轮归'其他'；不传则全场景")
    p.add_argument("--turn-gap", type=float, default=2.0,
                   help="话轮聚合间隔阈值（秒），默认 2.0")
    p.set_defaults(func=_run_analyze_pace)


def _run_analyze_pace(args):
    import asyncio
    import analyze_pace
    out_prefix = Path(args.out_prefix) if args.out_prefix else _common.OUT_PACE / "pace_map"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    focus = [s.strip() for s in args.focus.split(",")] if args.focus else None
    asyncio.run(analyze_pace.run(
        input_paths=_flatten_inputs(args.input),
        out_prefix=out_prefix,
        sessions=args.session,
        merge=args.merge,
        turn_gap=args.turn_gap,
        focus=focus,
    ))


# ==================== 子命令：generate-vectors ====================

def _add_generate_vectors(sub):
    p = sub.add_parser("generate-vectors", help="场景化陈述 → 语料向量库")
    p.add_argument("-i", "--input", default=None,
                   help="场景化陈述 JSON（缺省用内置 RAW_CORPUS）")
    p.add_argument("-o", "--output", default=None,
                   help="输出路径（默认 data/corpus_vectors.json，机器人读取）")
    p.set_defaults(func=_run_generate_vectors)


def _run_generate_vectors(args):
    import asyncio
    import generate_vectors
    out = Path(args.output) if args.output else _common.VECTOR_FILE
    asyncio.run(generate_vectors.run(input_path=args.input, output_file=str(out)))
    _common.report_saved(out)
    if not args.output:
        _common.warn_fixed_path(out)


# ==================== 子命令：generate-persona ====================

def _add_generate_persona(sub):
    p = sub.add_parser("generate-persona", help="场景化陈述 → 人格三件套")
    p.add_argument("-i", "--input", default=None,
                   help="场景化陈述 JSON（缺省用内置 RAW_CORPUS）")
    p.add_argument("-o", "--out-dir", default=None,
                   help="输出目录（默认 persona/，机器人读取）")
    p.set_defaults(func=_run_generate_persona)


def _run_generate_persona(args):
    import asyncio
    import generate_persona
    out_dir = Path(args.out_dir) if args.out_dir else _common.PERSONA_DIR
    asyncio.run(generate_persona.run(input_path=args.input, out_dir=str(out_dir)))
    _common.report_saved(out_dir / "persona_traits.json", out_dir / "persona_styles.json",
                         out_dir / "persona_behaviors.json")
    if not args.out_dir:
        _common.warn_fixed_path(out_dir)


# ==================== 子命令：precompute ====================

# 类型 → (模块名, 默认输入, 默认输出)
_PRECOMPUTE_TARGETS = {
    "triggers":      ("precompute_trigger_vectors", _common.BEHAVIORS_FILE, _common.TRIGGER_VECTOR_FILE),
    "voice-samples": ("precompute_voice_sample_vectors", _common.VOICE_SAMPLES_FILE, _common.VOICE_SAMPLE_VECTOR_FILE),
    "phrases":       ("precompute_phrase_vectors", _common.PHRASES_FILE, _common.PHRASE_VECTOR_FILE),
    "preferences":   ("precompute_preference_vectors", _common.PREFERENCES_FILE, _common.PREFERENCE_VECTOR_FILE),
    "core-stories":  ("precompute_core_stories", _common.CORE_STORIES_FILE, _common.CORE_STORY_VECTOR_FILE),
}


def _add_precompute(sub):
    p = sub.add_parser("precompute", help="预计算向量缓存（trigger / 声音样本 / 措辞）")
    p.add_argument("target", nargs="?", choices=list(_PRECOMPUTE_TARGETS.keys()),
                   help="要预计算的类型")
    p.add_argument("--all", action="store_true", help="一次性预计算全部三类")
    p.add_argument("-i", "--input", default=None,
                   help="源 JSON（缺省 persona/ 下默认文件）")
    p.add_argument("-o", "--output", default=None,
                   help="输出路径（默认 data/ 下固定文件，机器人读取，勿改）")
    p.set_defaults(func=_run_precompute)


def _run_precompute(args):
    import asyncio
    import importlib
    targets = list(_PRECOMPUTE_TARGETS.keys()) if args.all else ([args.target] if args.target else [])
    if not targets:
        print("❌ 请指定 target 或 --all，例如：run_tool precompute --all")
        return
    saved = []
    for key in targets:
        mod_name, default_in, default_out = _PRECOMPUTE_TARGETS[key]
        mod = importlib.import_module(mod_name)
        src = Path(args.input) if args.input else default_in
        dst = Path(args.output) if args.output else default_out
        asyncio.run(mod.run(input_file=str(src), output_file=str(dst)))
        saved.append(dst)
    _common.report_saved(*saved)
    if not args.output:
        print("   ⚠️ 这些是机器人启动要读的固定位置")


# ==================== 子命令：generate-statements ====================

def _add_generate_statements(sub):
    p = sub.add_parser("generate-statements", help="从清洗素材生成场景化陈述（50-120字，供 corpus RAG）")
    p.add_argument("-i", "--input", nargs="+", required=True, action="append",
                   help="清洗后的素材 JSON，可传多个")
    p.add_argument("-o", "--output", default=None,
                   help="输出路径（默认 outputs/transcribe/generated_statements.json）")
    p.add_argument("--batch-size", type=int, default=50,
                   help="每批话轮数，默认 50")
    p.set_defaults(func=_run_generate_statements)


def _run_generate_statements(args):
    import asyncio
    import generate_statements
    out = Path(args.output) if args.output else _common.OUT_TRANSCRIBE / "generated_statements.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(generate_statements.run(_flatten_inputs(args.input), out, batch_size=args.batch_size))
    _common.report_saved(out)


# ==================== 子命令：mine-phrases ====================

def _add_mine_phrases(sub):
    p = sub.add_parser("mine-phrases", help="从清洗素材里批量挖掘措辞指纹（多维度，输出候选 JSON 待审批）")
    p.add_argument("-i", "--input", nargs="+", required=True, action="append",
                   help="清洗后的素材 JSON，可传多个")
    p.add_argument("-o", "--output", default=None,
                   help="输出路径（默认 outputs/transcribe/mined_phrases.json）")
    p.add_argument("--batch-size", type=int, default=60,
                   help="每批话轮数，默认 60")
    p.set_defaults(func=_run_mine_phrases)


def _run_mine_phrases(args):
    import asyncio
    import mine_phrases
    out = Path(args.output) if args.output else _common.OUT_TRANSCRIBE / "mined_phrases.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(mine_phrases.run(_flatten_inputs(args.input), out, batch_size=args.batch_size))
    _common.report_saved(out)


# ==================== 子命令：regression ====================

def _add_regression(sub):
    p = sub.add_parser("regression", help="回复『灵性』回归测试（A/B 对比）")
    p.add_argument("--ab", action="store_true",
                   help="A/B：V0(无声音样本) vs 当前(有样本) 对比")
    p.add_argument("-i", "--danmaku", default=None,
                   help="自定义弹幕 JSON/文本（缺省用内置虚构弹幕）")
    p.add_argument("-o", "--out-dir", default=None,
                   help="输出目录（默认 outputs/regression/）")
    p.set_defaults(func=_run_regression)


def _run_regression(args):
    import regression_test
    out_dir = Path(args.out_dir) if args.out_dir else _common.OUT_REGRESSION
    out_dir.mkdir(parents=True, exist_ok=True)
    regression_test.run(ab_mode=args.ab, danmaku_file=args.danmaku,
                        out_dir=str(out_dir))


# ==================== 子命令：bili-check ====================

def _add_bili_check(sub):
    p = sub.add_parser("bili-check", help="验证 B站 UID 直播状态 + 最新动态（不依赖 bot 运行）")
    p.add_argument("--uid", default=None,
                   help="B站 UID（缺省读 .env.prod 的 BILI_UID）")
    p.set_defaults(func=_run_bili_check)


def _run_bili_check(args):
    import sys
    import subprocess
    cmd = [sys.executable, str(SCRIPTS_DIR / "bili_check.py")]
    if args.uid:
        cmd += ["--uid", args.uid]
    return subprocess.call(cmd)


# ==================== 子命令：vision-test ====================

def _add_vision_test(sub):
    p = sub.add_parser("vision-test", help="验证 glm-4.6v 视觉描述（本地图片或 URL）")
    p.add_argument("-u", "--image", required=True, help="本地图片路径或 http(s) 图链")
    p.add_argument("--model", default=None, help="视觉模型（默认读 .env 或 glm-4.6v）")
    p.set_defaults(func=_run_vision_test)


def _run_vision_test(args):
    import sys
    import subprocess
    cmd = [sys.executable, str(SCRIPTS_DIR / "vision_test.py"), "-u", args.image]
    if args.model:
        cmd += ["--model", args.model]
    return subprocess.call(cmd)


# ==================== 子命令：mine-theme ====================

def _add_mine_theme(sub):
    p = sub.add_parser("mine-theme", help="主题素材挖掘（模型语义找素材，如立Flag秒打脸）")
    p.add_argument("-i", "--input", required=True, help="转写或清洗后的 JSON")
    p.add_argument("--theme", required=True, choices=["立Flag秒打脸", "失约被催", "被表白", "泛闲聊回应", "身份问答"], help="要挖的主题")
    p.add_argument("-o", "--output", default=None, help="输出路径")
    p.set_defaults(func=_run_mine_theme)


def _run_mine_theme(args):
    import asyncio
    import mine_theme
    asyncio.run(mine_theme.run(args.input, args.theme, args.output))


# ==================== 主入口 ====================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_tool",
        description="灰泽满离线工具箱：统一所有离线脚本的入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True, title="可用工具")
    _add_transcribe(sub)
    _add_clean_transcript(sub)
    _add_convert_to_chat(sub)
    _add_analyze_pace(sub)
    _add_generate_vectors(sub)
    _add_generate_persona(sub)
    _add_precompute(sub)
    _add_regression(sub)
    _add_mine_phrases(sub)
    _add_generate_statements(sub)
    _add_bili_check(sub)
    _add_vision_test(sub)
    _add_mine_theme(sub)
    return parser


def main():
    _common.ensure_utf8_stdout()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
