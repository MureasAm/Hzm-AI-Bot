#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把清洗后的直播素材安全地补进 persona 文件。

支持两条独立流水线：

    corpus       清洗素材 → statement 候选 → 追加 statement_final → 补钩子 → 重建 corpus 向量
    behaviors    清洗素材 → 直播转聊天 → 行为提取 → 合并 behaviors

（voice-samples / phrases 两条已于 2026-10-04 下线，样本并入 behaviors。）

默认是 **prepare 模式**：只生成候选 JSON 和 review.md，不改 persona。
确认候选后加 `--apply` 才写回 persona，并自动备份原文件。

也支持直接从音频开始：
    --audio assets/audio/xxx.m4a
"""
import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import _common


TARGETS = ("corpus", "behaviors")   # voice-samples / phrases 已下线（2026-10-04：并入 behaviors）
PIPELINE_DIR = _common.OUTPUTS_DIR / "persona_pipeline"
STATEMENT_FILE = _common.WORLD_DIR / "statement_final.json"
_SELF_PRONOUN_RE = re.compile(r"(?<!其)[她他](?!们|俩)")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def _clear_output(path: Path) -> None:
    path = Path(path)
    if path.exists():
        path.unlink()


def _norm(text):
    return "".join(str(text or "").split())


def _items(data, key):
    if isinstance(data, dict):
        value = data.get(key, [])
        return list(value) if isinstance(value, list) else []
    return list(data) if isinstance(data, list) else []


def _safe_session(value: str) -> str:
    value = re.sub(r"[^\w.-]+", "_", (value or "").strip(), flags=re.UNICODE)
    return value.strip("._") or "session"


def _unique_append(items, item, key):
    normalized = {_norm(key(x)) for x in items}
    if _norm(key(item)) not in normalized:
        items.append(item)
        return True
    return False


def _has_self_pronoun(text: str) -> bool:
    """灰泽满出现后是否还有她/他；具体是否复指本人交给 LLM 判断。"""
    text = text or ""
    starts = [p for p in (text.find("灰泽满"), text.lower().find("hzm")) if p >= 0]
    if not starts:
        return False
    return bool(_SELF_PRONOUN_RE.search(text[min(starts):]))


async def _rewrite_corpus_self_pronouns(candidates: list, stage: Path) -> tuple[list, int]:
    """只改写确实用她/他复指灰泽满的 corpus 候选。"""
    flagged = [(i, item["statement"]) for i, item in enumerate(candidates)
               if _has_self_pronoun(item.get("statement", ""))]
    if not flagged:
        return candidates, 0

    from openai import AsyncOpenAI

    key = _common.get_api_key("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("需要 OPENAI_API_KEY 才能改写 corpus 自指代词")
    client = AsyncOpenAI(api_key=key, base_url=_common.get_openai_base_url())
    prompt = """下面是"灰泽满"的第三人称背景记忆。

只处理一个问题：**先写“灰泽满”，后面再用“她/他”指灰泽满**。
这种句式会让模型把她当成另一个人。

改写规则：
- 如果“她/他”指灰泽满：删除该代词，或改成“灰泽满”；优先删主语，不要堆名字。
- 如果“她/他”指女同学、满妈、同期等别人：保持原样。
- 只改指代问题，不改事实、不加信息、不删信息。
- 保持第三人称、80 字左右、不要引号、不要台词。

输入：
{items}

只输出 JSON：
{{"replacements": [{{"index": 1, "statement": "改写后的句子", "changed": true}}]}}
只列确实需要改的 index；指别人的她/他不要列入。"""

    replacements = {}
    batch_size = 15
    for start in range(0, len(flagged), batch_size):
        batch = flagged[start:start + batch_size]
        numbered = "\n".join(f"{i + 1}. {text}" for i, (_, text) in enumerate(batch))
        resp = await client.chat.completions.create(
            model=_common.get_model_name(),
            messages=[{"role": "user", "content": prompt.format(items=numbered)}],
            temperature=0,
            max_tokens=4000,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = (resp.choices[0].message.content or "").strip()
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in content:
            content = content.split("```", 1)[1].split("```", 1)[0].strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            continue
        for row in data.get("replacements", []):
            try:
                local_idx = int(row.get("index")) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= local_idx < len(batch):
                global_idx = batch[local_idx][0]
                new_text = str(row.get("statement", "")).strip()
                if new_text and row.get("changed") is not False:
                    replacements[global_idx] = new_text

    if replacements:
        _write_json(stage / "corpus_self_pronoun_before.json",
                    {"statements": [{"index": i, "statement": text} for i, text in flagged]})
        for i, text in replacements.items():
            candidates[i]["statement"] = text
        _write_json(stage / "corpus_self_pronoun_after.json",
                    {"statements": [{"index": i, "statement": text}
                                    for i, text in sorted(replacements.items())]})
    return candidates, len(replacements)


def _prepare_stage(session: str) -> Path:
    path = PIPELINE_DIR / _safe_session(session)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _backup(paths, stage: Path) -> Path:
    backup = stage / "backup" / time.strftime("%Y%m%d-%H%M%S")
    backup.mkdir(parents=True, exist_ok=True)
    for path in paths:
        path = Path(path)
        if path.exists():
            shutil.copy2(path, backup / path.name)
    return backup


def _build_corpus_hooks() -> None:
    """只补 corpus_keywords 里缺失的索引；已有手工钩子不会被覆盖。"""
    script = Path(__file__).with_name("build_corpus_keywords.py")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(_common.PROJECT_ROOT),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("corpus 已写回，但 corpus_keywords 钩子生成失败")


async def _run_audio_steps(audio: str, stage: Path, turn_gap: float, device: str,
                           whisper_model: str, whisper_python: str | None = None) -> Path:
    """音频 → 转写 JSON → 清洗 JSON；返回清洗文件路径。"""
    import importlib.util

    audio_path = Path(audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"音频不存在：{audio_path}")

    transcribed = stage / f"{audio_path.stem}_transcribed.json"
    _clear_output(transcribed)
    if whisper_python or importlib.util.find_spec("faster_whisper") is None:
        if not whisper_python:
            raise RuntimeError(
                "当前环境没有 faster-whisper。请设置 WHISPER_PYTHON 指向 whisper 环境。"
            )
        run_tool = Path(__file__).with_name("run_tool.py")
        subprocess.run(
            [whisper_python, str(run_tool), "transcribe", str(audio_path),
             "-o", str(stage), "--device", device, "--model", whisper_model],
            check=True,
        )
    else:
        import transcribe_whisper
        transcribe_whisper.run(
            str(audio_path), out_dir=str(stage), device=device, model=whisper_model
        )

    if not transcribed.exists():
        candidates = sorted(stage.glob("*_transcribed.json"))
        if not candidates:
            raise RuntimeError("转写完成但没有找到 *_transcribed.json")
        transcribed = candidates[-1]

    import clean_transcript
    cleaned = stage / "cleaned.json"
    _clear_output(cleaned)
    await clean_transcript.run([str(transcribed)], str(cleaned), turn_gap=turn_gap)
    if not cleaned.exists():
        raise RuntimeError("转写清洗失败，没有生成 cleaned.json")
    return cleaned


async def _prepare_inputs(args, stage: Path) -> list[str]:
    if args.audio:
        cached = stage / "cleaned.json"
        if cached.exists() and not args.refresh:
            print(f"↳ 复用已清洗素材：{cached}")
            return [str(cached)]
        cleaned = await _run_audio_steps(
            args.audio, stage, args.turn_gap, args.device, args.whisper_model,
            args.whisper_python or os.environ.get("WHISPER_PYTHON"),
        )
        return [str(cleaned)]
    if not args.input:
        raise ValueError("必须提供 --audio 或 -i/--input")
    paths = [Path(p) for p in args.input]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("输入不存在：" + ", ".join(missing))
    return [str(p) for p in paths]


async def _prepare_corpus(cleaned_paths, stage: Path, batch_size: int):
    import generate_statements

    raw = stage / "corpus_raw.json"
    _clear_output(raw)
    await generate_statements.run(cleaned_paths, raw, batch_size=batch_size)
    if not raw.exists():
        raise RuntimeError("statement 生成失败，没有写出 corpus_raw.json")
    data = _read_json(raw)
    texts = data.get("statements", []) if isinstance(data, dict) else data
    candidates, seen = [], set()
    for item in texts:
        text = (item.get("statement") if isinstance(item, dict) else str(item)).strip()
        key = _norm(text)
        if text and key not in seen:
            seen.add(key)
            candidates.append({"statement": text})
    candidates, rewritten = await _rewrite_corpus_self_pronouns(candidates, stage)
    if rewritten:
        print(f"↳ corpus 自指代词已改写 {rewritten}/{len(candidates)} 条")
    _write_json(stage / "corpus_candidates.json",
                {"source_files": cleaned_paths, "statements": candidates})
    return candidates


async def _convert_to_chat(cleaned_paths, stage: Path):
    import convert_to_chat

    output = stage / "converted_chat.json"
    _clear_output(output)
    await convert_to_chat.run(cleaned_paths, output)
    if not output.exists():
        raise RuntimeError("直播转聊天失败，没有写出 converted_chat.json")
    data = _read_json(output)
    return data if isinstance(data, list) else []


def _scene_to_name(scene: str) -> str:
    mapping = {
        "被夸": "被夸时嘴硬否认",
        "被质疑": "被质疑时心虚辩解",
        "被戳穿": "被质疑时心虚辩解",
        "被越界": "被越界时冷静推开",
        "被调戏": "被越界时冷静推开",
        "冷场": "冷场时主动自爆填补",
        "立Flag": "立Flag后秒打脸",
        "承诺": "立Flag后秒打脸",
        "抛梗": "主动抛梗与预判调侃",
        "推进": "主动抛梗与预判调侃",
        "感性": "感性流露后迅速缩回",
        "脆弱": "被真心戳中时别扭缩回",
        "失约": "失约被催时认栽滑跪",
        "催播": "失约被催时认栽滑跪",
        "同期": "能嘴熟络同期是亲近",
        "套话": "被套话点名排名他人时端水",
    }
    return mapping.get(scene, "")


async def _prepare_behaviors(cleaned_paths, stage: Path):
    import extract_persona

    await _convert_to_chat(cleaned_paths, stage)
    converted_path = stage / "converted_chat.json"
    raw = stage / "behaviors_raw.json"
    _clear_output(raw)
    await extract_persona.run([converted_path], out_file=str(raw))
    if not raw.exists():
        raise RuntimeError("行为提取失败，没有写出 behaviors_raw.json")
    data = _read_json(raw)
    raw_behaviors = data.get("behaviors", []) if isinstance(data, dict) else []
    candidates = []
    for item in raw_behaviors:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or _scene_to_name(item.get("scene", ""))).strip()
        trigger = (item.get("trigger") or "").strip()
        response = (item.get("response") or "").strip()
        samples = []
        for sample in item.get("samples", []):
            if not isinstance(sample, dict):
                continue
            user = (sample.get("user") or "").strip()
            reply = (sample.get("reply") or "").strip()
            if user and reply and {"user": user, "reply": reply} not in samples:
                samples.append({"user": user, "reply": reply})
        if not name or not trigger or not response or not samples:
            continue
        evidence = [f"素材样本：{s['user']} → {s['reply']}" for s in samples[:2]]
        candidates.append({
            "name": name,
            "trigger": trigger,
            "response": response,
            "samples": samples,
            "evidence": evidence,
            "scene": item.get("scene", ""),
        })
    _write_json(stage / "behaviors_candidates.json",
                {"source_files": cleaned_paths, "behaviors": candidates})
    return candidates


def _review_markdown(target: str, candidates: list, stage: Path) -> str:
    lines = [f"# persona pipeline review: {target}", "",
             f"- stage: `{stage}`",
             f"- candidates: {len(candidates)}", "", "---", ""]
    if target == "corpus":
        for item in candidates[:30]:
            lines += [f"## {item['statement']}", ""]
    elif target == "phrases":
        for item in candidates[:30]:
            lines += [f"## {item['meaning']}", "",
                      f"- id: {item['id']}",
                      f"- trigger: {item['trigger']}",
                      f"- phrases: {'、'.join(item['phrases'])}",
                      f"- usage: {item['usage']}", ""]
    else:
        for item in candidates[:30]:
            lines += [f"## {item['name']}", "",
                      f"- trigger: {item['trigger']}",
                      f"- response: {item['response']}",
                      f"- samples: {len(item['samples'])}", ""]
    lines += [
        "---", "",
        f"确认后执行：`python scripts/run_tool.py persona-pipeline {target} "
        f"--session {stage.name} --apply`",
        "",
    ]
    return "\n".join(lines)


def _merge_corpus(existing, candidates):
    items = _items(existing, "statements")
    seen = {_norm(x.get("statement") if isinstance(x, dict) else x) for x in items}
    added = []
    for item in candidates:
        text = (item.get("statement") if isinstance(item, dict) else str(item)).strip()
        if text and _norm(text) not in seen:
            seen.add(_norm(text))
            items.append({"statement": text})
            added.append({"statement": text})
    return items, added


def _merge_behaviors(existing, candidates):
    items = _items(existing, "behaviors")
    by_name = {_norm(x.get("name")): x for x in items}
    added = []
    for item in candidates:
        name = (item.get("name") or "").strip()
        target = by_name.get(_norm(name))
        if target:
            old = {(_norm(s.get("user")), _norm(s.get("reply"))) for s in target.get("samples", [])}
            new_samples = []
            for sample in item.get("samples", []):
                key = (_norm(sample.get("user")), _norm(sample.get("reply")))
                if key not in old:
                    old.add(key)
                    new_samples.append(sample)
            target.setdefault("samples", []).extend(new_samples)
            target.setdefault("evidence", [])
            for evidence in item.get("evidence", []):
                if evidence not in target["evidence"]:
                    target["evidence"].append(evidence)
            if not target.get("trigger"):
                target["trigger"] = item.get("trigger", "")
            if not target.get("response"):
                target["response"] = item.get("response", "")
            if new_samples:
                added.append({"name": name, "samples": new_samples})
            continue
        row = {
            "name": name,
            "trigger": item.get("trigger", ""),
            "response": item.get("response", ""),
            "samples": list(item.get("samples", [])),
            "evidence": list(item.get("evidence", [])),
        }
        items.append(row)
        by_name[_norm(name)] = row
        added.append(row)
    return items, added


async def _refresh_corpus_vectors():
    import generate_vectors

    vector_path = _common.VECTOR_FILE
    before_mtime = vector_path.stat().st_mtime if vector_path.exists() else None
    await generate_vectors.run(
        input_path=str(STATEMENT_FILE),
        output_file=str(vector_path),
    )
    after_mtime = vector_path.stat().st_mtime if vector_path.exists() else None
    if after_mtime is None or after_mtime == before_mtime:
        raise RuntimeError("corpus 已写回，但向量缓存没有更新；请检查 embedding 配置")


async def _sanitize_existing_corpus(args, stage: Path):
    data = _read_json(STATEMENT_FILE)
    items = _items(data, "statements")
    candidates = [
        {"statement": (item.get("statement") if isinstance(item, dict) else str(item)).strip()}
        for item in items
    ]
    start_index = max(0, int(args.from_index or 0))
    subset = candidates[start_index:]
    sub_stage = stage / "sanitize_existing"
    sub_stage.mkdir(parents=True, exist_ok=True)
    before_hits = sum(1 for item in subset if _has_self_pronoun(item["statement"]))
    subset, rewritten = await _rewrite_corpus_self_pronouns(subset, sub_stage)
    after_hits = sum(1 for item in subset if _has_self_pronoun(item["statement"]))
    candidates[start_index:] = subset
    print(
        f"corpus 自指代词候选（从索引 {start_index}）："
        f"改写前 {before_hits} 条，实际改写 {rewritten} 条，改写后候选 {after_hits} 条"
    )
    _write_json(sub_stage / "corpus_sanitized_candidates.json",
                {"statements": candidates})

    if not args.apply:
        print("   ℹ️ 当前是 prepare 模式，statement_final 未修改；确认后加 --apply")
        return

    backup = _backup([STATEMENT_FILE], sub_stage)
    if isinstance(data, dict):
        data["statements"] = candidates
        output = data
    else:
        output = candidates
    _write_json(STATEMENT_FILE, output)
    print(f"✅ 已写回 statement_final.json；备份：{backup}")
    if not args.no_vectorize:
        await _refresh_corpus_vectors()


async def _apply_target(target: str, candidates, session: str, stage: Path,
                        no_vectorize: bool, build_hooks: bool):
    paths = {
        "corpus": STATEMENT_FILE,
        "behaviors": _common.BEHAVIORS_FILE,
    }
    keys = {
        "corpus": "statements",
        "behaviors": "behaviors",
    }
    path = paths[target]
    before = _read_json(path)
    backup = _backup([path], stage)

    if target == "corpus":
        merged, added = _merge_corpus(before, candidates)
    else:
        merged, added = _merge_behaviors(before, candidates)

    if isinstance(before, dict):
        before[keys[target]] = merged
        output = before
    else:
        output = merged
    _write_json(path, output)
    print(f"\n✅ 已追加 {len(added)} 条 → {path}")
    print(f"   备份：{backup}")

    if target == "corpus" and build_hooks:
        _build_corpus_hooks()

    if no_vectorize:
        return
    if target == "corpus":
        await _refresh_corpus_vectors()
    # voice-samples 分支已删（2026-10-04：该通道停用，样本已并入 behaviors）


async def run(args):
    if args.session:
        session_name = args.session
    elif args.audio:
        session_name = Path(args.audio).stem
    elif args.input:
        session_name = Path(args.input[0]).stem
    else:
        raise ValueError("必须提供 --audio 或 -i/--input")

    stage = _prepare_stage(session_name)
    candidate_path = stage / f"{args.target}_candidates.json"
    review = stage / f"{args.target}_review.md"

    if args.sanitize_existing:
        if args.target != "corpus":
            raise ValueError("--sanitize-existing 只适用于 corpus")
        await _sanitize_existing_corpus(args, stage)
        return

    if args.apply and not args.refresh and candidate_path.exists():
        print(f"↳ 复用已审批候选：{candidate_path}")
        candidate_data = _read_json(candidate_path)
        key = {
            "corpus": "statements",
            "behaviors": "behaviors",
        }[args.target]
        candidates = _items(candidate_data, key)
        if args.target == "corpus":
            candidates, rewritten = await _rewrite_corpus_self_pronouns(candidates, stage)
            if rewritten:
                candidate_data["statements"] = candidates
                _write_json(candidate_path, candidate_data)
                review.write_text(
                    _review_markdown(args.target, candidates, stage), encoding="utf-8"
                )
                print(f"↳ 应用前又改写自指代词 {rewritten} 条")
        await _apply_target(
            args.target, candidates, session_name, stage,
            args.no_vectorize, not args.no_hooks,
        )
        return

    if args.audio and args.input:
        raise ValueError("--audio 和 -i/--input 不能同时使用")
    cleaned_paths = await _prepare_inputs(args, stage)

    if args.target == "corpus":
        candidates = await _prepare_corpus(cleaned_paths, stage, args.batch_size or 50)
    else:
        candidates = await _prepare_behaviors(cleaned_paths, stage)

    review.write_text(_review_markdown(args.target, candidates, stage), encoding="utf-8")
    print(f"\n✅ 候选生成：{candidate_path}")
    print(f"   人工查看：{review}")

    if args.apply:
        await _apply_target(
            args.target, candidates, session_name, stage,
            args.no_vectorize, not args.no_hooks,
        )
    else:
        print("   ℹ️ 当前是 prepare 模式；确认候选后重新执行并加 --apply")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=TARGETS)
    parser.add_argument("-i", "--input", nargs="+", help="清洗后的 JSON，可传多个")
    parser.add_argument("--audio", help="直接从音频开始；需要 faster-whisper 或 WHISPER_PYTHON")
    parser.add_argument("--session", help="本次素材标签，用于输出目录和新条目 id")
    parser.add_argument("--apply", action="store_true", help="确认后写回 persona")
    parser.add_argument("--refresh", action="store_true",
                        help="配合 --apply：忽略已有候选，重新生成后再写回")
    parser.add_argument("--no-vectorize", action="store_true",
                        help="写回后不自动重建 corpus/voice 向量")
    parser.add_argument("--no-hooks", action="store_true",
                        help="corpus 写回后不自动补 corpus_keywords 钩子")
    parser.add_argument("--sanitize-existing", action="store_true",
                        help="审计并改写现有 statement_final 里的自指代词，不生成新素材")
    parser.add_argument("--from-index", type=int, default=0,
                        help="配合 --sanitize-existing：只处理该索引及之后的条目")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--turn-gap", type=float, default=2.0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--whisper-model", default="medium")
    parser.add_argument("--whisper-python", default=None,
                        help="faster-whisper 所在解释器；设置后强制用该环境转写")
    return parser


def main():
    _common.ensure_utf8_stdout()
    args = build_parser().parse_args()
    try:
        asyncio.run(run(args))
    except Exception as e:
        print(f"\n❌ pipeline 失败：{e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
