"""Developer-only audit and bounded replay helpers for Memory V2.

The default audit is aggregate-only.  Raw source excerpts require both an
explicit user partition and ``--show-source``.  Historical replay reads only
private chat turns and writes derived shadow data under the gitignored
``outputs/memory_v2`` directory.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

try:  # direct script via run_tool.py
    import _common
except ModuleNotFoundError:  # imported as scripts.memory_v2_tool in tests/tools
    from . import _common

# 回放要 import `src.plugins.chatbot.memory_v2`，而 run_tool 只把 scripts/ 加进 sys.path
# ——项目根得自己补（其他离线脚本如 regression_test 也是这么做的）。
# 不补就会 `ModuleNotFoundError: No module named 'src'`（这个命令以前没跑通过）。
import sys as _sys  # noqa: E402

if str(_common.PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_common.PROJECT_ROOT))


DEFAULT_STATE_FILE = _common.PROJECT_ROOT / "user_memory" / "memory_v2_shadow.json"
DEFAULT_REPLAY_DIR = _common.OUTPUTS_DIR / "memory_v2"
MAX_REPLAY_LIMIT = 5000


def redact_user_id(user_id: str) -> str:
    digest = hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()[:10]
    return f"user-{digest}"


def load_state(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 2, "users": {}}
    if not isinstance(data, dict) or not isinstance(data.get("users"), dict):
        return {"schema_version": 2, "users": {}}
    return data


def _redact_source(memory: dict, show_source: bool) -> dict:
    copied = json.loads(json.dumps(memory, ensure_ascii=False))
    source = copied.get("source")
    if isinstance(source, dict) and not show_source:
        source["excerpt"] = "[redacted]"
    copied.pop("session_ids", None)
    return copied


def build_audit_report(
    state: dict,
    *,
    user_id: str | None = None,
    show_source: bool = False,
) -> dict:
    users = state.get("users") if isinstance(state, dict) else {}
    users = users if isinstance(users, dict) else {}
    if user_id is not None:
        partition = users.get(str(user_id)) or {}
        memories = partition.get("memories") if isinstance(partition, dict) else []
        memories = memories if isinstance(memories, list) else []
        return {
            "user": redact_user_id(user_id),
            "updated_at": partition.get("updated_at") if isinstance(partition, dict) else None,
            "memories": [_redact_source(m, show_source) for m in memories if isinstance(m, dict)],
        }

    by_kind: Counter[str] = Counter()
    by_status: Counter[str] = Counter()
    total = 0
    for partition in users.values():
        if not isinstance(partition, dict):
            continue
        for memory in partition.get("memories") or []:
            if not isinstance(memory, dict):
                continue
            total += 1
            by_kind[str(memory.get("kind") or "unknown")] += 1
            by_status[str(memory.get("status") or "unknown")] += 1
    return {
        "schema_version": state.get("schema_version", 2),
        "users": len(users),
        "memories": total,
        "by_kind": dict(sorted(by_kind.items())),
        "by_status": dict(sorted(by_status.items())),
    }


def positive_limit(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("limit 必须是正整数") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("limit 必须大于 0")
    if parsed > MAX_REPLAY_LIMIT:
        raise argparse.ArgumentTypeError(f"limit 不能超过 {MAX_REPLAY_LIMIT}")
    return parsed


def iter_private_turns(paths: Iterable[Path], *, limit: int) -> Iterator[dict]:
    yielded = 0
    for path in paths:
        try:
            handle = Path(path).open("r", encoding="utf-8")
        except OSError:
            continue
        with handle:
            for line in handle:
                if yielded >= limit:
                    return
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict) or record.get("kind") != "private":
                    continue
                session = str(record.get("session") or "").strip()
                user_text = str(record.get("user") or "").strip()
                reply = str(record.get("reply") or "").strip()
                if not session or not user_text or not reply:
                    continue
                yielded += 1
                yield record


def _resolve_inputs(raw_paths: list[str] | None) -> list[Path]:
    if raw_paths:
        return [Path(value) for value in raw_paths]
    log_dir = _common.DATA_DIR / "chat_log"
    # 2026-10-09 起聊天落盘**按会话分文件**（`<QQ号>.jsonl`），
    # 所以不能再 glob `chat*.jsonl`（那只匹配拆分前那个单文件，会静默回放 0 条）。
    return sorted(log_dir.glob("*.jsonl"), key=lambda path: path.stat().st_mtime)


def _safe_replay_dir(path: Path) -> Path:
    resolved = Path(path).resolve()
    output_root = _common.OUTPUTS_DIR.resolve()
    if resolved != output_root and output_root not in resolved.parents:
        raise ValueError("回放产物只能写入 outputs/ 目录")
    return resolved


def run_audit(args) -> None:
    state = load_state(Path(args.state))
    report = build_audit_report(
        state,
        user_id=args.user,
        show_source=bool(args.show_source),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


async def replay(
    *,
    paths: list[Path],
    limit: int,
    out_dir: Path,
) -> dict:
    """Replay historical private turns into an isolated shadow store."""
    from openai import AsyncOpenAI

    # Runtime modules expect a configured NoneBot driver, but this tool does not
    # load the chatbot plugin or start a bot connection.
    import nonebot

    try:
        nonebot.get_driver()
    except ValueError:
        nonebot.init()

    from src.plugins.chatbot.memory_v2 import ShadowMemoryStore, extract_and_ingest

    api_key = _common.get_api_key("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("未检测到 OPENAI_API_KEY")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=_common.get_openai_base_url(),
    )
    out_dir = _safe_replay_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = ShadowMemoryStore(
        state_file=out_dir / "replay_shadow.json",
        audit_file=out_dir / "replay_audit.jsonl",
    )

    turns = 0
    accepted = 0
    try:
        for record in iter_private_turns(paths, limit=limit):
            turns += 1
            try:
                timestamp = float(record.get("t") or 0)
            except (TypeError, ValueError):
                timestamp = 0
            observed_at = datetime.fromtimestamp(timestamp) if timestamp > 0 else datetime.now()
            # A six-hour evidence bucket is only an offline replay scope.  Runtime
            # shadow extraction uses the live topic/session timestamp from core.
            scope = f"{record['session']}:{int(timestamp // (6 * 3600)) if timestamp else turns}"
            items = await extract_and_ingest(
                client=client,
                model=_common.get_model_name(),
                store=store,
                user_id=str(record["session"]),
                user_text=str(record["user"]),
                assistant_text=str(record["reply"]),
                session_id=scope,
                observed_at=observed_at,
            )
            accepted += len(items)
    finally:
        await client.close()

    report = build_audit_report(load_state(out_dir / "replay_shadow.json"))
    report.update({"turns_replayed": turns, "accepted_updates": accepted})
    (out_dir / "replay_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def run_replay(args) -> None:
    paths = _resolve_inputs(args.input)
    report = asyncio.run(replay(
        paths=paths,
        limit=args.limit,
        out_dir=Path(args.out_dir),
    ))
    print(json.dumps(report, ensure_ascii=False, indent=2))


__all__ = [
    "redact_user_id",
    "load_state",
    "build_audit_report",
    "positive_limit",
    "iter_private_turns",
    "run_audit",
    "run_replay",
    "replay",
]
