"""Memory V2 shadow store.

This module deliberately does not affect live replies.  It validates typed LLM
extractions, consolidates evidence, persists a per-user shadow profile, and can
render the bounded context that *would* be injected after the shadow rollout is
approved.

Runtime data stays under ``user_memory/`` (gitignored).  Model output is treated
as untrusted input and must pass the allowlists below before it is stored.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .constants import PROJECT_ROOT


DEFAULT_STATE_FILE = PROJECT_ROOT / "user_memory" / "memory_v2_shadow.json"
DEFAULT_AUDIT_FILE = PROJECT_ROOT / "user_memory" / "memory_v2_audit.jsonl"

SCHEMA_VERSION = 2
EMOTION_TTL = timedelta(hours=6)
MAX_SOURCE_CHARS = 240
MAX_VALUE_CHARS = 160
MAX_USER_ID_CHARS = 128
MAX_SESSION_ID_CHARS = 96

_KINDS = {"fact", "interaction_preference", "emotional_state", "commitment"}
_ACTIONS = {"upsert", "retract"}
_PREFERENCE_VALUES = {
    "support_style": {"comfort", "problem_solving", "balanced"},
    "banter_tolerance": {"low", "light", "high"},
    "directness": {"gentle", "direct", "balanced"},
    "care_initiative": {"proactive", "reserved", "balanced"},
}
_SENSITIVE_KEYS = {
    "phone", "phone_number", "email", "exact_address", "home_address",
    "id_number", "identity_number", "bank_account", "finance_account",
    "medical_diagnosis", "health_diagnosis", "password", "credential",
}
_SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
)
_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")

_PREFERENCE_BEHAVIORS = {
    ("support_style", "comfort"): "用户低落时先接住情绪，不要马上讲道理或给方案。",
    ("support_style", "problem_solving"): "用户求助时可以更快给出具体办法，但先简短确认其感受。",
    ("support_style", "balanced"): "用户低落时先回应感受，再视情况给一个简短办法。",
    ("banter_tolerance", "low"): "减少针对用户的互怼和挤兑，保留灰泽满自己的嘴硬语气即可。",
    ("banter_tolerance", "light"): "普通闲聊可以轻度调侃，碰到认真或低落内容时收住。",
    ("banter_tolerance", "high"): "普通闲聊可以更强地互怼；用户明显低落时暂停。",
    ("directness", "gentle"): "涉及拒绝或纠正时说得委婉一点，但不要虚假附和。",
    ("directness", "direct"): "可以直接说结论，不必层层铺垫。",
    ("directness", "balanced"): "结论说清楚，同时保留一点缓冲。",
    ("care_initiative", "proactive"): "察觉用户状态变化时可以主动关心一句，但不要连续追问。",
    ("care_initiative", "reserved"): "用户没有主动展开时不要追问隐私或情绪原因。",
    ("care_initiative", "balanced"): "可以关心一次；用户不展开就自然换回当前话题。",
}


def _iso(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _bounded_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text[:limit]


def _contains_sensitive_value(value: str) -> bool:
    return any(pattern.search(value) for pattern in _SENSITIVE_VALUE_PATTERNS)


def _ngrams(text: str) -> set[str]:
    normalized = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text or "").lower()
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[i:i + 2] for i in range(len(normalized) - 1)}


def _relevance(query: str, record: dict) -> float:
    query_terms = _ngrams(query)
    memory_terms = _ngrams(f"{record.get('key', '')}{record.get('value', '')}")
    if not query_terms or not memory_terms:
        return 0.0
    return len(query_terms & memory_terms) / max(1, min(len(query_terms), len(memory_terms)))


class ShadowMemoryStore:
    """Validate, consolidate, and inspect Memory V2 shadow state.

    The interface is intentionally small: callers ingest model candidates,
    request a hypothetical context, inspect a user partition, or delete it.
    """

    def __init__(
        self,
        state_file: Path = DEFAULT_STATE_FILE,
        audit_file: Path = DEFAULT_AUDIT_FILE,
        context_budget_chars: int = 700,
    ) -> None:
        self.state_file = Path(state_file)
        self.audit_file = Path(audit_file)
        self.context_budget_chars = max(100, int(context_budget_chars))
        self._lock = threading.Lock()

    def _empty_state(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, "users": {}}

    def _load(self) -> dict:
        if not self.state_file.exists():
            return self._empty_state()
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._empty_state()
        if not isinstance(data, dict) or not isinstance(data.get("users"), dict):
            return self._empty_state()
        return data

    def _save(self, state: dict) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_file.with_suffix(self.state_file.suffix + ".tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.state_file)

    def _audit(self, entry: dict) -> None:
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.audit_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _normalize_source(self, source: dict, role: str, now: datetime) -> dict | None:
        if not isinstance(source, dict):
            return None
        session_id = _bounded_text(source.get("session_id"), MAX_SESSION_ID_CHARS)
        if not session_id:
            return None
        observed = _iso(source.get("observed_at")) or now
        raw = source.get("assistant_text") if role == "assistant" else source.get("user_text")
        excerpt = _bounded_text(raw, MAX_SOURCE_CHARS)
        if not excerpt:
            return None
        return {
            "role": role,
            "excerpt": excerpt,
            "session_id": session_id,
            "observed_at": observed.isoformat(),
        }

    def _normalize_item(self, item: dict, source: dict, now: datetime) -> tuple[dict | None, str]:
        if not isinstance(item, dict):
            return None, "item_not_object"
        kind = _bounded_text(item.get("kind"), 40)
        key = _bounded_text(item.get("key"), 48)
        value = _bounded_text(item.get("value"), MAX_VALUE_CHARS)
        action = _bounded_text(item.get("action") or "upsert", 16)
        explicit = item.get("explicit")
        confidence = item.get("confidence")

        if kind not in _KINDS or action not in _ACTIONS:
            return None, "unsupported_kind_or_action"
        if not _KEY_RE.fullmatch(key) or not value:
            return None, "invalid_key_or_value"
        if not isinstance(explicit, bool):
            return None, "explicit_must_be_boolean"
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            return None, "confidence_must_be_number"
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            return None, "confidence_out_of_range"
        if key in _SENSITIVE_KEYS or _contains_sensitive_value(value):
            return None, "sensitive_data_blocked"

        role = "assistant" if kind == "commitment" else "user"
        normalized_source = self._normalize_source(source, role, now)
        if normalized_source is None:
            return None, "invalid_source"

        if kind == "fact" and not explicit:
            return None, "inferred_fact_blocked"
        if kind == "interaction_preference":
            if key not in _PREFERENCE_VALUES or value not in _PREFERENCE_VALUES[key]:
                return None, "invalid_preference_dimension"
        if kind == "emotional_state" and key != "current_emotion":
            return None, "invalid_emotional_state_key"
        if kind == "commitment":
            if item.get("actor") != "assistant" or not explicit:
                return None, "assistant_commitment_required"

        normalized = {
            "kind": kind,
            "key": key,
            "value": value,
            "action": action,
            "explicit": explicit,
            "confidence": confidence,
            "source": normalized_source,
        }
        return normalized, "accepted"

    @staticmethod
    def _memory_id(kind: str, key: str, value: str) -> str:
        raw = f"{kind}\0{key}\0{value}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:20]

    def ingest(self, user_id: str, items: list, *, source: dict, now: datetime | None = None) -> list[dict]:
        now = now or datetime.now()
        user_id = _bounded_text(user_id, MAX_USER_ID_CHARS)
        if not user_id or not isinstance(items, list):
            return []

        accepted: list[dict] = []
        with self._lock:
            state = self._load()
            user = state["users"].setdefault(user_id, {"memories": [], "updated_at": now.isoformat()})
            memories = user.setdefault("memories", [])

            for raw_item in items[:20]:
                item, reason = self._normalize_item(raw_item, source, now)
                if item is None:
                    self._audit({
                        "t": now.isoformat(), "user_id": user_id,
                        "decision": "rejected", "reason": reason,
                    })
                    continue

                if item["action"] == "retract":
                    changed = False
                    for memory in memories:
                        if (memory.get("kind") == item["kind"] and memory.get("key") == item["key"]
                                and memory.get("status") not in {"superseded", "expired"}
                                and (not item["value"] or memory.get("value") == item["value"])):
                            memory["status"] = "superseded"
                            memory["invalidated_at"] = now.isoformat()
                            changed = True
                    self._audit({
                        "t": now.isoformat(), "user_id": user_id,
                        "decision": "retracted" if changed else "ignored",
                        "kind": item["kind"], "key": item["key"],
                    })
                    continue

                memory_id = self._memory_id(item["kind"], item["key"], item["value"])
                existing = next((m for m in memories if m.get("id") == memory_id), None)
                if existing is None:
                    if item["explicit"] and item["kind"] in {"fact", "interaction_preference"}:
                        status = "confirmed"
                    elif item["kind"] in {"emotional_state", "commitment"}:
                        status = "active"
                    else:
                        status = "candidate"
                    existing = {
                        "id": memory_id,
                        "kind": item["kind"],
                        "key": item["key"],
                        "value": item["value"],
                        "status": status,
                        "explicit": item["explicit"],
                        "confidence": item["confidence"],
                        "evidence_count": 1,
                        "session_ids": [item["source"]["session_id"]],
                        "session_count": 1,
                        "created_at": now.isoformat(),
                        "last_observed_at": now.isoformat(),
                        "source": item["source"],
                    }
                    if item["kind"] == "emotional_state":
                        existing["expires_at"] = (now + EMOTION_TTL).isoformat()
                    memories.append(existing)
                else:
                    existing["evidence_count"] = int(existing.get("evidence_count", 1)) + 1
                    session_ids = list(existing.get("session_ids") or [])
                    if item["source"]["session_id"] not in session_ids:
                        session_ids.append(item["source"]["session_id"])
                    existing["session_ids"] = session_ids[-20:]
                    existing["session_count"] = len(existing["session_ids"])
                    existing["last_observed_at"] = now.isoformat()
                    existing["source"] = item["source"]
                    existing["confidence"] = max(float(existing.get("confidence", 0)), item["confidence"])
                    if item["explicit"]:
                        existing["explicit"] = True
                        existing["status"] = "active" if item["kind"] in {"emotional_state", "commitment"} else "confirmed"
                    elif (item["kind"] == "interaction_preference"
                          and existing["evidence_count"] >= 3
                          and existing["session_count"] >= 2):
                        existing["status"] = "confirmed"
                    if item["kind"] == "emotional_state":
                        existing["status"] = "active"
                        existing["expires_at"] = (now + EMOTION_TTL).isoformat()

                # A new explicit scalar value supersedes older values for the same key.
                if item["explicit"] and item["kind"] in {"fact", "interaction_preference", "emotional_state"}:
                    for other in memories:
                        if (other is not existing and other.get("kind") == item["kind"]
                                and other.get("key") == item["key"]
                                and other.get("status") not in {"superseded", "expired"}):
                            other["status"] = "superseded"
                            other["invalidated_at"] = now.isoformat()

                accepted.append(json.loads(json.dumps(existing, ensure_ascii=False)))
                self._audit({
                    "t": now.isoformat(), "user_id": user_id, "decision": "accepted",
                    "memory_id": existing["id"], "kind": existing["kind"],
                    "key": existing["key"], "status": existing["status"],
                })

            user["updated_at"] = now.isoformat()
            self._save(state)
        return accepted

    def snapshot(self, user_id: str) -> dict:
        with self._lock:
            user = self._load().get("users", {}).get(str(user_id), {})
        if not isinstance(user, dict):
            return {"memories": []}
        result = json.loads(json.dumps(user, ensure_ascii=False))
        result.setdefault("memories", [])
        return result

    def build_context(self, user_id: str, query: str, *, now: datetime | None = None) -> str:
        now = now or datetime.now()
        memories = self.snapshot(user_id).get("memories", [])
        active = []
        for memory in memories:
            status = memory.get("status")
            if status not in {"confirmed", "active"}:
                continue
            if memory.get("kind") == "emotional_state":
                expires = _iso(memory.get("expires_at"))
                if expires is None or expires <= now:
                    continue
            active.append(memory)

        lines: list[str] = []
        preferences = [m for m in active if m.get("kind") == "interaction_preference"]
        emotions = [m for m in active if m.get("kind") == "emotional_state"]
        if preferences or emotions:
            lines.append("【与这个用户的相处方式】只调整相处方式，不改变灰泽满的核心性格、事实和边界。")
            for memory in preferences[:4]:
                behavior = _PREFERENCE_BEHAVIORS.get((memory.get("key"), memory.get("value")))
                if behavior:
                    lines.append(behavior)
            if emotions:
                latest = max(emotions, key=lambda m: m.get("last_observed_at", ""))
                lines.append(f"用户此刻{latest.get('value')}；本轮优先照顾当前情绪，必要时暂停互怼。")

        core_keys = {"preferred_name", "name", "city"}
        facts = [m for m in active if m.get("kind") == "fact"]
        facts.sort(key=lambda m: (m.get("key") not in core_keys, -_relevance(query, m)))
        selected_facts = [m for m in facts if m.get("key") in core_keys or _relevance(query, m) > 0][:4]
        if selected_facts:
            rendered = "；".join(f"{m.get('key')}：{m.get('value')}" for m in selected_facts)
            lines.append(f"【相关用户事实】{rendered}")

        commitments = [m for m in active if m.get("kind") == "commitment" and _relevance(query, m) > 0]
        if commitments:
            rendered = "；".join(str(m.get("value")) for m in commitments[:2])
            lines.append(f"【相关未完成约定】{rendered}")

        if not lines:
            return ""
        output: list[str] = []
        used = 0
        for line in lines:
            separator = 1 if output else 0
            remaining = self.context_budget_chars - used - separator
            if remaining <= 0:
                break
            piece = line[:remaining]
            if piece:
                output.append(piece)
                used += len(piece) + separator
            if len(piece) < len(line):
                break
        return "\n".join(output)

    def delete_user(self, user_id: str, *, now: datetime | None = None) -> None:
        now = now or datetime.now()
        user_id = _bounded_text(user_id, MAX_USER_ID_CHARS)
        if not user_id:
            return
        with self._lock:
            state = self._load()
            existed = state.get("users", {}).pop(user_id, None) is not None
            self._save(state)
            self._audit({
                "t": now.isoformat(), "user_id": user_id,
                "decision": "deleted" if existed else "delete_noop",
            })


__all__ = ["ShadowMemoryStore", "DEFAULT_STATE_FILE", "DEFAULT_AUDIT_FILE"]
