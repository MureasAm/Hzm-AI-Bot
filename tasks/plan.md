# Implementation Plan: Memory V2 Shadow Pipeline

## Overview

Build a private-chat-only, disabled-by-default shadow memory pipeline. It extracts four typed memory classes (facts, interaction preferences, temporary emotional state, and commitments), keeps provenance and time semantics, consolidates evidence conservatively, renders a bounded hypothetical context, and exposes developer-only audit/replay tooling. The legacy memory path remains the source of live replies until the shadow data is reviewed.

## Architecture Decisions

- Add one deep `memory_v2` module with a small interface for extraction validation, ingestion, context selection, and user deletion.
- Store shadow state and append-only audit data under ignored `user_memory/`; never write real chat content into tracked files or ordinary logs.
- Treat all LLM output as untrusted: strict allowlists, length limits, type validation, and fail-closed parsing.
- Confirm explicit interaction preferences immediately; inferred preferences need three matching observations across at least two conversation sessions.
- Keep emotional state episode-scoped with a six-hour hard expiry; it never promotes itself into a profile.
- Keep the feature disabled by default and separate `shadow` from `apply`; this plan implements shadow only.
- Keep persona behavior authoritative: rendered adaptation changes support/banter/directness/initiative, not identity, facts, or safety boundaries.

## Task List

### Phase 1: Foundation

- [x] Task 1: Add failing tests for the typed memory contract, privacy validation, evidence consolidation, expiry, and bounded rendering.
- [x] Task 2: Implement the pure Memory V2 model and file-backed shadow store.

### Checkpoint: Foundation

- [x] Focused Memory V2 tests pass.
- [x] No existing memory files are read or modified by tests.

### Phase 2: Runtime Shadow Slice

- [x] Task 3: Add the structured extraction prompt and asynchronous private-chat shadow hook behind `MEMORY_V2_SHADOW=1`.
- [x] Task 4: Correct the legacy promise example so user plans cannot be recorded as assistant commitments.

### Checkpoint: Runtime

- [x] Existing legacy replies and memory injection are unchanged when the flag is absent.
- [x] Invalid or failed extraction silently degrades without affecting replies.

### Phase 3: Developer Audit and Replay

- [x] Task 5: Add developer-only `memory-audit` and bounded historical `memory-replay` tool commands.
- [x] Task 6: Document enablement, output locations, privacy constraints, and promotion criteria.

### Checkpoint: Complete

- [x] Focused memory/session/core tests pass.
- [x] Full pytest suite passes.
- [x] Ruff passes on touched Python files.
- [x] No secrets or real `user_memory/` / `data/chat_log/` contents appear in the diff.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| New extractor increases API cost | Medium | Disabled by default; replay is explicitly bounded by `--limit` |
| Model invents or misclassifies memory | High | Strict schema, conservative confirmation, source provenance, shadow-only rollout |
| Emotional adaptation erases persona | High | Fixed behavioral dimensions and explicit persona-preservation renderer |
| Private chat data leaks into repository/output | High | Runtime files remain under ignored paths; audit defaults to aggregates and redacted identifiers |
| Existing memory behavior regresses | High | No live injection in V2; legacy path remains intact and covered by regression tests |

## Open Questions

None. The user approved the design tree on 2026-10-06.
