# Memory V2 Shadow Pipeline

## Task 1: Specify the typed memory contract

**Acceptance criteria:**
- [x] Tests cover facts, four interaction-preference dimensions, emotional expiry, and assistant-only commitments.
- [x] Tests reject malformed, oversized, sensitive, and cross-role records.
- [x] Tests prove inferred preferences require three observations across two sessions.

**Verification:**
- [x] `.venv\Scripts\python.exe -m pytest tests\test_memory_v2.py -q`

**Dependencies:** None

## Task 2: Implement shadow storage and context rendering

**Acceptance criteria:**
- [x] Shadow writes are atomic and isolated by user ID.
- [x] Audit entries retain bounded provenance without logging to stdout.
- [x] Rendered context respects the 700-character ceiling and persona-preservation rules.

**Verification:**
- [x] `.venv\Scripts\python.exe -m pytest tests\test_memory_v2.py -q`

**Dependencies:** Task 1

## Task 3: Integrate disabled-by-default extraction

**Acceptance criteria:**
- [ ] Only private chats are eligible.
- [ ] The absent feature flag produces zero Memory V2 calls/writes.
- [ ] Extraction failure never affects the user reply.

**Verification:**
- [ ] `.venv\Scripts\python.exe -m pytest tests\test_memory_v2.py tests\test_core.py -q`

**Dependencies:** Task 2

## Task 4: Fix legacy commitment extraction ambiguity

**Acceptance criteria:**
- [ ] The prompt never treats a user's plan as an assistant promise.
- [ ] A regression test guards the distinction.

**Verification:**
- [ ] `.venv\Scripts\python.exe -m pytest tests\test_memory.py -q`

**Dependencies:** None

## Task 5: Add developer audit and replay commands

**Acceptance criteria:**
- [ ] Audit defaults to aggregate counts and redacted user identifiers.
- [ ] Detailed inspection requires an explicit user ID.
- [ ] Replay is private-chat-only, bounded by a required positive limit, and writes only to ignored output paths.

**Verification:**
- [ ] `.venv\Scripts\python.exe -m pytest tests\test_memory_v2_tool.py -q`

**Dependencies:** Tasks 2-3

## Task 6: Final verification and documentation

**Acceptance criteria:**
- [ ] Architecture and operation docs describe shadow mode and promotion gates.
- [ ] Full tests and lint pass.
- [ ] Diff contains no real user data or secrets.

**Verification:**
- [ ] `.venv\Scripts\python.exe -m pytest -q`
- [ ] `.venv\Scripts\ruff.exe check memory_manager.py src\plugins\chatbot\memory_v2.py src\plugins\chatbot\core.py scripts\memory_v2_tool.py scripts\run_tool.py tests\test_memory_v2.py tests\test_memory_v2_tool.py`

**Dependencies:** Tasks 1-5
