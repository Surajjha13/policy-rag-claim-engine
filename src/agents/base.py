"""Shared timing helper: every agent action gets one TraceEvent.

Centralizing this in one place guarantees the trace format returned to the
frontend/API is consistent across all five agents, and that no agent can
accidentally leak something other than name/action/elapsed_ms/detail.
"""

import time

from src.schemas.decision import TraceEvent


def timed(agent_name: str, action: str, trace: list[TraceEvent], fn, *args, detail: dict | None = None, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    trace.append(
        TraceEvent(agent=agent_name, action=action, elapsed_ms=round(elapsed_ms, 1), detail=detail or {})
    )
    return result
