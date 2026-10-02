"""OpenAI-compatible LLM backend used by catalogue modules."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_MODEL = os.getenv("CATALOGUE_LLM_MODEL", "deepseek-ai/DeepSeek-V4-Flash")
DEFAULT_BASE_URL = os.getenv("CATALOGUE_LLM_BASE_URL", "http://127.0.0.1:18021/v1")
DEFAULT_TIMEOUT_S = float(os.getenv("CATALOGUE_LLM_TIMEOUT_S", "180"))
DEFAULT_MAX_TOKENS = os.getenv("CATALOGUE_LLM_MAX_TOKENS")
BACKEND_RECOVERY_WAIT_S = float(
    os.getenv("CATALOGUE_LLM_BACKEND_RECOVERY_WAIT_S", "600")
)
BACKEND_RECOVERY_HEARTBEAT_S = float(
    os.getenv("CATALOGUE_LLM_BACKEND_RECOVERY_HEARTBEAT_S", "60")
)
_CLIENT_LOCAL = threading.local()
_RECOVERY_CONDITION = threading.Condition()
_RECOVERY_STATE: Dict[str, Dict[str, Any]] = {}
_AUDIT_LOCK = threading.Lock()


def _write_audit(record: dict) -> None:
    destination = os.getenv("CATALOGUE_LLM_AUDIT_LOG")
    if not destination:
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    # Authentication headers and API keys are never part of the audit schema.
    with _AUDIT_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class BackendRecoveryExhausted(RuntimeError):
    """A connection failed again after the one delayed backend-recovery retry."""


class CompletionTruncatedError(RuntimeError):
    """The backend stopped because the requested output-token limit was reached."""

    def __init__(self, max_tokens: Optional[int], partial_content: str) -> None:
        self.max_tokens = max_tokens
        self.partial_content = partial_content
        limit = "the configured limit" if max_tokens is None else f"{max_tokens} tokens"
        super().__init__(f"LLM completion was truncated at {limit}")


def _is_backend_connection_error(exc: BaseException) -> bool:
    """Return true only for transport-level OpenAI client failures.

    vLLM restarts surface through the OpenAI-compatible client as
    ``APIConnectionError`` (including timeout subclasses).  JSON/schema/model
    errors deliberately stay on the existing short retry path.
    """

    try:
        import openai
    except ImportError:
        return False
    return isinstance(exc, openai.APIConnectionError)


def _api_key() -> str:
    if os.getenv("CATALOGUE_LLM_API_KEY"):
        return os.environ["CATALOGUE_LLM_API_KEY"]
    if os.getenv("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    key_path = Path.home() / ".config" / "openai_key"
    if key_path.exists():
        return key_path.read_text().strip()
    return "local"


def _extra_body() -> Dict[str, Any]:
    raw = os.getenv("CATALOGUE_LLM_EXTRA_BODY")
    if raw:
        return json.loads(raw)
    if not any(key in os.environ for key in (
        "CATALOGUE_LLM_ENABLE_THINKING", "CATALOGUE_LLM_THINKING_BUDGET",
        "CATALOGUE_LLM_ENABLE_THINKING_BUDGET",
    )):
        return {}
    enable_thinking = os.getenv("CATALOGUE_LLM_ENABLE_THINKING", "false").lower()
    thinking_budget = os.getenv("CATALOGUE_LLM_THINKING_BUDGET")
    enable_thinking_budget = os.getenv("CATALOGUE_LLM_ENABLE_THINKING_BUDGET")
    chat_template_kwargs: Dict[str, Any] = {
        "enable_thinking": enable_thinking in {"1", "true", "yes", "on"},
    }
    if thinking_budget:
        chat_template_kwargs["thinking_budget"] = int(thinking_budget)
    if enable_thinking_budget is not None:
        chat_template_kwargs["enable_thinking_budget"] = (
            enable_thinking_budget.lower() in {"1", "true", "yes", "on"}
        )
    return {
        "chat_template_kwargs": chat_template_kwargs
    }


# Backend(s): one CATALOGUE_LLM_BASE_URL, or MANY via
# CATALOGUE_LLM_BASE_URLS="http://127.0.0.1:18020/v1,http://127.0.0.1:18030/v1"
# for load-balancing across vLLM nodes. Each request goes to the backend with the
# fewest in-flight requests (least-connections), so a faster node naturally serves
# more — the right behavior for HETEROGENEOUS nodes (e.g. H100 + A100). Read once
# at import; set the env before launching. Both backends must serve the SAME model.
_BASE_URLS = [u.strip() for u in
              (os.getenv("CATALOGUE_LLM_BASE_URLS") or DEFAULT_BASE_URL or "").split(",")
              if u.strip()] or [DEFAULT_BASE_URL]
_LB_LOCK = threading.Lock()
_INFLIGHT = {u: 0 for u in _BASE_URLS}


def _build_client(base_url: str):
    import openai
    import httpx

    # Per-phase httpx timeout instead of one scalar (see B15): a stuck read raises
    # ReadTimeout instead of wedging a worker forever; connect/pool fail fast on a
    # dead socket. read stays long (DEFAULT_TIMEOUT_S) for big generations.
    read_to = DEFAULT_TIMEOUT_S
    connect_to = float(os.getenv("CATALOGUE_LLM_CONNECT_TIMEOUT_S", "10"))
    pool_to = float(os.getenv("CATALOGUE_LLM_POOL_TIMEOUT_S", "10"))
    timeout = httpx.Timeout(read_to, connect=connect_to, write=read_to, pool=pool_to)
    # Retry only in ``chat_completion``.  The OpenAI SDK otherwise retries each
    # network attempt internally as well, multiplying the configured deadline
    # (for example, 3 evaluator attempts x 3 SDK attempts x 180 seconds).
    kwargs = {"api_key": _api_key(), "timeout": timeout, "max_retries": 0}
    if base_url:
        kwargs["base_url"] = base_url
    return openai.OpenAI(**kwargs)


def _client_for(base_url: str):
    """Thread-local client PER backend URL. Thread-local (not shared) so dropping a
    client on error (a half-open socket — B15) only resets THIS thread's pool, and
    concurrent threads don't contend on one client."""
    clients = getattr(_CLIENT_LOCAL, "clients", None)
    if clients is None:
        clients = {}
        _CLIENT_LOCAL.clients = clients
    c = clients.get(base_url)
    if c is None:
        c = _build_client(base_url)
        clients[base_url] = c
    return c


def client():
    """Back-compat single-client accessor (first configured backend)."""
    return _client_for(_BASE_URLS[0])


def _acquire_url() -> str:
    """Least-connections pick across the configured backends + reserve a slot.
    One backend = trivial no-op returning that URL (zero overhead)."""
    if len(_BASE_URLS) == 1:
        return _BASE_URLS[0]
    with _LB_LOCK:
        u = min(_BASE_URLS, key=lambda x: _INFLIGHT[x])
        _INFLIGHT[u] += 1
        return u


def _release_url(base_url: str) -> None:
    if len(_BASE_URLS) == 1:
        return
    with _LB_LOCK:
        _INFLIGHT[base_url] -= 1


def _backend_health_url(base_url: str) -> str:
    """Return the lightweight OpenAI-compatible model-list endpoint."""

    return base_url.rstrip("/") + "/models"


def _probe_backend_health(base_url: str) -> bool:
    """Send one lightweight heartbeat without consuming a completion retry."""

    import httpx

    try:
        response = httpx.get(
            _backend_health_url(base_url),
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        response.raise_for_status()
        return True
    except (httpx.HTTPError, OSError):
        return False


def _wait_for_backend_recovery(base_url: str) -> bool:
    """Poll for at most the recovery window, coalescing concurrent waiters.

    One thread per process/backend becomes the heartbeat leader. Other workers
    wait on its condition and reuse the result, preventing a restart from
    causing a heartbeat thundering herd. A false return means the full window
    elapsed without a successful health probe; the caller still performs its
    one delayed completion retry so recovery at the boundary is not missed.
    """

    if BACKEND_RECOVERY_WAIT_S <= 0:
        return False
    if BACKEND_RECOVERY_HEARTBEAT_S <= 0:
        raise ValueError(
            "CATALOGUE_LLM_BACKEND_RECOVERY_HEARTBEAT_S must be positive"
        )

    with _RECOVERY_CONDITION:
        state = _RECOVERY_STATE.setdefault(
            base_url,
            {"active": False, "generation": 0, "healthy": False},
        )
        if state["active"]:
            generation = state["generation"]
            while state["active"] and state["generation"] == generation:
                _RECOVERY_CONDITION.wait()
            return bool(state["healthy"])
        state["active"] = True
        state["generation"] += 1

    healthy = False
    elapsed = 0.0
    print(
        "[backend-recovery] connection failure from "
        f"{base_url}; monitoring for up to {BACKEND_RECOVERY_WAIT_S:.0f}s "
        f"with a heartbeat every {BACKEND_RECOVERY_HEARTBEAT_S:.0f}s",
        file=sys.stderr,
        flush=True,
    )
    try:
        while elapsed < BACKEND_RECOVERY_WAIT_S:
            delay = min(
                BACKEND_RECOVERY_HEARTBEAT_S,
                BACKEND_RECOVERY_WAIT_S - elapsed,
            )
            time.sleep(delay)
            elapsed += delay
            healthy = _probe_backend_health(base_url)
            print(
                "[backend-recovery] heartbeat "
                f"at {elapsed:.0f}s: {'healthy' if healthy else 'unavailable'}",
                file=sys.stderr,
                flush=True,
            )
            if healthy:
                break
        return healthy
    finally:
        with _RECOVERY_CONDITION:
            state = _RECOVERY_STATE[base_url]
            state["healthy"] = healthy
            state["active"] = False
            _RECOVERY_CONDITION.notify_all()


def chat_completion(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    response_format: Optional[dict] = None,
    temperature: float = 0.0,
    max_retries: int = 2,
    max_tokens: Optional[int] = None,
) -> str:
    if BACKEND_RECOVERY_WAIT_S < 0:
        raise ValueError("CATALOGUE_LLM_BACKEND_RECOVERY_WAIT_S cannot be negative")
    if BACKEND_RECOVERY_WAIT_S > 0 and BACKEND_RECOVERY_HEARTBEAT_S <= 0:
        raise ValueError(
            "CATALOGUE_LLM_BACKEND_RECOVERY_HEARTBEAT_S must be positive"
        )
    last_err = None
    fast_attempt = 0
    backend_recovery_used = False
    effective_max_tokens = (
        max_tokens
        if max_tokens is not None
        else (int(DEFAULT_MAX_TOKENS) if DEFAULT_MAX_TOKENS else None)
    )
    if effective_max_tokens is not None and effective_max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    while True:
        base_url = _acquire_url()
        request_id = uuid.uuid4().hex
        started = time.monotonic()
        request = {"model": model or DEFAULT_MODEL, "messages": messages,
                   "response_format": response_format, "temperature": temperature,
                   "max_tokens": effective_max_tokens, "extra_body": _extra_body()}
        _write_audit({"event": "request", "request_id": request_id, "request": request})
        try:
            resp = _client_for(base_url).chat.completions.create(
                model=model or DEFAULT_MODEL,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                **({"max_tokens": effective_max_tokens} if effective_max_tokens else {}),
                extra_body=_extra_body(),
            )
            usage = getattr(resp, "usage", None)
            if usage is not None and hasattr(usage, "model_dump"):
                usage = usage.model_dump()
            elif usage is not None:
                usage = {key: getattr(usage, key, None) for key in (
                    "prompt_tokens", "completion_tokens", "total_tokens")}
            message = resp.choices[0].message
            content = message.content
            _write_audit({"event": "response", "request_id": request_id,
                          "elapsed_s": time.monotonic() - started,
                          "response_id": getattr(resp, "id", None),
                          "model": getattr(resp, "model", model or DEFAULT_MODEL),
                          "content": content, "usage": usage,
                          "finish_reason": getattr(resp.choices[0], "finish_reason", None)})
            if content is None:
                reasoning = getattr(message, "reasoning", None)
                raise RuntimeError(
                    "LLM returned empty message.content"
                    + (" with reasoning output. Set CATALOGUE_LLM_ENABLE_THINKING=false."
                       if reasoning else ".")
                )
            if getattr(resp.choices[0], "finish_reason", None) == "length":
                raise CompletionTruncatedError(effective_max_tokens, content)
            return content
        except Exception as e:
            last_err = e
            error = str(e)
            for key in ("CATALOGUE_LLM_API_KEY", "OPENAI_API_KEY"):
                secret = os.getenv(key)
                if secret:
                    error = error.replace(secret, "[redacted]")
            _write_audit({"event": "error", "request_id": request_id,
                          "elapsed_s": time.monotonic() - started,
                          "error_type": type(e).__name__, "error": error})
            # Drop this thread's client for this backend so a half-open socket
            # isn't reused (B15); the next attempt rebuilds and may land on a
            # different backend.
            clients = getattr(_CLIENT_LOCAL, "clients", None)
            if clients is not None:
                clients[base_url] = None
            if _is_backend_connection_error(e):
                if not backend_recovery_used:
                    backend_recovery_used = True
                    _wait_for_backend_recovery(base_url)
                    continue
                raise BackendRecoveryExhausted(
                    "vLLM connection failed after a recovery-monitoring window "
                    f"of up to {BACKEND_RECOVERY_WAIT_S:.0f}s and one retry"
                ) from e
            if fast_attempt < max_retries:
                time.sleep(1 + fast_attempt)
                fast_attempt += 1
                continue
            raise
        finally:
            _release_url(base_url)


def chat_json(
    system: str,
    user: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_retries: int = 2,
    max_tokens: Optional[int] = None,
) -> dict:
    # A successful HTTP response can still contain incomplete JSON when the
    # backend hits its generation cap.  Treat that exactly like a transient
    # completion failure so taxonomy generation does not fail the entire run.
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            content = chat_completion(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                model=model,
                response_format={"type": "json_object"},
                temperature=temperature,
                max_retries=0,
                max_tokens=max_tokens,
            )
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise TypeError(f"Expected a JSON object, got {type(parsed).__name__}")
            return parsed
        except BackendRecoveryExhausted:
            # The request already consumed its one 10-minute recovery retry.
            # Do not multiply that wait through the JSON/schema retry loop.
            raise
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                time.sleep(1 + attempt)
    assert last_err is not None
    raise last_err


class LLMBatchAborted(RuntimeError):
    """Too many LLM calls in a concurrent batch failed or wedged. Signals an
    unstable backend so the caller aborts loudly instead of writing a degenerate
    result (e.g. every chunk empty -> everything becomes a singleton)."""


def map_with_deadline(fn, items, *, n_workers, default, label,
                      stall_s=None, fail_frac=0.25):
    """Run ``fn`` over ``items`` concurrently and return results IN ORDER, made
    robust to a wedged backend — the replacement for ``list(ex.map(fn, items))``
    in the LLM fan-out steps.

    Why this exists: our forwarded endpoint can drop a request into limbo where
    the socket read never returns AND the httpx read timeout never fires (seen
    live: a worker stuck >20 min on a 180s read timeout; the request never even
    reached vLLM). A plain ``ex.map`` then blocks forever on the slowest worker
    and ``with ThreadPoolExecutor`` would re-hang on join. So instead of trusting
    the socket timeout, we watchdog on PROGRESS: if no future completes for
    ``stall_s`` seconds we declare the backend wedged, abandon the still-running
    calls (their threads leak on the dead read — the CLI force-exits via os._exit
    in __main__ so the leak can't hang shutdown) and fill them with ``default``.
    Calls that raise are caught and counted. If failed+abandoned exceeds
    ``fail_frac`` of all items we raise :class:`LLMBatchAborted` so the step
    stops with a clear message instead of emitting garbage. Returns
    ``(results, n_bad)``.
    """
    import concurrent.futures as cf

    if stall_s is None:
        stall_s = float(os.getenv("CATALOGUE_LLM_STALL_S", "120"))
    items = list(items)
    results = [default] * len(items)
    if not items:
        return results, 0

    ex = cf.ThreadPoolExecutor(max_workers=n_workers)
    futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
    remaining = set(futs)
    n_err = 0
    while remaining:
        # FIRST_COMPLETED returns as soon as ANY call finishes, so healthy (even
        # slow) throughput never trips the watchdog; only a full stall_s window
        # with ZERO completions does.
        done, remaining = cf.wait(remaining, timeout=stall_s,
                                  return_when=cf.FIRST_COMPLETED)
        if not done:
            break  # stall: backend wedged -> abandon the rest
        for f in done:
            i = futs[f]
            try:
                results[i] = f.result()
            except Exception as e:  # noqa: BLE001
                n_err += 1
                print(f"    [warn] {label}: call failed: {e}", flush=True)

    n_abandoned = len(remaining)
    ex.shutdown(wait=False)  # never join wedged workers
    n_bad = n_err + n_abandoned
    if n_bad / len(items) > fail_frac:
        raise LLMBatchAborted(
            f"[abort] {label}: {n_bad}/{len(items)} LLM calls failed or wedged "
            f"({n_abandoned} abandoned after {stall_s:.0f}s of zero progress). "
            f"Backend unstable — rerun when the endpoint is healthy.")
    if n_bad:
        print(f"    [warn] {label}: {n_bad}/{len(items)} calls failed/abandoned "
              f"(filled with default)", flush=True)
    return results, n_bad
