import hmac
import threading
import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse


class SlidingWindowLimiter:
    """Thread-safe per-key sliding-window rate limiter."""

    def __init__(self, limit: int, window_seconds: float = 60.0):
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            dq = self._hits[key]
            while dq and now - dq[0] > self.window:
                dq.popleft()
            if len(dq) >= self.limit:
                return False
            dq.append(now)
            if len(self._hits) > 20_000:  # bound memory under spoofed-IP floods
                for k in [k for k, v in self._hits.items() if not v]:
                    self._hits.pop(k, None)
            return True


def client_ip(request: Request, trust_proxy: bool) -> str:
    # Only honor X-Forwarded-For when WE control the proxy in front,
    # otherwise clients could rotate fake IPs to dodge per-IP limits.
    if trust_proxy:
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def api_key_ok(provided: str | None, expected: str) -> bool:
    if not expected:
        return True
    return hmac.compare_digest(provided or "", expected)


def json_error(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status)