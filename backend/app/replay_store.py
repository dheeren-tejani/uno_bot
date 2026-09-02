"""Replay persistence: 8-char codes (secure RNG, collision-proof under
concurrency via in-flight reservation), memory + optional local disk with
atomic writes + optional Cloudflare R2 mirror."""
import json
import logging
import re
import secrets
import threading
from pathlib import Path

from .config import settings

log = logging.getLogger("uno.replay")

ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # 32 unambiguous chars
CODE_LEN = 8                                    # 32^8 ≈ 1.1e12 codes
_CODE_RE = re.compile(r"^[A-Z0-9]{8}$")

_mem: dict[str, dict] = {}
_in_flight: set[str] = set()
_lock = threading.Lock()


def valid_code(code: str) -> bool:
    return bool(_CODE_RE.match(code))


def new_code() -> str:
    with _lock:
        for _ in range(200):
            code = "".join(secrets.choice(ALPHABET) for _ in range(CODE_LEN))
            if code not in _mem and code not in _in_flight:
                _in_flight.add(code)     # reserve until the game is saved
                return code
    return secrets.token_hex(4).upper()  # astronomically-unlikely fallback


def _disk_path(code: str) -> Path:
    return Path(settings.data_dir) / "replays" / f"{code}.json"


def _r2_client():
    import boto3  # lazy optional dependency
    return boto3.client(
        "s3",
        endpoint_url=f"https://{settings.r2_account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
    )


def save(payload: dict) -> None:
    code = payload["code"]
    with _lock:
        _mem[code] = payload
        _in_flight.discard(code)
    persisted = False
    if settings.data_dir:
        try:
            p = _disk_path(code)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(p)              # atomic — never a torn replay file
            persisted = True
        except Exception:
            log.exception("disk replay save failed")
    if settings.r2_enabled:
        try:
            _r2_client().put_object(
                Bucket=settings.r2_bucket, Key=f"replays/{code}.json",
                Body=json.dumps(payload), ContentType="application/json")
            persisted = True
        except Exception:
            log.exception("R2 replay upload failed")
    if persisted:
        with _lock:
            _mem.pop(code, None)


def get(code: str) -> dict | None:
    with _lock:
        if code in _mem:
            return _mem[code]
    if settings.data_dir:
        p = _disk_path(code)
        if p.exists():
            try:
                payload = json.loads(p.read_text())
                with _lock:
                    _mem[code] = payload
                return payload
            except Exception:
                log.exception("disk replay read failed")
    if settings.r2_enabled:
        try:
            obj = _r2_client().get_object(Bucket=settings.r2_bucket, Key=f"replays/{code}.json")
            payload = json.loads(obj["Body"].read())
            with _lock:
                _mem[code] = payload
            return payload
        except Exception:
            return None
    return None

def release(code: str) -> None:
    """Free a reserved code that will never be saved (e.g. abandoned session)."""
    with _lock:
        _in_flight.discard(code)