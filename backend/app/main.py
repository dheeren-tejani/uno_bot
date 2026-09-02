import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import replay_store, rlbot
from .config import settings
from .schemas import ActionRequest, StartRequest
from .security import SlidingWindowLimiter, api_key_ok, client_ip, json_error
from .sessions import CapacityError, SessionRegistry, start_sweeper

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("uno.api")

# NOTE: run uvicorn with --workers 1 — sessions/replays live in this process.
# (To scale horizontally later, move SessionRegistry/replay_store to Redis.)
sessions = SessionRegistry(
    max_sessions=settings.max_sessions,
    idle_ttl=settings.session_idle_ttl_min * 60,
    finished_ttl=settings.session_finished_ttl_min * 60,
)
start_limiter = SlidingWindowLimiter(settings.rate_start_per_min)
game_limiter = SlidingWindowLimiter(settings.rate_game_per_min)
replay_limiter = SlidingWindowLimiter(settings.rate_replay_per_min)

_GAME_ID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    rlbot.warmup(["easy", "normal", "hard"])   # pre-load checkpoints (fast first game)
    for d in ("easy", "normal", "hard"):
        log.info("bot[%s] → %s", d, rlbot.bot_mode(d))
    log.info("replay codes: %d chars | R2: %s | disk: %s",
             replay_store.CODE_LEN, settings.r2_enabled, settings.data_dir or "off")
    start_sweeper(sessions)
    yield


app = FastAPI(title="UNO 3D Duel API", version="1.0.0", lifespan=lifespan)


# Added BEFORE CORS so CORS ends up outermost and 429/401 responses still
# carry CORS headers (the browser must be able to READ the error).
@app.middleware("http")
async def gatekeeper(request: Request, call_next):
    path = request.url.path
    if not path.startswith("/api/") or request.method == "OPTIONS":
        return await call_next(request)
    ip = client_ip(request, settings.trust_proxy)
    if settings.api_key and path.startswith("/api/game"):
        if not api_key_ok(request.headers.get("x-api-key"), settings.api_key):
            return json_error(401, "Unauthorized.")
    limiter = (start_limiter if path == "/api/game/start"
               else replay_limiter if path.startswith("/api/replay")
               else game_limiter)
    if not limiter.allow(ip):
        return json_error(429, "Too many requests — slow down.")
    if request.method == "POST":
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > 4096:
            return json_error(413, "Payload too large.")
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# Sync endpoints (plain def) → FastAPI runs them in the threadpool, so torch
# inference never blocks the event loop. Each session is guarded by its lock.
@app.post("/api/game/start")
def start_game(body: StartRequest) -> dict:
    try:
        game = sessions.create(body.difficulty)
    except CapacityError:
        raise HTTPException(status_code=503, detail="Server is busy — try again in a moment.")
    with game.lock:
        game.touch()
        if game.engine.status == "playing" and game.engine.current_player == 1:
            game.run_bot()          # bot-led openers resolve before returning
        game.maybe_persist()
        return game.public_state()


@app.post("/api/game/action")
def play_action(body: ActionRequest) -> dict:
    if not _GAME_ID_RE.match(body.game_id):
        raise HTTPException(status_code=400, detail="Malformed game_id.")
    game = sessions.get(body.game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="Game not found — it may have expired.")
    with game.lock:
        game.touch()
        e = game.engine
        if e.status != "playing":
            raise HTTPException(status_code=409, detail="This match is already over.")
        if e.current_player != 0:
            raise HTTPException(status_code=409, detail="It is not your turn.")
        if body.action not in e.legal:
            raise HTTPException(status_code=400, detail="That action is not legal right now.")
        e.apply_action(0, body.action)
        game.run_bot()              # server-side turn chaining (+2/+4/skip sequences)
        game.maybe_persist()
        return game.public_state()


@app.get("/api/replay/{code}")
def get_replay(code: str) -> dict:
    c = code.strip().upper()
    if not replay_store.valid_code(c):
        raise HTTPException(status_code=404, detail="No replay found for that code.")
    payload = replay_store.get(c)
    if payload is None:
        raise HTTPException(status_code=404, detail="No replay found for that code.")
    return payload


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "torch": rlbot.TORCH_OK,
            "training_modules": rlbot.modules_status(),
            "modes": {d: rlbot.bot_mode(d) for d in ("easy", "normal", "hard")},
            "active_sessions": sessions.count(),
            "replay_code_length": replay_store.CODE_LEN}


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse({"detail": "Internal server error."}, status_code=500)


# Optional: serve the built frontend from the API itself — one process,
# same-origin /api, no CORS needed. Set UNO_STATIC_DIR=../frontend/dist.
_static = (Path(settings.static_dir).resolve() if settings.static_dir
           else Path(__file__).resolve().parent.parent.parent / "frontend" / "dist")
if _static.exists():
    app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")