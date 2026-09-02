import random
import secrets
import threading
import time
import uuid

from . import replay_store
from .config import settings
from .engine import UnoEngine
from .rlbot import BotBrain


class CapacityError(Exception):
    pass


class GameSession:
    def __init__(self, difficulty: str):
        self.id = str(uuid.uuid4())
        self.replay_code = replay_store.new_code()      # reserved up-front, 8 chars
        self.brain = BotBrain(difficulty)
        self.lock = threading.Lock()
        self.created_at = time.time()
        self.last_active = time.time()
        self.finished = False
        self.engine = UnoEngine(
            difficulty=difficulty,
            hand_size=settings.initial_hand_size,
            max_turns=settings.max_turns,
            strict_wild4=True,                            # UnoCardConfig.strict_wild_draw_four
            rng=random.Random(secrets.randbits(64)),      # CSPRNG-seeded per game
            value_fn=self.brain.value,
        )

    def touch(self) -> None:
        self.last_active = time.time()

    def run_bot(self) -> None:
        """Turn chaining: resolve every consecutive bot turn until control
        returns to the human or the game ends (guard-bounded)."""
        e = self.engine
        guard = 0
        while e.status == "playing" and e.current_player == 1 and guard < 400:
            action = self.brain.act(e)
            if action not in e.legal:                     # defense-in-depth
                action = e.legal[0] if e.legal else 61
            e.apply_action(1, action)
            guard += 1

    def maybe_persist(self) -> None:
        if self.engine.status == "over" and not self.finished:
            self.finished = True
            replay_store.save(self.replay_payload())

    def replay_payload(self) -> dict:
        e = self.engine
        return {
            "code": self.replay_code,
            "difficulty": e.difficulty,
            "winner": e.winner if e.winner is not None else -1,
            "total_turns": e.turn_count,
            "duration_seconds": round(e.duration),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "frames": e.frames,
        }

    def public_state(self) -> dict:
        """Hidden-information guard: the bot's hand and the deck contents NEVER
        leave this function — only counts do."""
        e = self.engine
        state = {
            "game_id": self.id,
            "replay_code": self.replay_code,
            "status": e.status,
            "phase": e.phase,
            "current_player": e.current_player,
            "turn_count": min(e.turn_count + 1, e.max_turns),
            "top_card": e.top_card,
            "active_color": e.active_color,
            "hand": e.hand_list(0),
            "bot_card_count": e.hand_size(1),
            "deck_count": len(e.deck),
            "legal_actions": list(e.legal),
            "animation_queue": e.take_events(),
        }
        # Only the HUMAN's own post-draw card is exposed (bot post-draw resolves
        # fully server-side before we ever return).
        state["drawn_card"] = (e.last_drawn
                               if e.status == "playing" and e.current_player == 0
                               and e.phase == 1 and e.last_drawn >= 0 else None)
        if e.status == "over":
            state["winner"] = e.winner if e.winner is not None else -1
            state["duration_seconds"] = round(e.duration)
        return state


class SessionRegistry:
    def __init__(self, max_sessions: int, idle_ttl: float, finished_ttl: float):
        self.max_sessions = max_sessions
        self.idle_ttl = idle_ttl
        self.finished_ttl = finished_ttl
        self._games: dict[str, GameSession] = {}
        self._lock = threading.Lock()

    def create(self, difficulty: str) -> GameSession:
        with self._lock:
            if len(self._games) >= self.max_sessions:
                self._sweep_locked()
                if len(self._games) >= self.max_sessions:
                    raise CapacityError()
            game = GameSession(difficulty)
            self._games[game.id] = game
            return game

    def get(self, game_id: str) -> GameSession | None:
        with self._lock:
            return self._games.get(game_id)

    def count(self) -> int:
        with self._lock:
            return len(self._games)

    def sweep(self) -> None:
        with self._lock:
            self._sweep_locked()

    def _sweep_locked(self) -> None:
        now = time.time()
        for gid in list(self._games):
            g = self._games[gid]
            if (g.finished and now - g.created_at > self.finished_ttl) or \
               (now - g.last_active > self.idle_ttl):
                if not g.finished:
                    replay_store.release(g.replay_code)
                self._games.pop(gid, None)


def start_sweeper(registry: SessionRegistry) -> threading.Thread:
    def loop() -> None:
        while True:
            time.sleep(60)
            registry.sweep()
    t = threading.Thread(target=loop, daemon=True, name="session-sweeper")
    t.start()
    return t