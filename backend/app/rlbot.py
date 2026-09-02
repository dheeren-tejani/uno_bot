"""
Bot serving — three difficulty tiers, three checkpoints:
    hard → latest | normal → mid | easy → earliest snapshot.
Same architecture → one class, three weight files (~10MB each, ~30MB RAM total,
loaded once at startup, shared read-only across sessions; per-session GRU
hidden state). Falls back to heuristic archetypes if torch/files are missing.
"""
import importlib.util
import logging
import os
import random
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from .config import settings
from .engine import UnoEngine

log = logging.getLogger("uno.rl")

# Thread caps must be set BEFORE torch import (OMP reads env at import time).
os.environ.setdefault("OMP_NUM_THREADS", str(settings.torch_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(settings.torch_threads))

TORCH_OK = False
_MODULES_ERR: Optional[str] = None
try:
    import torch
    TORCH_OK = True
except Exception as e:  # pragma: no cover
    _MODULES_ERR = f"torch import failed: {e}"

ModelConfig = None
ActorCritic = None
HIDDEN_DIM = 512
_modules_loaded = False

FALLBACK_ARCHETYPE = {"easy": "erratic", "normal": "hoarder", "hard": "aggro"}
_mode_cache: dict[str, str] = {}


def _training_roots() -> list[Path]:
    roots: list[Path] = []
    if settings.training_dir:
        roots.append(Path(settings.training_dir).expanduser().resolve())
    backend_root = Path(__file__).resolve().parent.parent          # backend/
    roots.append(backend_root / "training")                        # backend/training/
    roots.append(backend_root)                                     # backend/ (checkpoints/)
    return [r for r in roots if r.exists()]


def _load_module(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod          # lets models.py do `from config import ...`
    spec.loader.exec_module(mod)
    return mod


def load_training_modules() -> bool:
    """Import config.py + models.py from the training repo without sys.path."""
    global ModelConfig, ActorCritic, HIDDEN_DIM, _modules_loaded, _MODULES_ERR
    if _modules_loaded:
        return ModelConfig is not None
    _modules_loaded = True
    if not TORCH_OK:
        return False
    for root in _training_roots():
        cfg_path, models_path = root / "config.py", root / "models.py"
        if not (cfg_path.exists() and models_path.exists()):
            continue
        try:
            cfg_mod = _load_module(cfg_path, "config")
            sys.modules.setdefault("config", cfg_mod)
            models_mod = _load_module(models_path, "models")
            ModelConfig = cfg_mod.ModelConfig
            ActorCritic = models_mod.MaskedRecurrentActorCritic
            HIDDEN_DIM = ModelConfig().core_gru_dim
            torch.set_num_threads(max(1, settings.torch_threads))
            _MODULES_ERR = None
            return True
        except Exception as e:
            _MODULES_ERR = f"loading training modules from {root} failed: {e}"
            log.error("%s", _MODULES_ERR)
    if _MODULES_ERR is None:
        _MODULES_ERR = "no training dir found (set UNO_TRAINING_DIR or add backend/training/)"
    return False


def modules_status() -> str:
    load_training_modules()
    return _MODULES_ERR or "ok"


def _clean(sd: dict) -> dict:
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def _resolve_model_path(difficulty: str) -> Optional[Path]:
    override = {"easy": settings.model_easy, "normal": settings.model_normal,
                "hard": settings.model_hard}[difficulty]
    cands: list[Path] = []
    if override:
        p = Path(override).expanduser()
        cands.append(p if p.is_absolute() else Path.cwd() / p)
    for r in _training_roots():
        cands.append(r / "checkpoints" / f"{difficulty}.pt")
        cands.append(r / f"{difficulty}.pt")
    if difficulty == "hard":
        for r in _training_roots():
            cands.append(r / "checkpoints" / "latest.pt")
    for c in cands:
        if c.exists():
            return c
    return None


_model_cache: dict[str, tuple[object, str]] = {}
_model_lock = threading.Lock()


def _describe(path: Path, ckpt) -> str:
    if isinstance(ckpt, dict):
        parts = []
        if ckpt.get("iteration") is not None:
            parts.append(f"iter {ckpt['iteration']}")
        if ckpt.get("elo") is not None:
            parts.append(f"elo {int(ckpt['elo'])}")
        if parts:
            return f"{path.name} ({', '.join(parts)})"
    return path.name


def load_model_for(difficulty: str):
    """Returns (model | None, description). One shared instance per file."""
    if not load_training_modules():
        return None, "training modules unavailable"
    path = _resolve_model_path(difficulty)
    if path is None:
        return None, "no checkpoint"
    key = str(path)
    with _model_lock:
        if key in _model_cache:
            return _model_cache[key]
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model = ActorCritic()
        model.load_state_dict(_clean(sd))
        model.eval()
        entry = (model, _describe(path, ckpt))
        with _model_lock:
            _model_cache[key] = entry
        log.info("loaded '%s' checkpoint: %s", difficulty, entry[1])
        return entry
    except Exception as e:
        log.error("failed to load %s: %s", path, e)
        return None, f"load error: {e}"


# ── Heuristic archetypes (port of league.HeuristicArchetypes — fallback only) ──
def archetype_action(kind: str, engine: UnoEngine) -> int:
    legal = engine.legal
    if len(legal) == 1:
        return legal[0]
    hand = engine.hands[1]
    color_counts = [sum(hand[c * 13:(c + 1) * 13]) for c in range(4)]
    best_color = max(range(4), key=lambda c: color_counts[c])
    legal_set = set(legal)

    if engine.phase == 1:
        if kind == "hoarder" and random.random() < 0.30:
            return 61
        playable = [a for a in legal if a != 61]
        return playable[0] if playable else 61

    if kind == "aggro":
        for act in [56 + best_color, 56, 57, 58, 59]:
            if act in legal_set:
                return act
        for c in range(4):
            if c * 13 + 12 in legal_set:
                return c * 13 + 12
        for c in range(4):
            for t in (10, 11):
                if c * 13 + t in legal_set:
                    return c * 13 + t
        for act in legal:
            if act < 52 and act // 13 == best_color:
                return act
        return legal[0]
    if kind == "hoarder":
        for act in legal:
            if act < 52 and act % 13 <= 9:
                return act
        for c in range(4):
            for t in (10, 11):
                if c * 13 + t in legal_set:
                    return c * 13 + t
        if 60 in legal_set:
            return 60
        return legal[0]
    if kind == "color_manipulator":
        for act in [52 + best_color, 56 + best_color]:
            if act in legal_set:
                return act
        for act in legal:
            if act < 52 and act // 13 == best_color:
                return act
        return legal[0]
    if random.random() < 0.25:                     # erratic
        return random.choice(legal)
    return legal[0]


def heuristic_value(engine: UnoEngine) -> float:
    h0, h1 = engine.hand_size(0), engine.hand_size(1)
    bot, human = engine.hands[1], engine.hands[0]
    ac = engine.active_color
    v = (h0 - h1) * 0.09
    v += sum(bot[ac * 13:(ac + 1) * 13]) * 0.03
    v -= sum(human[ac * 13:(ac + 1) * 13]) * 0.025
    if h0 == 1:
        v -= 0.3
    if h1 == 1:
        v += 0.3
    return max(-1.0, min(1.0, v))


class BotBrain:
    def __init__(self, difficulty: str):
        self.difficulty = difficulty
        self.fallback = FALLBACK_ARCHETYPE[difficulty]
        self.model, self.model_src = load_model_for(difficulty)
        self.greedy = settings.bot_greedy
        self.hidden = None
        self.last_value = 0.0

    @property
    def kind(self) -> str:
        return f"model:{self.model_src}" if self.model is not None else f"heuristic:{self.fallback}"

    def value(self, engine: UnoEngine) -> float:
        return self.last_value if self.model is not None else heuristic_value(engine)

    def act(self, engine: UnoEngine) -> int:
        if self.model is not None:
            try:
                return self._model_act(engine)
            except Exception:
                log.exception("model inference failed — heuristic fallback this turn")
        return archetype_action(self.fallback, engine)

    def _model_act(self, engine: UnoEngine) -> int:
        legal = engine.legal
        if not legal:
            return 61
        obs = torch.from_numpy(engine.observation(1)).unsqueeze(0)
        mask = torch.zeros((1, 62), dtype=torch.bool)
        for a in legal:
            mask[0, a] = True
        hist = torch.from_numpy(engine.history(1)).unsqueeze(0)
        if self.hidden is None:
            self.hidden = torch.zeros((1, 1, HIDDEN_DIM))
        with torch.no_grad():
            dist, value, self.hidden = self.model(
                obs=obs, mask=mask, history=hist, hidden_state=self.hidden)
            self.last_value = float(value.item())
            action = int(dist.masked_logits.argmax(dim=-1).item() if self.greedy
                         else dist.sample().item())
        return action if action in legal else archetype_action(self.fallback, engine)


def warmup(difficulties: list[str]) -> None:
    load_training_modules()
    for d in difficulties:
        _mode_cache[d] = BotBrain(d).kind


def bot_mode(difficulty: str) -> str:
    return _mode_cache.get(difficulty, "uninitialized")