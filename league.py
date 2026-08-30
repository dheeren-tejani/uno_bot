"""
league.py
Prioritized Fictitious Self-Play (PFSP) League System with Multi-Archetype Population Sparring.
Includes dynamic Elo updates and non-uniform opponent matchmaking to prevent policy cycling.
"""

from typing import List, Dict, Optional, Tuple
import os
import numpy as np
import torch

from config import LeagueConfig, ModelConfig, HardwareConfig
from models import MaskedRecurrentActorCritic


class LeagueOpponent:
    """Represents a frozen historical snapshot of the agent in the league."""
    def __init__(self, model_path: str, elo: float, iteration: int):
        self.model_path = model_path
        self.elo = elo
        self.iteration = iteration
        self.games_played = 0
        self.wins = 0  # Opponent wins against main agent

    @property
    def win_rate_against_main(self) -> float:
        return self.wins / max(1, self.games_played)


class HeuristicArchetypes:
    """Rule-based sparring population exposing diverse human-like playstyles."""
    ARCHETYPES = ["aggro", "hoarder", "color_manipulator", "erratic"]

    @staticmethod
    def get_action(archetype: str, env, env_idx: int, player_idx: int, mask: np.ndarray) -> int:
        legal = np.where(mask)[0]
        if len(legal) == 1:
            return int(legal[0])

        hand = env.hands[env_idx, player_idx]
        color_counts = [sum(hand[c * 13:(c + 1) * 13]) for c in range(4)]
        best_color = int(np.argmax(color_counts))
        phase = env.current_phase[env_idx]

        # Post-draw phase handling
        if phase == 1:
            if archetype == "hoarder" and np.random.rand() < 0.30:
                return 61  # Pass and keep drawn card
            # Play drawn card if available
            playable = [a for a in legal if a != 61]
            return int(playable[0]) if len(playable) > 0 else 61

        # Main Phase: Archetype-specific decision trees
        if archetype == "aggro":
            # Priority: +4 -> +2 -> Skip/Reverse -> Match dominant color -> Any legal
            for act in [56 + best_color, 56, 57, 58, 59]:
                if act in legal: return act
            for c in range(4):
                act = c * 13 + 12
                if act in legal: return act
            for c in range(4):
                for t in [10, 11]:
                    act = c * 13 + t
                    if act in legal: return act
            for act in legal:
                if act < 52 and (act // 13) == best_color: return act
            return int(legal[0])

        elif archetype == "hoarder":
            # Priority: Numbers (0-9) -> Skips/Reverses -> Wilds/Draws only when forced -> Draw (60)
            for act in legal:
                if act < 52 and (act % 13) <= 9: return act
            for c in range(4):
                for t in [10, 11]:
                    act = c * 13 + t
                    if act in legal: return act
            if 60 in legal:
                return 60
            return int(legal[0])

        elif archetype == "color_manipulator":
            # Forces active color into dominant hand color relentlessly
            for act in [52 + best_color, 56 + best_color]:
                if act in legal: return act
            for act in legal:
                if act < 52 and (act // 13) == best_color: return act
            return int(legal[0])

        elif archetype == "erratic":
            # Simulates casual, noisy human play
            if np.random.rand() < 0.25:
                return int(np.random.choice(legal))
            return int(legal[0])

        return int(legal[0])


class LeagueManager:
    """
    Orchestrates Prioritized Fictitious Self-Play (PFSP) matchmaking.
    Weights historical snapshots and heuristic archetypes by their win rates against the active policy.
    """
    def __init__(self, league_cfg: LeagueConfig, model_cfg: ModelConfig, hw_cfg: HardwareConfig):
        self.cfg = league_cfg
        self.model_cfg = model_cfg
        self.hw = hw_cfg

        self.main_agent_elo = self.cfg.initial_elo
        self.best_bot_elo = self.cfg.initial_elo
        self.best_bot_path: Optional[str] = None

        self.pool: List[LeagueOpponent] = []
        self._cached_opponent_net: Optional[MaskedRecurrentActorCritic] = None
        self._cached_opponent_path: Optional[str] = None

        # Archetype win tracking
        self.archetype_stats: Dict[str, Dict[str, int]] = {
            name: {"games": 0, "wins": 0} for name in HeuristicArchetypes.ARCHETYPES
        }

        self.current_cycle_games = 0
        self.current_cycle_wins = 0
        self.last_snapshot_games = 0

    def calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update_elo(self, main_win: float, opponent_idx: Optional[int] = None, arch_name: Optional[str] = None):
        """
        Updates Elo ratings strictly on asymmetric matches.
        Ignores pure self-play clones to prevent inflation.
        """
        # 1. Skip self-play updates entirely
        if opponent_idx is None and arch_name is None:
            return

        # 2. Determine Opponent Rating
        if opponent_idx is not None and opponent_idx < len(self.pool):
            opp = self.pool[opponent_idx]
            opp_elo = opp.elo
        else:
            # Baseline archetypes are anchored around 1200-1400 baseline
            opp_elo = 1300.0

        # 3. Elo update
        k_factor = self.cfg.elo_k_factor
        expected_main = self.calculate_expected_score(self.main_agent_elo, opp_elo)
        expected_opp = 1.0 - expected_main

        self.main_agent_elo += k_factor * (main_win - expected_main)

        if opponent_idx is not None and opponent_idx < len(self.pool):
            opp.elo += k_factor * ((1.0 - main_win) - expected_opp)
            opp.games_played += 1
            if main_win == 0.0:
                opp.wins += 1

    def _clean_state_dict(self, state_dict: dict) -> dict:
        """Strips '_orig_mod.' prefix added by torch.compile."""
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    def sample_matchup(self) -> Tuple[str, Optional[str]]:
        """
        Samples matchmaking channel:
        Returns: (category, specific_archetype_name_if_any)
        """
        r = np.random.rand()
        if r < self.cfg.self_play_prob or len(self.pool) == 0:
            return "self", None
        elif r < (self.cfg.self_play_prob + self.cfg.archetype_prob):
            # Prioritize archetypes that beat the main agent most frequently
            arch_names = HeuristicArchetypes.ARCHETYPES
            loss_rates = []
            for name in arch_names:
                g = self.archetype_stats[name]["games"]
                w = self.archetype_stats[name]["wins"]
                loss_rates.append((w / max(1, g)) + 0.05)  # Epsilon base weight

            probs = np.array(loss_rates) ** self.cfg.pfsp_alpha
            probs /= probs.sum()
            chosen_arch = str(np.random.choice(arch_names, p=probs))
            return "archetype", chosen_arch
        else:
            return "historical_pfsp", None

    def get_opponent_model(
        self,
        current_model: MaskedRecurrentActorCritic,
        category: str
    ) -> Tuple[Optional[MaskedRecurrentActorCritic], Optional[int]]:
        """Selects neural opponent using PFSP weights if category == 'historical_pfsp'."""
        if category in ("self", "archetype") or len(self.pool) == 0:
            return current_model, None

        # Prioritized Fictitious Self-Play (PFSP) sampling
        opp_win_rates = [opp.win_rate_against_main + 0.05 for opp in self.pool]
        probs = np.array(opp_win_rates) ** self.cfg.pfsp_alpha
        probs /= probs.sum()
        target_idx = int(np.random.choice(len(self.pool), p=probs))
        target_path = self.pool[target_idx].model_path

        if self._cached_opponent_net is None:
            self._cached_opponent_net = MaskedRecurrentActorCritic(self.model_cfg).to(self.hw.device)
            self._cached_opponent_net.eval()

        if self._cached_opponent_path != target_path:
            checkpoint = torch.load(target_path, map_location=self.hw.device, weights_only=True)
            raw_state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            clean_state = self._clean_state_dict(raw_state)
            self._cached_opponent_net.load_state_dict(clean_state)
            self._cached_opponent_path = target_path

        return self._cached_opponent_net, target_idx

    def record_match_outcome(self, winner: int, opp_pool_idx: Optional[int] = None, arch_name: Optional[str] = None):
        self.current_cycle_games += 1
        if winner == 0:
            self.current_cycle_wins += 1
            self.update_elo(1.0, opp_pool_idx)
        elif winner == 1:
            self.update_elo(0.0, opp_pool_idx)
            if arch_name is not None and arch_name in self.archetype_stats:
                self.archetype_stats[arch_name]["wins"] += 1
        else:
            self.update_elo(0.5, opp_pool_idx)

        if arch_name is not None and arch_name in self.archetype_stats:
            self.archetype_stats[arch_name]["games"] += 1

    def should_save_snapshot(self, total_games: int) -> bool:
        # Interval trigger uses a tracked counter (exact-multiple modulo checks
        # silently skip triggers when game completions arrive in uneven jumps).
        if total_games - self.last_snapshot_games >= self.cfg.snapshot_interval_games:
            return True
        if self.current_cycle_games >= 1500:
            win_rate = self.current_cycle_wins / self.current_cycle_games
            if win_rate >= self.cfg.win_rate_snapshot_threshold and self.main_agent_elo > self.best_bot_elo + 20.0:
                return True
        return False

    def restore_pool_from_disk(self, league_dir: str) -> int:
        """
        Rebuilds the matchmaking pool from previously saved league checkpoints.
        Used on --resume so Ctrl+C/restart does not empty the PFSP opposition.
        Files are never deleted here; if more than max_league_capacity exist,
        only the strongest are kept in the roster (all stay on disk).
        Returns the number of opponents restored.
        """
        if not os.path.isdir(league_dir):
            return 0

        entries: List[LeagueOpponent] = []
        for fname in sorted(os.listdir(league_dir)):
            if not fname.endswith(".pt"):
                continue
            path = os.path.join(league_dir, fname)
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=True)
                elo = float(ckpt.get("elo", self.cfg.initial_elo))
                iteration = int(ckpt.get("iteration", -1))
            except Exception:
                continue
            entries.append(LeagueOpponent(model_path=path, elo=elo, iteration=iteration))

        entries.sort(key=lambda x: x.elo, reverse=True)
        kept = entries[: self.cfg.max_league_capacity]
        self.pool.extend(kept)

        if kept:
            top = max(self.pool, key=lambda x: x.elo)
            if top.elo > self.best_bot_elo:
                self.best_bot_elo = top.elo
                self.best_bot_path = top.model_path

        return len(kept)

    def add_snapshot(self, model: MaskedRecurrentActorCritic, iteration: int, save_dir: str, total_games: int = 0) -> str:
        filename = f"league_model_iter_{iteration}_elo_{int(self.main_agent_elo)}.pt"
        save_path = os.path.join(save_dir, filename)

        raw_model = getattr(model, "_orig_mod", model)
        clean_state = self._clean_state_dict(raw_model.state_dict())

        torch.save({
            "iteration": iteration,
            "elo": self.main_agent_elo,
            "model_state_dict": clean_state
        }, save_path)

        new_opp = LeagueOpponent(model_path=save_path, elo=self.main_agent_elo, iteration=iteration)
        self.pool.append(new_opp)

        if self.main_agent_elo >= self.best_bot_elo:
            self.best_bot_elo = self.main_agent_elo
            self.best_bot_path = save_path

        if len(self.pool) > self.cfg.max_league_capacity:
            self.pool.sort(key=lambda x: x.elo)
            removed = self.pool.pop(0)  # Evict the weakest snapshot
            if os.path.exists(removed.model_path) and removed.model_path != self.best_bot_path:
                try:
                    os.remove(removed.model_path)
                except OSError:
                    pass
        
        self.last_snapshot_games = total_games
        self.current_cycle_games = 0
        self.current_cycle_wins = 0
        return save_path

    def get_stats(self) -> Dict[str, float]:
        return {
            "league/main_elo": self.main_agent_elo,
            "league/best_elo": self.best_bot_elo,
            "league/pool_size": float(len(self.pool)),
            "league/cycle_win_rate": (self.current_cycle_wins / max(1, self.current_cycle_games))
        }