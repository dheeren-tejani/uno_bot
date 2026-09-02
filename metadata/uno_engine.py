"""
uno_engine.py
High-throughput vectorized 2-player UNO game engine with strict action masking,
two-node decision phases (Main Phase & Post-Draw Phase), and full card-counting state tracking.
Supports official-rule Wild Draw Four legality (strict: no matching color cards) via config.
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from config import UnoCardConfig, EnvironmentConfig
from uno_engine_fast import step_kernel
from collections import deque


class UnoCardMapper:
    """
    Card-to-Index and Action-to-Index mappings.
    """
    def __init__(self, config: Optional[UnoCardConfig] = None):
        self.cfg = config if config is not None else UnoCardConfig()

        self.CARD_NUMBERS = list(range(10))
        self.CARD_SKIP = 10
        self.CARD_REVERSE = 11
        self.CARD_DRAW_TWO = 12
        self.CARD_WILD = 52
        self.CARD_WILD_DRAW_FOUR = 53

        self.ACTION_DRAW = 60
        self.ACTION_PASS = 61

        self.PHASE_MAIN = 0
        self.PHASE_POST_DRAW = 1

    @staticmethod
    def get_card_color(card_id: int) -> int:
        if card_id < 52:
            return card_id // 13
        return -1

    @staticmethod
    def get_card_type(card_id: int) -> int:
        if card_id < 52:
            return card_id % 13
        return card_id


class VectorizedUnoEnv:
    """
    Batched 2-player UNO simulation engine with full discard history tracking and explicit post-draw phases.
    """
    def __init__(self, env_cfg: EnvironmentConfig, card_cfg: UnoCardConfig, seed: int = 42):
        self.num_envs = env_cfg.num_envs
        self.max_turns = env_cfg.max_turns_per_game
        self.obs_dim = 176
        self.num_actions = 62
        self.history_len = env_cfg.history_length
        self.history_dim = env_cfg.history_feature_dim

        self.mapper = UnoCardMapper(card_cfg)
        self.strict_wild_four = bool(getattr(card_cfg, "strict_wild_draw_four", True))
        self.rng = np.random.default_rng(seed)

        # Standard deck template (108 cards)
        deck = []
        for color in range(4):
            offset = color * 13
            deck.append(offset + 0)
            for num in range(1, 13):
                deck.extend([offset + num, offset + num])
        for _ in range(4):
            deck.append(self.mapper.CARD_WILD)
            deck.append(self.mapper.CARD_WILD_DRAW_FOUR)

        self.base_deck = np.array(deck, dtype=np.int16)

        # Player hands: [num_envs, 2, 54]
        self.hands = np.zeros((self.num_envs, 2, 54), dtype=np.int16)

        # Draw and Discard Decks per environment
        self.draw_piles = np.zeros((self.num_envs, 108), dtype=np.int16)
        self.draw_pile_pos = np.zeros(self.num_envs, dtype=np.int32)

        # Discard pile counts per card type: [num_envs, 54]
        self.discard_counts = np.zeros((self.num_envs, 54), dtype=np.int16)
        self.discard_pile_sizes = np.zeros(self.num_envs, dtype=np.int32)

        # Public game state tracking
        self.current_player = np.zeros(self.num_envs, dtype=np.int32)
        self.top_card = np.zeros(self.num_envs, dtype=np.int32)
        self.active_color = np.zeros(self.num_envs, dtype=np.int32)
        self.turn_counts = np.zeros(self.num_envs, dtype=np.int32)

        # Phase tracking: 0 = Main Phase, 1 = Post-Draw Phase
        self.current_phase = np.zeros(self.num_envs, dtype=np.int32)
        self.last_drawn_card = -np.ones(self.num_envs, dtype=np.int32)

        # Inference clue flags: Opponent drew under color [num_envs, 2, 4]
        self.drew_last_turn_flags = np.zeros((self.num_envs, 2, 4), dtype=np.float32)

        # Move History buffers: [num_envs, history_len, history_dim]
        self.move_history = np.zeros((self.num_envs, self.history_len, self.history_dim), dtype=np.float32)

    def _shuffle_deck(self, env_idx: int) -> np.ndarray:
        deck = self.base_deck.copy()
        self.rng.shuffle(deck)
        return deck

    def _deal_single_env(self, e: int):
        self.hands[e].fill(0)
        self.discard_counts[e].fill(0)
        self.current_player[e] = int(self.rng.integers(0, 2))
        self.current_phase[e] = self.mapper.PHASE_MAIN
        self.last_drawn_card[e] = -1
        self.turn_counts[e] = 0
        self.drew_last_turn_flags[e].fill(0.0)
        self.move_history[e].fill(0.0)
        self.discard_pile_sizes[e] = 0

        deck = self._shuffle_deck(e)

        # Deal 7 cards each using numpy slicing (fast)
        for _ in range(10):
            self.hands[e, 0, deck[0]] += 1
            self.hands[e, 1, deck[1]] += 1
            deck = deck[2:]

        # Find starter card (plain number card)
        mask = (deck < 52) & ((deck % 13) <= 9)
        candidates = np.where(mask)[0]
        start_idx = int(candidates[self.rng.integers(len(candidates))]) if len(candidates) > 0 else 0

        first_card = int(deck[start_idx])
        # Remove starter from deck
        deck = np.delete(deck, start_idx)

        self.top_card[e] = first_card
        self.active_color[e] = self.mapper.get_card_color(first_card)

        # Store draw pile as numpy array
        self.draw_piles[e, :len(deck)] = deck
        self.draw_pile_pos[e] = len(deck)

        # Discard pile starts with the first card
        self.discard_counts[e, first_card] = 1
        self.discard_pile_sizes[e] = 1

    def reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        for e in range(self.num_envs):
            self._deal_single_env(e)

        obs = self.get_observations()
        masks = self.get_action_masks()
        histories = self._get_histories()
        return obs, masks, histories

    def _recycle_discard_if_needed(self, env_idx: int, needed_cards: int):
        if self.draw_pile_pos[env_idx] >= needed_cards:
            return

        if self.discard_pile_sizes[env_idx] > 1:
            # Build recyclable pool from discard_counts (everything except top card)
            pool_counts = self.discard_counts[env_idx].copy()
            pool_counts[self.top_card[env_idx]] -= 1

            # Build the list of cards to recycle
            recycled = np.repeat(np.arange(54, dtype=np.int16), pool_counts)
            self.rng.shuffle(recycled)

            # Append to draw pile
            n_recycled = len(recycled)
            start_pos = self.draw_pile_pos[env_idx]
            # Put recycled cards at the BEGINNING (bottom) of draw pile
            # Shift existing cards up, then put recycled at bottom
            existing = self.draw_piles[env_idx, :start_pos].copy()
            self.draw_piles[env_idx, :n_recycled] = recycled
            self.draw_piles[env_idx, n_recycled:n_recycled + start_pos] = existing
            self.draw_pile_pos[env_idx] = start_pos + n_recycled

            # Reset discard counts to only the top card
            self.discard_counts[env_idx].fill(0)
            self.discard_counts[env_idx, self.top_card[env_idx]] = 1
            self.discard_pile_sizes[env_idx] = 1

    def _draw_cards(self, env_idx: int, player_idx: int, count: int):
        """Returns (actual_count_drawn, last_card_drawn)"""
        self._recycle_discard_if_needed(env_idx, count)
        available = self.draw_pile_pos[env_idx]
        actual_draw = min(count, available)

        if actual_draw == 0:
            return 0, None

        # Draw from the TOP of the pile (highest position = top)
        start = self.draw_pile_pos[env_idx] - actual_draw
        end = self.draw_pile_pos[env_idx]
        drawn_cards = self.draw_piles[env_idx, start:end].copy()

        # Add to hand
        for c in drawn_cards:
            self.hands[env_idx, player_idx, c] += 1

        self.draw_pile_pos[env_idx] -= actual_draw
        last_drawn = int(drawn_cards[-1]) if actual_draw > 0 else None

        return actual_draw, last_drawn

    def _update_history(self, env_idx: int, player: int, action: int, declared_color: int, drawn: int):
        event = np.array([
            float(player) / 1.0,
            float(action) / 61.0,
            (float(declared_color) + 1.0) / 5.0,
            float(drawn) / 4.0
        ], dtype=np.float32)

        self.move_history[env_idx] = np.roll(self.move_history[env_idx], shift=-1, axis=0)
        self.move_history[env_idx, -1] = event

    def _is_card_legal(self, env_idx: int, card_id: int) -> bool:
        if card_id >= 52:
            return True
        top_c = self.top_card[env_idx]
        top_type = self.mapper.get_card_type(top_c) if top_c < 52 else top_c
        c_color = self.mapper.get_card_color(card_id)
        c_type = self.mapper.get_card_type(card_id)
        return (c_color == self.active_color[env_idx]) or (c_type == top_type)

    def _get_histories(self) -> np.ndarray:
        """Egocentric move history: player column becomes 0=me, 1=opponent."""
        h = self.move_history.copy()
        flip = self.current_player == 1
        h[flip, :, 0] = 1.0 - h[flip, :, 0]
        return h

    def step(self, actions: np.ndarray, env_mask: Optional[np.ndarray] = None,
             build_outputs: bool = True):
        """
        T1: Uses numba-compiled step_kernel for high-throughput env stepping.
        T5: build_outputs flag allows skipping obs/mask/history construction
            when the caller will build them manually.
        """

        if env_mask is None:
            env_mask = np.ones(self.num_envs, dtype=bool)

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        winners = -np.ones(self.num_envs, dtype=np.int8)
        turn_out = np.empty(self.num_envs, dtype=np.int32)

        step_kernel(
            self.hands, self.draw_piles, self.draw_pile_pos,
            self.discard_counts, self.discard_pile_sizes,
            self.current_player, self.current_phase, self.last_drawn_card,
            self.top_card, self.active_color, self.turn_counts,
            self.drew_last_turn_flags, self.move_history,
            np.ascontiguousarray(actions, dtype=np.int64), env_mask,
            rewards, dones, winners, turn_out, self.max_turns
        )

        for e in np.nonzero(dones)[0]:
            self._deal_single_env(e)

        if build_outputs:
            obs = self.get_observations()
            masks = self.get_action_masks()
            histories = self._get_histories()
        else:
            obs = None
            masks = None
            histories = None

        return obs, masks, histories, rewards, dones, {"winners": winners, "turn_counts": turn_out}

    def _execute_card_play(self, e: int, p: int, opp: int, act: int) -> Tuple[bool, int]:
        grant_extra_turn = False
        declared_color = -1

        self.drew_last_turn_flags[e, p, :] = 0.0

        if act < 52:
            card_id = act
            self.hands[e, p, card_id] -= 1
            self.top_card[e] = card_id
            self.discard_counts[e, card_id] += 1
            self.discard_pile_sizes[e] += 1
            self.active_color[e] = self.mapper.get_card_color(card_id)

            c_type = self.mapper.get_card_type(card_id)
            if c_type in (self.mapper.CARD_SKIP, self.mapper.CARD_REVERSE):
                grant_extra_turn = True
            elif c_type == self.mapper.CARD_DRAW_TWO:
                self._draw_cards(e, opp, 2)
                grant_extra_turn = True

        elif 52 <= act <= 55:
            self.hands[e, p, self.mapper.CARD_WILD] -= 1
            self.top_card[e] = self.mapper.CARD_WILD
            self.discard_counts[e, self.mapper.CARD_WILD] += 1
            self.discard_pile_sizes[e] += 1
            declared_color = act - 52
            self.active_color[e] = declared_color

        elif 56 <= act <= 59:
            self.hands[e, p, self.mapper.CARD_WILD_DRAW_FOUR] -= 1
            self.top_card[e] = self.mapper.CARD_WILD_DRAW_FOUR
            self.discard_counts[e, self.mapper.CARD_WILD_DRAW_FOUR] += 1
            self.discard_pile_sizes[e] += 1
            declared_color = act - 56
            self.active_color[e] = declared_color
            self._draw_cards(e, opp, 4)
            grant_extra_turn = True

        return grant_extra_turn, declared_color

    def _reset_single_env(self, e: int):
        self._deal_single_env(e)

    def get_action_masks(self) -> np.ndarray:
        """
        Vectorized legal action mask computation.
        Shape: (num_envs, 62)
        """
        masks = np.zeros((self.num_envs, self.num_actions), dtype=bool)

        p = self.current_player
        phase = self.current_phase
        env_idx = np.arange(self.num_envs)

        # --- MAIN PHASE (phase == 0) ---
        main = (phase == 0)

        hand = self.hands[env_idx, p]

        card_colors = np.arange(54, dtype=np.int32) // 13
        card_colors[52:] = -1

        card_types = np.arange(54, dtype=np.int32) % 13
        card_types[52] = 52
        card_types[53] = 53

        active_color = self.active_color[:, None]

        # B1 FIX: use % 13 for colored top cards, raw id for wilds
        top_type = np.where(self.top_card < 52, self.top_card % 13, self.top_card).astype(np.int32)

        color_match = (card_colors[None, :] == active_color)
        type_match = (card_types[None, :] == top_type[:, None])
        has_card = hand > 0

        legal_colored = has_card & (color_match | type_match)
        legal_colored = legal_colored & main[:, None]

        masks[:, :52] = legal_colored[:, :52]

        has_wild = hand[:, 52] > 0
        wild_legal = has_wild & main
        masks[:, 52:56] = wild_legal[:, None]

        has_wd4 = hand[:, 53] > 0

        color_sums = np.zeros(self.num_envs, dtype=np.int32)
        for c in range(4):
            color_sums += np.where(
                self.active_color == c,
                np.sum(hand[:, c*13:(c+1)*13], axis=1),
                0
            )
        no_color_match = (color_sums == 0)

        wd4_legal = has_wd4 & ((not self.strict_wild_four) | no_color_match) & main
        masks[:, 56:60] = wd4_legal[:, None]

        masks[:, 60] = main

        # --- POST-DRAW PHASE (phase == 1) ---
        post = (phase == 1)

        drawn = self.last_drawn_card

        drawn_colored = (drawn >= 0) & (drawn < 52)
        if np.any(drawn_colored & post):
            envs = np.where(drawn_colored & post)[0]
            masks[envs, drawn[envs]] = True

        drawn_wild = (drawn == 52)
        if np.any(drawn_wild & post):
            envs = np.where(drawn_wild & post)[0]
            masks[envs, 52:56] = True

        drawn_wd4 = (drawn == 53)
        if np.any(drawn_wd4 & post):
            envs = np.where(drawn_wd4 & post)[0]
            masks[envs, 56:60] = True

        masks[:, 61] = post

        return masks

    def get_observations(self) -> np.ndarray:
        """
        Vectorized observation construction.
        Shape: (num_envs, 176)
        """
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)

        p = self.current_player
        opp = 1 - p
        env_idx = np.arange(self.num_envs)

        # 1. Own Hand Count Vector [0:54]
        obs[:, 0:54] = self.hands[env_idx, p] / 4.0

        # 2. Discard History Count Vector [54:108]
        obs[:, 54:108] = self.discard_counts / 4.0

        # 3. Top Card One-Hot [108:162]
        obs[env_idx, 108 + self.top_card] = 1.0

        # 4. Active Color One-Hot [162:166]
        obs[env_idx, 162 + self.active_color] = 1.0

        # 5. Public Context & Flags [166:176]
        # B7 FIX: normalize by /54.0 (was /20.0, could exceed 1.0 with large hands)
        obs[:, 166] = np.sum(self.hands[env_idx, opp], axis=1) / 54.0
        obs[:, 167] = self.draw_pile_pos / 108.0
        obs[:, 168] = self.turn_counts / float(self.max_turns)
        obs[:, 169] = self.discard_pile_sizes / 108.0

        obs[:, 170:174] = self.drew_last_turn_flags[env_idx, opp]

        main_mask = (self.current_phase == 0)
        obs[:, 174] = main_mask.astype(np.float32)
        obs[:, 175] = (~main_mask).astype(np.float32)

        return obs