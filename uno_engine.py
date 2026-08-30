"""
uno_engine.py
High-throughput vectorized 2-player UNO game engine with strict action masking,
two-node decision phases (Main Phase & Post-Draw Phase), and full card-counting state tracking.
Supports official-rule Wild Draw Four legality (strict: no matching color cards) via config.
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from config import UnoCardConfig, EnvironmentConfig
from collections import deque


class UnoCardMapper:
    """
    Card-to-Index and Action-to-Index mappings.
    
    Card ID Layout (0..53):
      - 00..12: RED   (0-9, Skip=10, Reverse=11, Draw Two=12)
      - 13..25: YELLOW(0-9, Skip=10, Reverse=11, Draw Two=12)
      - 26..38: GREEN (0-9, Skip=10, Reverse=11, Draw Two=12)
      - 39..51: BLUE  (0-9, Skip=10, Reverse=11, Draw Two=12)
      - 52    : WILD
      - 53    : WILD DRAW FOUR (+4)
      
    Action ID Layout (0..61):
      - 00..51: Play specific colored card
      - 52..55: Play Wild + declare color (52: RED, 53: YELLOW, 54: GREEN, 55: BLUE)
      - 56..59: Play Wild Draw Four + declare color (56: RED, 57: YELLOW, 58: GREEN, 59: BLUE)
      - 60    : DRAW action (only legal in Main Phase)
      - 61    : PASS action (only legal in Post-Draw Phase)
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
        """Returns 0..3 for colored cards, -1 for wild cards."""
        if card_id < 52:
            return card_id // 13
        return -1

    @staticmethod
    def get_card_type(card_id: int) -> int:
        """Returns card value/action (0..12) for colored cards, or card_id for wilds."""
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
        self.obs_dim = 176  # 54 (hand) + 54 (discard counts) + 54 (top card) + 4 (color) + 10 (context)
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
        
        # Draw and Discard Decks per environment using native Python lists (avoids np.append heap churn)
        self.draw_piles = np.zeros((self.num_envs, 108), dtype=np.int16)
        self.draw_pile_pos = np.zeros(self.num_envs, dtype=np.int32)
        
        # Discard pile counts per card type for legal card counting: [num_envs, 54]
        self.discard_counts = np.zeros((self.num_envs, 54), dtype=np.int16)
        self.discard_pile_sizes = np.zeros(self.num_envs, dtype=np.int32)
        
        # Public game state tracking (using int32 to prevent indexing overflow)
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
        for _ in range(7):
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
        histories = self.move_history.copy()
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
        # Cards are stored at positions [0, draw_pile_pos)
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
        # FIX: Normalize all features to [0, 1] range
        event = np.array([
            float(player) / 1.0,                    # Already [0, 1]
            float(action) / 61.0,                   # Normalize from [0, 61] to [0, 1]
            (float(declared_color) + 1.0) / 5.0,    # Normalize from [-1, 3] to [0, 0.8]
            float(drawn) / 4.0                       # Normalize from [0, 4] to [0, 1]
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
        """Returns the move history buffer (no copy — caller must not modify in-place)."""
        return self.move_history

    def step(self, actions: np.ndarray, env_mask: Optional[np.ndarray] = None):
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        winners = -np.ones(self.num_envs, dtype=np.int8)

        if env_mask is None:
            env_mask = np.ones(self.num_envs, dtype=bool)

        # Only process active envs
        active = env_mask
        env_indices = np.where(active)[0]

        for e in env_indices:
            p = int(self.current_player[e])
            opp = 1 - p
            act = int(actions[e])
            phase = int(self.current_phase[e])

            drawn_count = 0
            declared_color = -1
            grant_extra_turn = False
            switch_to_opponent = False

            if phase == self.mapper.PHASE_MAIN:
                if act == self.mapper.ACTION_DRAW:
                    drawn_count, drawn_card = self._draw_cards(e, p, 1)
                    self.turn_counts[e] += 1
                    self.drew_last_turn_flags[e, p, self.active_color[e]] = 1.0

                    if drawn_card is not None and self._is_card_legal(e, drawn_card):
                        self.current_phase[e] = self.mapper.PHASE_POST_DRAW
                        self.last_drawn_card[e] = drawn_card
                        switch_to_opponent = False
                    else:
                        self.current_phase[e] = self.mapper.PHASE_MAIN
                        self.last_drawn_card[e] = -1
                        switch_to_opponent = True
                else:
                    self.turn_counts[e] += 1
                    grant_extra_turn, declared_color = self._execute_card_play(e, p, opp, act)
                    switch_to_opponent = not grant_extra_turn
            else:
                if act == self.mapper.ACTION_PASS:
                    self.current_phase[e] = self.mapper.PHASE_MAIN
                    self.last_drawn_card[e] = -1
                    switch_to_opponent = True
                else:
                    grant_extra_turn, declared_color = self._execute_card_play(e, p, opp, act)
                    self.current_phase[e] = self.mapper.PHASE_MAIN
                    self.last_drawn_card[e] = -1
                    switch_to_opponent = not grant_extra_turn

            # Update history
            dc = declared_color if declared_color != -1 else int(self.active_color[e])
            self._update_history(e, p, act, dc, drawn_count)

            # --- Termination check ---
            player_hand_total = int(np.sum(self.hands[e, p]))

            if player_hand_total == 0:
                dones[e] = True
                winners[e] = p
                rewards[e] = 1.0
            elif self.turn_counts[e] >= self.max_turns:
                dones[e] = True
                opp_total = int(np.sum(self.hands[e, opp]))
                if player_hand_total < opp_total:
                    winners[e] = p
                    rewards[e] = 1.0
                elif player_hand_total > opp_total:
                    winners[e] = opp
                    rewards[e] = -1.0
                else:
                    rewards[e] = 0.0
            else:
                # FIX: Dense reward shaping — penalize having more cards than opponent
                # This gives the agent a continuous signal to play cards and reduce hand size
                opp_total = int(np.sum(self.hands[e, opp]))
                rewards[e] = float(opp_total - player_hand_total) / 54.0 * 0.05
                if switch_to_opponent:
                    self.current_player[e] = opp

        final_turn_counts = self.turn_counts.copy()

        # Reset done envs
        for e in range(self.num_envs):
            if dones[e]:
                self._deal_single_env(e)

        # Vectorized observation/mask construction (see TK2, TK3 below)
        obs = self.get_observations()
        masks = self.get_action_masks()
        histories = self._get_histories()
        infos = {"winners": winners, "turn_counts": final_turn_counts}

        return obs, masks, histories, rewards, dones, infos

    def _execute_card_play(self, e: int, p: int, opp: int, act: int) -> Tuple[bool, int]:
        grant_extra_turn = False
        declared_color = -1

        self.drew_last_turn_flags[e, p, :] = 0.0

        if act < 52:
            card_id = act
            self.hands[e, p, card_id] -= 1
            self.top_card[e] = card_id
            self.discard_counts[e, card_id] += 1
            self.discard_pile_sizes[e] += 1  # NEW: track size
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

        p = self.current_player           # (num_envs,)
        phase = self.current_phase         # (num_envs,)
        env_idx = np.arange(self.num_envs)

        # --- MAIN PHASE (phase == 0) ---
        main = (phase == 0)

        # Get current player's hand: (num_envs, 54)
        hand = self.hands[env_idx, p]

        # Card colors: 0..3 for cards 0..51, -1 for 52..53
        card_colors = np.arange(54, dtype=np.int32) // 13
        card_colors[52:] = -1

        # Card types: 0..12 for cards 0..51, 52 for Wild, 53 for Wild+4
        card_types = np.arange(54, dtype=np.int32) % 13
        card_types[52] = 52
        card_types[53] = 53

        # Active color: (num_envs,) -> (num_envs, 1) for broadcasting
        active_color = self.active_color[:, None]  # (num_envs, 1)

        # Top card type: (num_envs,) -> (num_envs, 1)
        top_type = self.top_card.copy().astype(np.int32)
        # For wild top cards, use the card id as the type (like the original code)
        # card_types already handles this: 52 for Wild, 53 for Wild+4

        # Color match: card_colors[None, :] == active_color -> (num_envs, 54)
        color_match = (card_colors[None, :] == active_color)  # (num_envs, 54)

        # Type match: card_types[None, :] == top_type[:, None] -> (num_envs, 54)
        type_match = (card_types[None, :] == top_type[:, None])  # (num_envs, 54)

        # Has card in hand: (num_envs, 54)
        has_card = hand > 0

        # Legal colored cards: has_card AND (color_match OR type_match) AND main_phase
        legal_colored = has_card & (color_match | type_match)  # (num_envs, 54)
        legal_colored = legal_colored & main[:, None]          # Only in main phase

        masks[:, :52] = legal_colored[:, :52]

        # Wild cards (52:56): legal if hand has Wild AND main phase
        has_wild = hand[:, 52] > 0  # (num_envs,)
        wild_legal = has_wild & main  # (num_envs,)
        masks[:, 52:56] = wild_legal[:, None]

        # Wild Draw Four (56:60): legal if hand has WD4 AND (not strict OR no color match) AND main phase
        has_wd4 = hand[:, 53] > 0  # (num_envs,)

        # Check no cards of active color in hand
        # active_color is (num_envs,), hand is (num_envs, 54)
        # For each env, sum hand[active_color[e]*13 : (active_color[e]+1)*13]
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

        # DRAW action (60): always legal in main phase
        masks[:, 60] = main

        # --- POST-DRAW PHASE (phase == 1) ---
        post = (phase == 1)

        # Drawn card: (num_envs,)
        drawn = self.last_drawn_card  # (num_envs,)

        # For each env, if drawn card is a colored card, that card is playable
        drawn_colored = (drawn >= 0) & (drawn < 52)
        if np.any(drawn_colored & post):
            envs = np.where(drawn_colored & post)[0]
            masks[envs, drawn[envs]] = True

        # Drawn Wild: actions 52:56
        drawn_wild = (drawn == 52)
        if np.any(drawn_wild & post):
            envs = np.where(drawn_wild & post)[0]
            masks[envs, 52:56] = True

        # Drawn Wild+4: actions 56:60
        drawn_wd4 = (drawn == 53)
        if np.any(drawn_wd4 & post):
            envs = np.where(drawn_wd4 & post)[0]
            masks[envs, 56:60] = True

        # PASS action (61): always legal in post-draw phase
        masks[:, 61] = post

        return masks

    def get_observations(self) -> np.ndarray:
        """
        Vectorized observation construction.
        Shape: (num_envs, 176)
        """
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)

        p = self.current_player          # (num_envs,)
        opp = 1 - p                      # (num_envs,)
        env_idx = np.arange(self.num_envs)

        # 1. Own Hand Count Vector [0:54] — vectorized
        obs[:, 0:54] = self.hands[env_idx, p] / 4.0

        # 2. Discard History Count Vector [54:108] — already vectorized
        obs[:, 54:108] = self.discard_counts / 4.0

        # 3. Top Card One-Hot [108:162] — vectorized with advanced indexing
        obs[env_idx, 108 + self.top_card] = 1.0

        # 4. Active Color One-Hot [162:166] — vectorized
        obs[env_idx, 162 + self.active_color] = 1.0

        # 5. Public Context & Flags [166:176] — vectorized
        obs[:, 166] = np.sum(self.hands[env_idx, opp], axis=1) / 20.0
        obs[:, 167] = self.draw_pile_pos / 108.0        # Uses new numpy draw pile
        obs[:, 168] = self.turn_counts / float(self.max_turns)
        obs[:, 169] = self.discard_pile_sizes / 108.0   # Uses new size tracker

        # Opponent drew on color flags [170:174] — vectorized
        obs[:, 170:174] = self.drew_last_turn_flags[env_idx, opp]

        # Phase indicator [174: Main, 175: Post-Draw] — vectorized
        main_mask = (self.current_phase == 0)
        obs[:, 174] = main_mask.astype(np.float32)
        obs[:, 175] = (~main_mask).astype(np.float32)

        return obs