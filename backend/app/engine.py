"""
Authoritative server-side UNO engine — a faithful single-game mirror of the
training environment (uno_engine.VectorizedUnoEnv + uno_engine_fast.step_kernel):

  • Official 108-card deck: 1x zero, 2x each 1-9/Skip/Reverse/+2 per color,
    4x Wild, 4x Wild +4 (duplicates exist — the frontend renders instances).
  • 10-card opening hands: the training engine's `_deal_single_env` loops
    `for _ in range(10)` (its comment says 7, but the code deals 10 and the
    policy was trained on that distribution). Override: UNO_INITIAL_HAND_SIZE.
  • Strict Wild+4 in the Main Phase (zero cards of active color) — including
    the training quirk that a *drawn* Wild+4 in Post-Draw is always playable.
  • turn_count advances ONLY on Main-Phase actions (draw / play), never on
    Post-Draw actions — exactly like the kernel.
  • 200-turn termination: fewer cards wins, exact tie = draw (-1).
  • Deck recycling: discards minus the top card are shuffled to the BOTTOM.
  • drew_last_turn flags (set on draw under active color, cleared only when
    that player plays a card — PASS keeps them, as in the kernel).
  • 176-dim card-counting observation + 8-step move history, built exactly
    like get_observations() / _get_histories() for bot inference.

The event/animation_queue shape matches the REST contract the frontend consumes.
"""
import random
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

COL_NAMES = ["RED", "YELLOW", "GREEN", "BLUE"]
CARD_WILD, CARD_WILD_DF = 52, 53
ACTION_DRAW, ACTION_PASS = 60, 61
PHASE_MAIN, PHASE_POST_DRAW = 0, 1
OBS_DIM, NUM_ACTIONS, HISTORY_LEN = 176, 62, 8


def card_name(cid: int) -> str:
    if cid == CARD_WILD:
        return "Wild"
    if cid == CARD_WILD_DF:
        return "Wild +4"
    c = ["Red", "Yellow", "Green", "Blue"][cid // 13]
    t = cid % 13
    return f"{c} {t if t <= 9 else ['Skip', 'Reverse', '+2'][t - 10]}"


def build_deck() -> List[int]:
    deck: List[int] = []
    for color in range(4):
        off = color * 13
        deck.append(off + 0)
        for t in range(1, 13):
            deck.extend([off + t, off + t])
    deck.extend([CARD_WILD] * 4)
    deck.extend([CARD_WILD_DF] * 4)
    return deck  # 108 cards


class UnoEngine:
    def __init__(self, difficulty: str, hand_size: int, max_turns: int,
                 strict_wild4: bool, rng: random.Random,
                 value_fn: Callable[["UnoEngine"], float]):
        self.difficulty = difficulty
        self.max_turns = max_turns
        self.strict_wild4 = strict_wild4
        self.rng = rng
        self._value_fn = value_fn

        # ── Deal (mirrors _deal_single_env) ──────────────────────────
        deck = build_deck()
        rng.shuffle(deck)
        self.hands: List[List[int]] = [[0] * 54, [0] * 54]
        for i in range(hand_size):
            self.hands[0][deck[2 * i]] += 1
            self.hands[1][deck[2 * i + 1]] += 1
        rest = deck[2 * hand_size:]
        candidates = [i for i, c in enumerate(rest) if c < 52 and c % 13 <= 9]
        start_idx = rng.choice(candidates) if candidates else 0
        starter = rest.pop(start_idx)

        self.deck: List[int] = rest            # draw from the END (top)
        self.discard: List[int] = [starter]
        self.top_card = starter
        self.active_color = starter // 13
        self.current_player = rng.randrange(2)  # randomized starter
        self.phase = PHASE_MAIN
        self.last_drawn = -1
        self.turn_count = 0
        self.drew_flags = [[False] * 4, [False] * 4]
        self.move_history: List[Tuple[int, int, int, int]] = []

        self.status = "playing"
        self.winner: Optional[int] = None
        self.duration = 0.0
        self.started_at = time.time()
        self.events: List[dict] = []
        self.frames: List[dict] = []
        self.legal: List[int] = self.legal_actions()

        self._emit({
            "type": "deal",
            "human_cards": self.hand_list(0),
            "bot_count": self.hand_size(1),
            "starter_card": starter,
            "text": "Cards dealt — you lead." if self.current_player == 0
                    else "Cards dealt — R3X leads.",
        })
        self._record("Match start — cards dealt.")

    # ── Public helpers ─────────────────────────────────────────────
    def hand_size(self, p: int) -> int:
        return sum(self.hands[p])

    def hand_list(self, p: int) -> List[int]:
        out: List[int] = []
        for cid in range(54):
            out.extend([cid] * self.hands[p][cid])
        return out

    def take_events(self) -> List[dict]:
        out, self.events = self.events, []
        return out

    # ── Legality (mirror of get_action_masks, ascending ids) ───────
    def _matches(self, card: int) -> bool:
        if card // 13 == self.active_color:
            return True
        tt = self.top_card % 13 if self.top_card < 52 else self.top_card
        return card % 13 == tt

    def legal_actions(self) -> List[int]:
        if self.status != "playing":
            return []
        hand = self.hands[self.current_player]
        acts: List[int] = []
        if self.phase == PHASE_MAIN:
            for cid in range(52):
                if hand[cid] > 0 and (cid // 13 == self.active_color
                                      or self._matches(cid)):
                    acts.append(cid)
            if hand[CARD_WILD] > 0:
                acts += [52, 53, 54, 55]
            has_active_color = sum(hand[self.active_color * 13:(self.active_color + 1) * 13]) > 0
            if hand[CARD_WILD_DF] > 0 and (not self.strict_wild4 or not has_active_color):
                acts += [56, 57, 58, 59]
            acts.append(ACTION_DRAW)
        else:
            d = self.last_drawn
            if 0 <= d < 52:
                acts.append(d)
            elif d == CARD_WILD:
                acts += [52, 53, 54, 55]
            elif d == CARD_WILD_DF:
                acts += [56, 57, 58, 59]     # post-draw quirk: strict check bypassed
            acts.append(ACTION_PASS)
        return acts

    # ── Observation / history for the network (byte-for-byte) ──────
    def observation(self, p: int) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        opp = 1 - p
        obs[0:54] = np.asarray(self.hands[p], dtype=np.float32) / 4.0
        disc = np.bincount(np.asarray(self.discard, dtype=np.int64), minlength=54)
        obs[54:108] = disc.astype(np.float32) / 4.0
        obs[108 + self.top_card] = 1.0
        obs[162 + self.active_color] = 1.0
        obs[166] = self.hand_size(opp) / 54.0
        obs[167] = len(self.deck) / 108.0
        obs[168] = self.turn_count / float(self.max_turns)
        obs[169] = len(self.discard) / 108.0
        obs[170:174] = self.drew_flags[opp]
        obs[174] = 1.0 if self.phase == PHASE_MAIN else 0.0
        obs[175] = 1.0 - obs[174]
        return obs

    def history(self, p: int) -> np.ndarray:
        # Egocentric last-8 moves: [mover(0=me), act/61, (color+1)/5, drawn/4]
        h = np.zeros((HISTORY_LEN, 4), dtype=np.float32)
        n = len(self.move_history)
        for i, (mover, act, dc, drawn) in enumerate(self.move_history):
            h[i + (HISTORY_LEN - n)] = [
                0.0 if mover == p else 1.0, act / 61.0,
                (dc + 1.0) / 5.0, drawn / 4.0,
            ]
        return h

    # ── Internals ────────────────────────────────────────────────────
    def _emit(self, ev: dict) -> None:
        self.events.append(ev)

    def _emit_draw(self, p: int, drawn: List[int]) -> None:
        if not drawn:
            return
        text = (f"You draw {len(drawn)} cards." if len(drawn) != 1 else "You draw a card.") \
            if p == 0 else f"R3X draws {len(drawn)} card{'s' if len(drawn) != 1 else ''}."
        ev = {"type": "draw", "actor": p, "count": len(drawn), "text": text}
        if p == 0:
            ev["card_ids"] = list(drawn)     # the human's own cards are public to them
        self._emit(ev)

    def _draw(self, p: int, count: int) -> List[int]:
        # Recycle discards (minus top) to the BOTTOM of the pile, like the kernel.
        if len(self.deck) < count and len(self.discard) > 1:
            pool = self.discard[:-1]
            self.rng.shuffle(pool)
            self.deck = pool + self.deck
            self.discard = [self.discard[-1]]
            self._emit({"type": "reshuffle", "deck_count": len(self.deck),
                        "text": "Deck exhausted — discards reshuffled."})
        actual = min(count, len(self.deck))
        drawn = [self.deck.pop() for _ in range(actual)]
        for c in drawn:
            self.hands[p][c] += 1
        return drawn

    @staticmethod
    def _decode(action: int) -> Tuple[int, int]:
        if action <= 51:
            return action, -1
        if action <= 55:
            return CARD_WILD, action - 52
        return CARD_WILD_DF, action - 56

    def _exec_play(self, p: int, opp: int, action: int) -> Tuple[bool, int]:
        grant_extra, declared = False, -1
        self.drew_flags[p] = [False, False, False, False]
        if action < 52:
            cid = action
            self.hands[p][cid] -= 1
            self.discard.append(cid)
            self.top_card = cid
            self.active_color = cid // 13
            t = cid % 13
            if t in (10, 11):                      # Skip / Reverse → extra turn in 2P
                grant_extra = True
            elif t == 12:                          # +2
                self._emit_draw(opp, self._draw(opp, 2))
                grant_extra = True
        elif action <= 55:                         # Wild
            self.hands[p][CARD_WILD] -= 1
            self.discard.append(CARD_WILD)
            self.top_card = CARD_WILD
            declared = action - 52
            self.active_color = declared
        else:                                      # Wild +4
            self.hands[p][CARD_WILD_DF] -= 1
            self.discard.append(CARD_WILD_DF)
            self.top_card = CARD_WILD_DF
            declared = action - 56
            self.active_color = declared
            self._emit_draw(opp, self._draw(opp, 4))
            grant_extra = True
        return grant_extra, declared

    def _finish(self, winner: int) -> None:
        self.status = "over"
        self.winner = winner
        self.duration = time.time() - self.started_at

    def _record(self, event: str) -> None:
        try:
            v = round(float(self._value_fn(self)), 3)
        except Exception:
            v = 0.0
        self.frames.append({
            "turn": self.turn_count,
            "p0_hand": self.hand_list(0), "p1_hand": self.hand_list(1),
            "top_card": self.top_card, "active_color": self.active_color,
            "bot_value": v, "deck_count": len(self.deck), "event": event,
            "deck": list(self.deck), "discard": list(self.discard),
        })

    # ── Main resolution (mirror of step_kernel) ─────────────────────
    def apply_action(self, p: int, action: int) -> None:
        opp = 1 - p
        me = "You" if p == 0 else "R3X"
        switch = False
        drawn_count = 0
        declared = -1
        grant_extra = False

        if self.phase == PHASE_MAIN:
            if action == ACTION_DRAW:
                self.turn_count += 1
                self.drew_flags[p][self.active_color] = True
                drawn = self._draw(p, 1)
                drawn_count = len(drawn)
                card = drawn[0] if drawn else None
                playable = card is not None and (card >= CARD_WILD or self._matches(card))
                if playable:
                    self.phase = PHASE_POST_DRAW
                    self.last_drawn = card
                    text = (f"You drew {card_name(card)} — it's playable!" if p == 0
                            else "R3X drew a card…")
                else:
                    self.phase = PHASE_MAIN
                    self.last_drawn = -1
                    switch = True
                    text = ("You had nothing to draw — turn passes." if card is None else
                            (f"You drew {card_name(card)} — no match." if p == 0
                             else "R3X drew a card — no match."))
                ev = {"type": "draw", "actor": p, "count": drawn_count, "text": text}
                if p == 0 and drawn:
                    ev["card_ids"] = drawn
                self._emit(ev)
            else:
                self.turn_count += 1
                card_id, dec = self._decode(action)
                text = f"{me} played {card_name(card_id)}"
                if dec >= 0:
                    text += f" and called {COL_NAMES[dec]}"
                self._emit({"type": "play_card", "actor": p, "card_id": card_id,
                            "active_color": dec if dec >= 0 else card_id // 13,
                            "text": text + "."})
                grant_extra, declared = self._exec_play(p, opp, action)
                switch = not grant_extra
        else:  # PHASE_POST_DRAW
            if action == ACTION_PASS:
                self.phase = PHASE_MAIN
                self.last_drawn = -1
                switch = True
                self._emit({"type": "pass", "actor": p,
                            "text": f"{me} kept the drawn card and passed."})
            else:
                card_id, dec = self._decode(action)
                text = f"{me} played {card_name(card_id)}"
                if dec >= 0:
                    text += f" and called {COL_NAMES[dec]}"
                self._emit({"type": "play_card", "actor": p, "card_id": card_id,
                            "active_color": dec if dec >= 0 else card_id // 13,
                            "text": text + "."})
                grant_extra, declared = self._exec_play(p, opp, action)
                self.phase = PHASE_MAIN
                self.last_drawn = -1
                switch = not grant_extra

        if grant_extra:
            card_id, _ = self._decode(action)
            is_rev = card_id < 52 and card_id % 13 == 11
            if p == 0:
                t2 = "Flow reverses — go again!" if is_rev else "R3X is skipped — go again!"
            else:
                t2 = "Flow reverses — you are skipped!" if is_rev else "You are skipped — R3X goes again!"
            self._emit({"type": "skip", "actor": 1 - p, "text": t2})

        # Move history (color feature = declared if any, else post-move active color)
        dc = declared if declared != -1 else self.active_color
        self.move_history.append((p, action, dc, drawn_count))
        if len(self.move_history) > HISTORY_LEN:
            self.move_history.pop(0)

        # Termination — engine order: empty hand first, then turn-limit tiebreak
        my_total, opp_total = self.hand_size(p), self.hand_size(opp)
        if my_total == 0:
            self._finish(p)
            self._emit({"type": "game_over", "winner": p,
                        "text": "You win the duel!" if p == 0 else "R3X wins the duel."})
        elif self.turn_count >= self.max_turns:
            if my_total < opp_total:
                w = p
            elif opp_total < my_total:
                w = opp
            else:
                w = -1
            self._finish(w)
            self._emit({"type": "game_over", "winner": w, "text": (
                "Turn limit reached — you win on card count." if w == 0 else
                "Turn limit reached — R3X wins on card count." if w == 1 else
                "Turn limit reached — the duel ends in a draw.")})
        elif switch:
            self.current_player = opp

        self.legal = self.legal_actions()
        last_text = self.events[-1].get("text", "") if self.events else ""
        self._record(last_text)