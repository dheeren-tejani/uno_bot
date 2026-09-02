"""
mcts_baseline.py
Information Set Monte Carlo Tree Search (ISMCTS) baseline for 2-player UNO.

Method:
  - Determinization: at every decision, sample a plausible world consistent with
    the actor's information set (own hand + public discard counts). The opponent
    hand + draw pile are dealt uniformly from the unseen-card multiset.
  - Search: vanilla UCB1 MCTS on the determinized world (single tree, rebuilt
    fresh each decision -- standard practical ISMCTS for card games).
  - Playouts: random moves truncated at `playout_depth`; unfinished leaves are
    scored tanh((opp_cards - my_cards) / div), terminals score +/-1 / 0.

Rules intentionally mirror uno_engine.VectorizedUnoEnv exactly, INCLUDING its
quirks (e.g. post-draw Wild+4 bypasses the strict color-match rule), verified
by a transition-parity test against the engine.
"""

import math
import random
from typing import List, Optional, Tuple

ACTION_DRAW = 60
ACTION_PASS = 61
CARD_WILD = 52
CARD_WILD_DF = 53

# Standard 108-card deck composition indexed by card id 0..53
TOTAL_COUNTS: List[int] = [
    (1 if (i % 13) == 0 else 2) if i < CARD_WILD else 4 for i in range(54)
]


def _card_color(cid: int) -> int:
    return cid // 13 if cid < CARD_WILD else -1


def _card_type(cid: int) -> int:
    return cid % 13 if cid < CARD_WILD else cid


class MCTSUnoState:
    """Lightweight, clonable 2-player UNO state mirroring uno_engine rules."""
    __slots__ = (
        "hands", "draw_pile", "discard_counts", "discarded_total",
        "top_card", "active_color", "current_player", "phase",
        "last_drawn", "turn_count", "max_turns", "rng",
    )

    PHASE_MAIN = 0
    PHASE_POST_DRAW = 1

    def __init__(self, max_turns: int, rng: random.Random):
        self.hands: List[List[int]] = [[0] * 54, [0] * 54]
        self.draw_pile: List[int] = []
        self.discard_counts: List[int] = [0] * 54
        self.discarded_total: int = 0
        self.top_card: int = 0
        self.active_color: int = 0
        self.current_player: int = 0
        self.phase: int = self.PHASE_MAIN
        self.last_drawn: int = -1
        self.turn_count: int = 0
        self.max_turns: int = max_turns
        self.rng: random.Random = rng

    # ---------- construction ----------
    @classmethod
    def from_env(cls, env, env_idx: int = 0) -> "MCTSUnoState":
        st = cls(int(env.max_turns), random.Random())
        st.rng = env.rng if isinstance(env.rng, random.Random) else random.Random()
        st.hands[0] = [int(x) for x in env.hands[env_idx, 0]]
        st.hands[1] = [int(x) for x in env.hands[env_idx, 1]]
        pos = int(env.draw_pile_pos[env_idx])
        st.draw_pile = list(env.draw_piles[env_idx][:pos])
        st.discard_counts = [int(x) for x in env.discard_counts[env_idx]]
        st.discarded_total = int(sum(st.discard_counts))
        st.top_card = int(env.top_card[env_idx])
        st.active_color = int(env.active_color[env_idx])
        st.current_player = int(env.current_player[env_idx])
        st.phase = int(env.current_phase[env_idx])
        st.last_drawn = int(env.last_drawn_card[env_idx])
        st.turn_count = int(env.turn_counts[env_idx])
        return st

    def clone(self) -> "MCTSUnoState":
        st = MCTSUnoState.__new__(MCTSUnoState)
        st.hands = [self.hands[0][:], self.hands[1][:]]
        st.draw_pile = self.draw_pile[:]
        st.discard_counts = self.discard_counts[:]
        st.discarded_total = self.discarded_total
        st.top_card = self.top_card
        st.active_color = self.active_color
        st.current_player = self.current_player
        st.phase = self.phase
        st.last_drawn = self.last_drawn
        st.turn_count = self.turn_count
        st.max_turns = self.max_turns
        st.rng = self.rng
        return st

    # ---------- helpers ----------
    def hand_size(self, p: int) -> int:
        return sum(self.hands[p])

    def _is_legal_card(self, cid: int) -> bool:
        if cid >= CARD_WILD:
            return True
        top_type = _card_type(self.top_card)
        return (_card_color(cid) == self.active_color) or (_card_type(cid) == top_type)

    def _recycle_if_needed(self, needed: int):
        if len(self.draw_pile) >= needed:
            return
        if self.discarded_total > 1:
            # Recyclable multiset = everything except the current top card,
            # which alone remains in the discard pile (mirrors engine).
            pool_counts = self.discard_counts[:]
            pool_counts[self.top_card] -= 1
            recycled: List[int] = []
            for cid, cnt in enumerate(pool_counts):
                recycled.extend([cid] * cnt)
            self.rng.shuffle(recycled)
            self.draw_pile.extend(recycled)
            fresh_discard = [0] * 54
            fresh_discard[self.top_card] = 1
            self.discard_counts = fresh_discard
            self.discarded_total = 1

    def _draw(self, p: int, count: int) -> Optional[int]:
        self._recycle_if_needed(count)
        actual = min(count, len(self.draw_pile))
        last: Optional[int] = None
        for _ in range(actual):
            c = self.draw_pile.pop(0)
            self.hands[p][c] += 1
            last = c
        return last

    # ---------- legality ----------
    def legal_actions(self) -> List[int]:
        acts: List[int] = []
        hand = self.hands[self.current_player]
        if self.phase == self.PHASE_MAIN:
            top_type = _card_type(self.top_card)
            for cid in range(CARD_WILD):
                if hand[cid] > 0 and (
                    _card_color(cid) == self.active_color or _card_type(cid) == top_type
                ):
                    acts.append(cid)
            if hand[CARD_WILD] > 0:
                acts.extend(range(CARD_WILD, CARD_WILD + 4))
            if hand[CARD_WILD_DF] > 0:
                no_match = sum(
                    hand[self.active_color * 13:(self.active_color + 1) * 13]
                ) == 0
                if no_match:
                    acts.extend(range(CARD_WILD + 4, CARD_WILD + 8))
            acts.append(ACTION_DRAW)
        else:
            drawn = self.last_drawn
            if drawn != -1:
                if drawn < CARD_WILD:
                    acts.append(drawn)
                elif drawn == CARD_WILD:
                    acts.extend(range(CARD_WILD, CARD_WILD + 4))  # engine quirk: no strict check here
                else:
                    acts.extend(range(CARD_WILD + 4, CARD_WILD + 8))
            acts.append(ACTION_PASS)
        return acts

    # ---------- execution ----------
    def _execute_play(self, p: int, opp: int, act: int) -> bool:
        """act is an ACTION id; wild declarations map to their underlying card id."""
        extra = False
        if act < CARD_WILD:
            cid = act
            self.active_color = act // 13
            ctype = act % 13
        elif act < CARD_WILD + 4:
            cid = CARD_WILD
            self.active_color = act - CARD_WILD
            ctype = -1
        else:
            cid = CARD_WILD_DF
            self.active_color = act - (CARD_WILD + 4)
            ctype = -1

        self.hands[p][cid] -= 1
        self.discard_counts[cid] += 1
        self.discarded_total += 1
        self.top_card = cid

        if ctype == 10 or ctype == 11:           # skip / reverse (2P: both skip)
            extra = True
        elif ctype == 12:                        # draw two
            self._draw(opp, 2)
            extra = True
        elif cid == CARD_WILD_DF:                # wild draw four
            self._draw(opp, 4)
            extra = True
        return extra

    def apply(self, act: int) -> Tuple[bool, int, float]:
        """Mirrors VectorizedUnoEnv.step for a single env. Returns (done, winner, reward-from-actor)."""
        p = self.current_player
        opp = 1 - p
        switch = False

        if self.phase == self.PHASE_MAIN:
            if act == ACTION_DRAW:
                drawn = self._draw(p, 1)
                self.turn_count += 1
                if drawn is not None and self._is_legal_card(drawn):
                    self.phase = self.PHASE_POST_DRAW
                    self.last_drawn = drawn
                else:
                    self.last_drawn = -1
                    switch = True
            else:
                self.turn_count += 1
                extra = self._execute_play(p, opp, act)
                switch = not extra
        else:
            if act == ACTION_PASS:
                self.phase = self.PHASE_MAIN
                self.last_drawn = -1
                switch = True
            else:
                extra = self._execute_play(p, opp, act)
                self.phase = self.PHASE_MAIN
                self.last_drawn = -1
                switch = not extra

        # Termination check (after the action, before switching -- engine order)
        mine = self.hand_size(p)
        if mine == 0:
            return True, p, 1.0
        if self.turn_count >= self.max_turns:
            theirs = self.hand_size(opp)
            if mine < theirs:
                return True, p, 1.0
            if mine > theirs:
                return True, opp, -1.0
            return True, -1, 0.0
        if switch:
            self.current_player = opp
        return False, -1, 0.0

    # ---------- determinization ----------
    def determinize(self, rng: random.Random) -> "MCTSUniWorld":
        world = self.clone()
        me = world.current_player
        opp = 1 - me
        unseen: List[int] = []
        for cid in range(54):
            n = TOTAL_COUNTS[cid] - world.hands[me][cid] - world.discard_counts[cid]
            if n > 0:
                unseen.extend([cid] * n)
            elif n < 0:
                raise ValueError(f"information-set underflow at card {cid}")
        rng.shuffle(unseen)
        opp_size = world.hand_size(opp)
        opp_cards, rest = unseen[:opp_size], unseen[opp_size:]
        # Replace the true hidden hand with the sampled one (do not accumulate!)
        world.hands[opp] = [0] * 54
        for cid in opp_cards:
            world.hands[opp][cid] += 1
        world.draw_pile = rest
        return world


# Alias kept for readability at call sites
MCTSUnoWorld = MCTSUnoState


class _Node:
    __slots__ = ("action", "player", "parent", "children", "untried", "visits", "value")

    def __init__(self, action: Optional[int], player: int, parent: Optional["_Node"]):
        self.action = action
        self.player = player          # player_just_moved (who chose `action`)
        self.parent = parent
        self.children: List[_Node] = []
        self.untried: List[int] = []
        self.visits = 0
        self.value = 0.0              # cumulative value from `player`'s perspective


def _simulate(state: MCTSUnoState, depth_limit: int, div: float) -> Tuple[float, float]:
    """Random playout, truncated. Returns (z_p0, z_p1), zero-sum."""
    for _ in range(depth_limit):
        acts = state.legal_actions()
        a = acts[state.rng.randrange(len(acts))]
        done, winner, _ = state.apply(a)
        if done:
            if winner == -1:
                return 0.0, 0.0
            z = 1.0 if winner == 0 else -1.0
            return z, -z
    h0, h1 = state.hand_size(0), state.hand_size(1)
    z0 = math.tanh((h1 - h0) / div)
    return z0, -z0


def ismcts_select_action(
    state: MCTSUnoState,
    simulations: int,
    rng: random.Random,
    ucb_c: float = 1.2,
    playout_depth: int = 48,
    playout_div: float = 6.0,
) -> int:
    """Runs ISMCTS from `state` (whose true hidden info is respected) and returns the best action."""
    legal = state.legal_actions()
    if len(legal) == 1:
        return legal[0]

    root_world = state.determinize(rng)
    root = _Node(None, root_world.current_player, None)
    root.untried = legal[:]
    rng.shuffle(root.untried)

    for _ in range(simulations):
        node = root
        world = root_world.clone()
        terminal = False
        z0 = z1 = 0.0

        # 1. Selection (fully expanded interior)
        while not node.untried and node.children:
            best, best_ucb = None, -float("inf")
            log_n = math.log(node.visits + 1)
            for ch in node.children:
                ucb = (ch.value / ch.visits) + ucb_c * math.sqrt(log_n / ch.visits)
                if ucb > best_ucb:
                    best_ucb, best = ucb, ch
            node = best
            done, winner, _ = world.apply(node.action)
            if done:
                terminal = True
                if winner == -1:
                    z0 = z1 = 0.0
                else:
                    zw = 1.0 if winner == 0 else -1.0
                    z0, z1 = zw, -zw
                break

        # 2. Expansion
        if not terminal and node.untried:
            a = node.untried.pop()
            child = _Node(a, world.current_player, node)
            node.children.append(child)
            node = child
            done, winner, _ = world.apply(a)
            if done:
                terminal = True
                if winner == -1:
                    z0 = z1 = 0.0
                else:
                    zw = 1.0 if winner == 0 else -1.0
                    z0, z1 = zw, -zw

        # 3. Simulation
        if not terminal:
            z0, z1 = _simulate(world, playout_depth, playout_div)

        # 4. Backup (each node scored from its own mover's perspective)
        z_by_player = (z0, z1)
        while node is not None:
            node.visits += 1
            node.value += z_by_player[node.player]
            node = node.parent

    return max(root.children, key=lambda ch: ch.visits).action
