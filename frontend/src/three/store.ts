import { create } from 'zustand';
import type { AnimEvent, PublicGameState, ReplayFrame } from '../types';
import { CardInst, TOTAL_CARDS, View } from './layout';

interface SceneStore {
  mode: 'lobby' | 'game' | 'replay';
  view: View;
  revealBot: boolean;
  interactive: boolean;
  legalIds: Set<number>;
  activeColor: number;
  deckClickable: boolean;
  delays: Map<number, number>;
  pendingPlayIid: number | null;
  onCardClick?: (cid: number, iid?: number) => void;
  onDeckClick?: () => void;
  showLobby: () => void;
  gather: () => void;
  applyDeal: (ev: Extract<AnimEvent, { type: 'deal' }>) => void;
  applyEvent: (ev: AnimEvent) => void;
  syncFinal: (gs: PublicGameState) => void;
  applyReplayFrame: (f: ReplayFrame, reveal: boolean) => void;
  setRevealBot: (v: boolean) => void;
  setPendingPlayIid: (iid: number | null) => void;
  setCallbacks: (c: { onCardClick?: (cid: number, iid?: number) => void; onDeckClick?: () => void }) => void;
}

let _iid = 0;
const nextIid = () => ++_iid;
const anon = (): CardInst => ({ iid: nextIid(), cid: -1 });

function showcaseTargets() {
  const hand = [3, 16, 27, 33, 45, 22, 52];
  const bot = [8, 21, 34, 47, 53];
  const discard = [12, 24, 38];
  const deck = Array<number>(TOTAL_CARDS - hand.length - bot.length - discard.length).fill(-1);
  return { hand, bot, discard, deck };
}

/**
 * Re-assigns the 108 persistent card bodies to a target multiset per zone,
 * keeping identity stable wherever the same face is already on screen, so
 * cards FLY between zones instead of swapping textures.
 */
function remapView(v: View, t: { hand: number[]; bot: number[]; discard: number[]; deck: number[] }): View {
  const pool: Record<keyof View, CardInst[]> = {
    hand: v.hand.slice(), bot: v.bot.slice(),
    discard: v.discard.slice(), deck: v.deck.slice(),
  };
  const order: (keyof View)[] = ['hand', 'bot', 'discard', 'deck'];

  const take = (cid: number, prefer: keyof View): CardInst => {
    let i = pool[prefer].findIndex(x => x.cid === cid);
    if (i >= 0) return pool[prefer].splice(i, 1)[0];
    for (const z of order) {
      const j = pool[z].findIndex(x => x.cid === cid);
      if (j >= 0) return pool[z].splice(j, 1)[0];
    }
    for (const z of [prefer, ...order]) {
      if (pool[z].length) return pool[z].shift()!;
    }
    return { iid: nextIid(), cid: -1 };
  };

  const hand = t.hand.map(cid => ({ ...take(cid, 'hand'), cid }));
  const discard = t.discard.map(cid => ({ ...take(cid, 'discard'), cid }));
  const bot = t.bot.map(cid => ({ ...take(cid, 'bot'), cid }));
  const deck = t.deck.map(cid => ({ ...take(cid, 'deck'), cid }));
  return { hand, bot, discard, deck };
}

export const useSceneStore = create<SceneStore>((set, get) => ({
  mode: 'lobby',
  view: remapView({ hand: [], bot: [], discard: [], deck: [] }, showcaseTargets()),
  revealBot: false,
  interactive: false,
  legalIds: new Set(),
  activeColor: 2,
  deckClickable: false,
  delays: new Map(),
  pendingPlayIid: null,

  showLobby: () => set({
    mode: 'lobby', revealBot: false, interactive: false,
    legalIds: new Set(), activeColor: 2, deckClickable: false,
    delays: new Map(), pendingPlayIid: null,
    view: remapView(get().view, showcaseTargets()),
  }),

  // All 108 physical card bodies fly into one stacked draw pile.
  gather: () => {
    const v = get().view;
    const all = [...v.hand, ...v.bot, ...v.discard, ...v.deck];
    while (all.length < TOTAL_CARDS) all.push(anon());
    set({
      mode: 'game', interactive: false, legalIds: new Set(), activeColor: 0,
      deckClickable: false, delays: new Map(), pendingPlayIid: null,
      view: { hand: [], bot: [], discard: [], deck: all },
    });
  },

  // Deal: pop bodies off the pile; bot's stay face-down and fly to the fan;
  // starter + human cards get their faces and flip mid-flight into position.
  applyDeal: (ev) => {
    let deck = get().view.deck.slice();
    const delays = new Map<number, number>();
    const hand: CardInst[] = [], bot: CardInst[] = [], discard: CardInst[] = [];
    const pop = (): CardInst => (deck.length ? deck.pop()! : anon());
    const starter = { ...pop(), cid: ev.starter_card };
    delays.set(starter.iid, 0.05);
    discard.push(starter);
    for (let i = 0; i < ev.bot_count; i++) {
      const inst = pop();
      delays.set(inst.iid, 0.10 + i * 0.13);
      bot.push(inst);
    }
    ev.human_cards.forEach((cid, i) => {
      const inst = pop();
      delays.set(inst.iid, 0.12 + i * 0.13);
      hand.push({ ...inst, cid });
    });
    set({
      view: { hand, bot, discard, deck }, delays,
      activeColor: Math.floor(ev.starter_card / 13), pendingPlayIid: null,
    });
  },

  // Incremental transitions from the backend's animation_queue.
  applyEvent: (ev) => {
    const v = get().view;
    const delays = new Map<number, number>();
    let hand = v.hand.slice();
    let bot = v.bot.slice();
    let deck = v.deck.slice();
    let discard = v.discard.slice();

    switch (ev.type) {
      case 'play_card': {
        let played: CardInst;
        if (ev.actor === 0) {
          // Prefer the exact card body the player clicked; else the NEWEST
          // instance with that face (covers duplicates & just-drawn cards).
          const pending = get().pendingPlayIid;
          let idx = pending != null
            ? hand.findIndex(x => x.iid === pending && x.cid === ev.card_id)
            : -1;
          if (idx < 0) {
            for (let i = hand.length - 1; i >= 0; i--) {
              if (hand[i].cid === ev.card_id) { idx = i; break; }
            }
          }
          played = idx >= 0 ? hand.splice(idx, 1)[0]
                            : { iid: nextIid(), cid: ev.card_id };
        } else {
          // R3X reveals: rightmost fan body gets the real face and flies to
          // the pile, flipping face-up mid-air.
          const n = bot.length;
          played = n > 0 ? { ...bot[n - 1], cid: ev.card_id }
                         : { iid: nextIid(), cid: ev.card_id };
          bot = bot.slice(0, n - 1);
        }
        discard.push(played);        // ← THE core fix: the body stays mounted and flies to the pile
        set({
          activeColor: ev.active_color,
          pendingPlayIid: null,
          delays: new Map(),
          view: { hand, bot, deck, discard },
        });
        return;
      }
      case 'draw': {
        const ids = ev.card_ids ?? [];
        if (ev.actor === 0) {
          ids.forEach((cid, i) => {
            const top = deck.length ? deck.pop()! : anon();
            const inst = { ...top, cid };
            delays.set(inst.iid, i * 0.12);
            hand.push(inst);
          });
          for (let i = ids.length; i < ev.count; i++) {   // defensive
            const top = deck.length ? deck.pop()! : anon();
            delays.set(top.iid, i * 0.12);
            hand.push(top);
          }
        } else {
          for (let i = 0; i < ev.count; i++) {
            const top = deck.length ? deck.pop()! : anon();
            delays.set(top.iid, i * 0.12);
            bot.push(top);
          }
        }
        break;
      }
      case 'reshuffle': {
        // Deck recycle: everything except the top card flies back into the
        // draw pile face-down, with a small stagger so it reads as a cascade.
        const top = discard.length ? discard[discard.length - 1] : null;
        const recycled = discard.slice(0, -1).map((x, i) => {
          delays.set(x.iid, Math.min(i * 0.008, 0.5));
          return { ...x, cid: -1 };
        });
        discard = top ? [top] : [];
        deck = [...recycled, ...deck];   // recycled to the bottom — mirrors the engine
        break;
      }
      default:
        return;                          // skip / pass / notice / game_over — narration only
    }
    set({ view: { hand, bot, deck, discard }, delays });
  },

  // Reconcile to the authoritative public state (multiset-aware, duplicate-safe).
  syncFinal: (gs) => {
    const v = get().view;
    const need = new Map<number, number>();
    gs.hand.forEach(c => need.set(c, (need.get(c) ?? 0) + 1));
    const hand: CardInst[] = [];
    let deck = v.deck.slice();
    v.hand.forEach(inst => {
      const n = inst.cid >= 0 ? (need.get(inst.cid) ?? 0) : 0;
      if (n > 0) { need.set(inst.cid, n - 1); hand.push(inst); }
      else deck.unshift({ ...inst, cid: -1 });   // stray body → bottom of deck, face-down
    });
    need.forEach((n, cid) => {
      for (let k = 0; k < n; k++) {
        const top = deck.length ? deck[deck.length - 1] : null;
        if (top) { deck = deck.slice(0, -1); hand.push({ ...top, cid }); }
        else hand.push({ iid: nextIid(), cid });
      }
    });
    let bot = v.bot.slice();
    while (bot.length > gs.bot_card_count) {
      const extra = bot.pop()!;
      deck.unshift({ ...extra, cid: -1 });
    }
    while (bot.length < gs.bot_card_count) {
      const top = deck.length ? deck[deck.length - 1] : null;
      if (top) { deck = deck.slice(0, -1); bot.push(top); }
      else bot.push(anon());
    }
    while (deck.length > gs.deck_count) deck = deck.slice(0, -1);
    // Pad at the BOTTOM (least visible spot) — spawning at the top caused the
    // "a card popped into the deck out of nowhere" effect.
    while (deck.length < gs.deck_count) deck = [anon(), ...deck];
    let discard = v.discard.slice();
    const last = discard.length ? discard[discard.length - 1] : null;
    if (!last || last.cid !== gs.top_card) {
      if (last && last.cid < 0) discard[discard.length - 1] = { ...last, cid: gs.top_card };
      else discard.push({ iid: nextIid(), cid: gs.top_card });
    }
    const interactive = gs.status === 'playing' && gs.current_player === 0;
    const legalIds = new Set<number>();
    gs.legal_actions.forEach(a => {
      if (a <= 51) legalIds.add(a);
      else if (a <= 55) legalIds.add(52);
      else if (a <= 59) legalIds.add(53);
    });
    set({
      view: { hand, bot, discard, deck },
      interactive, legalIds, activeColor: gs.active_color,
      deckClickable: interactive && gs.legal_actions.includes(60),
      delays: new Map(), pendingPlayIid: null,
    });
  },

  applyReplayFrame: (f, reveal) => {
    const targets = {
      hand: f.p0_hand.slice(),
      bot: f.p1_hand.slice(),
      discard: f.discard && f.discard.length ? f.discard.slice() : [f.top_card],
      deck: f.deck ? f.deck.slice() : Array<number>(Math.max(0, f.deck_count)).fill(-1),
    };
    const v = remapView(get().view, targets);
    // Mask the bot hand when hidden: identity preserved (same bodies), faces hidden.
    const bot = reveal ? v.bot : v.bot.map(inst => ({ ...inst, cid: -1 }));
    set({
      mode: 'replay', revealBot: reveal, activeColor: f.active_color,
      view: { ...v, bot },
      interactive: false, legalIds: new Set(), deckClickable: false,
      delays: new Map(), pendingPlayIid: null,
    });
  },

  setRevealBot: (v) => set({ revealBot: v }),
  setPendingPlayIid: (iid) => set({ pendingPlayIid: iid }),
  setCallbacks: (c) => set({ onCardClick: c.onCardClick, onDeckClick: c.onDeckClick }),
}));