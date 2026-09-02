import { ALL_IDS, cardName, COL_NAMES, shuffle, sleep, typeOf } from '../../cards/constants';
import type { AnimEvent, Difficulty, ReplayFrame } from '../../types';
import { botPolicy, evalValue } from './bot';

export interface Engine {
  difficulty: Difficulty; deck: number[]; discard: number[]; hands: [number[], number[]];
  topCard: number; activeColor: number; turn: 0 | 1; phase: 0 | 1; drawnCard: number | null;
  actions: number; status: 'playing' | 'over'; winner: 0 | 1 | -1 | null;
  legal: number[]; startedAt: number; events: AnimEvent[]; frames: ReplayFrame[];
  duration: number;
}

const emit = (st: Engine, ev: AnimEvent) => { st.events.push(ev); };

const isPlayable = (st: Engine, c: number) =>
  c >= 52 || Math.floor(c / 13) === st.activeColor ||
  (st.topCard < 52 && typeOf(c) === typeOf(st.topCard));

function computeLegal(st: Engine): number[] {
  if (st.phase === 1) {
    const c = st.drawnCard!;
    const acts: number[] = [];
    if (isPlayable(st, c)) {
      if (c === 52) acts.push(52, 53, 54, 55);
      else if (c === 53) acts.push(56, 57, 58, 59);
      else acts.push(c);
    }
    acts.push(61);
    return acts;
  }
  const acts: number[] = [];
  st.hands[st.turn].forEach(c => {
    if (isPlayable(st, c)) {
      if (c === 52) acts.push(52, 53, 54, 55);
      else if (c === 53) acts.push(56, 57, 58, 59);
      else acts.push(c);
    }
  });
  const canDraw = st.deck.length > 0 || st.discard.length > 1;
  if (canDraw) acts.push(60);
  else if (acts.length === 0) acts.push(61);
  return acts;
}

function recordFrame(st: Engine, event: string) {
  st.frames.push({
    turn: st.actions, p0_hand: [...st.hands[0]], p1_hand: [...st.hands[1]],
    top_card: st.topCard, active_color: st.activeColor, bot_value: +evalValue(st).toFixed(3),
    deck_count: st.deck.length, event,
    deck: [...st.deck], discard: [...st.discard],
  });
}

export function createEngine(difficulty: Difficulty): Engine {
  const deck = shuffle(ALL_IDS);
  const p0: number[] = [], p1: number[] = [];
  for (let i = 0; i < 7; i++) { p0.push(deck.pop()!); p1.push(deck.pop()!); }
  while (!(deck[deck.length - 1] <= 51 && typeOf(deck[deck.length - 1]) <= 9)) deck.unshift(deck.pop()!);
  const start = deck.pop()!;
  const st: Engine = {
    difficulty, deck, discard: [start], hands: [p0, p1],
    topCard: start, activeColor: Math.floor(start / 13),
    turn: Math.random() < 0.5 ? 0 : 1,   // randomized starting player (per contract)
    phase: 0, drawnCard: null, actions: 0,
    status: 'playing', winner: null, legal: [],
    startedAt: Date.now(), events: [], frames: [],
  };
  st.legal = computeLegal(st);
  emit(st, {
    type: 'deal', human_cards: [...p0], bot_count: 7, starter_card: start,
    text: st.turn === 0 ? 'Cards dealt — you lead.' : 'Cards dealt — R3X leads.',
  });
  recordFrame(st, 'Match start — cards dealt.');
  return st;
}

function drawOne(st: Engine): number | null {
  if (st.deck.length === 0) {
    if (st.discard.length <= 1) return null;
    const top = st.discard.pop()!;
    st.deck = shuffle(st.discard);
    st.discard = [top];
    emit(st, { type: 'reshuffle', deck_count: st.deck.length, text: 'Deck exhausted — discards reshuffled.' });
  }
  return st.deck.pop() ?? null;
}

function drawCards(st: Engine, p: 0 | 1, n: number): number[] {
  const got: number[] = [];
  for (let i = 0; i < n; i++) {
    const c = drawOne(st);
    if (c == null) break;
    st.hands[p].push(c);
    got.push(c);
  }
  const ev: AnimEvent = {
    type: 'draw', actor: p, count: got.length,
    text: p === 0
      ? `You draw ${got.length === 1 ? cardName(got[0]) : `${got.length} cards`}.`
      : `R3X draws ${got.length} card${got.length === 1 ? '' : 's'}.`,
  };
  if (p === 0 && got.length) (ev as any).card_ids = [...got]; // human's cards are public
  emit(st, ev);
  return got;
}

function finish(st: Engine, winner: 0 | 1 | -1) {
  st.status = 'over';
  st.winner = winner;
  st.legal = [];
  st.duration = (Date.now() - st.startedAt) / 1000;
}
// small augment for finish()
declare module './engine' {}
interface EngineExtra { duration?: number }
(Object.prototype as any); // (no-op guard, see finishSave below)

function saveReplayLocal(st: Engine, code: string) {
  try {
    const payload = {
      code, difficulty: st.difficulty, winner: st.winner ?? -1,
      total_turns: st.actions, duration_seconds: Math.round((st as any).duration ?? 0),
      created_at: new Date().toISOString(), frames: st.frames,
    };
    const key = 'uno3d_replays_v1';
    const db = JSON.parse(localStorage.getItem(key) || '{}');
    db[code] = payload;
    const keys = Object.keys(db).sort((a, b) => db[a].created_at.localeCompare(db[b].created_at));
    while (keys.length > 12) delete db[keys.shift()!];
    localStorage.setItem(key, JSON.stringify(db));
  } catch { /* storage unavailable — replays stay in-memory for this session */ }
}

function applyAction(st: Engine, p: 0 | 1, action: number) {
  st.actions++;
  const me = p === 0 ? 'You' : 'R3X';
  const them = p === 0 ? 'R3X' : 'you';

  if (action === 60) {
    const card = drawOne(st);
    if (card == null) {
      st.turn = (1 - p) as 0 | 1;
      emit(st, { type: 'notice', text: `${me} had nothing to draw — turn passes.` });
    } else {
      st.hands[p].push(card);
      const playable = isPlayable(st, card);
      if (playable) {
        st.phase = 1; st.drawnCard = card;
        emit(st, { type: 'draw', actor: p, count: 1, card_ids: p === 0 ? [card] : undefined, text: p === 0 ? `You drew ${cardName(card)} — it's playable!` : 'R3X drew a card…' });
      } else {
        st.phase = 0; st.drawnCard = null; st.turn = (1 - p) as 0 | 1;
        emit(st, { type: 'draw', actor: p, count: 1, card_ids: p === 0 ? [card] : undefined, text: p === 0 ? `You drew ${cardName(card)} — no match.` : 'R3X drew a card — no match.' });
      }
    }
  } else if (action === 61) {
    st.phase = 0; st.drawnCard = null; st.turn = (1 - p) as 0 | 1;
    emit(st, { type: 'pass', actor: p, text: `${me} kept the drawn card and passed.` });
  } else {
    let cardId: number, chosen: number;
    if (action <= 51) { cardId = action; chosen = -1; }
    else if (action <= 55) { cardId = 52; chosen = action - 52; }
    else { cardId = 53; chosen = action - 56; }

    st.hands[p] = st.hands[p].filter(c => c !== cardId);
    st.discard.push(cardId);
    st.topCard = cardId;
    st.activeColor = cardId >= 52 ? chosen : Math.floor(cardId / 13);
    st.phase = 0; st.drawnCard = null;

    let text = `${me} played ${cardName(cardId)}`;
    if (cardId >= 52) text += ` and called ${COL_NAMES[chosen]}`;

    if (st.hands[p].length === 0) {
      finish(st, p);
      emit(st, { type: 'play_card', actor: p, card_id: cardId, active_color: st.activeColor, text: text + (p === 0 ? ' — YOU WIN!' : ' — R3X wins.') });
      emit(st, { type: 'game_over', winner: p, text: p === 0 ? 'You win!' : 'R3X wins.' });
      recordFrame(st, text);
      return;
    }
    emit(st, { type: 'play_card', actor: p, card_id: cardId, active_color: st.activeColor, text: text + '.' });

    let again = false;
    const t = cardId >= 52 ? -1 : typeOf(cardId);
    if (t === 10) {
      again = true;
      emit(st, { type: 'skip', actor: (1 - p) as 0 | 1, text: p === 0 ? 'R3X is skipped — go again!' : 'You are skipped!' });
    } else if (t === 11) {
      again = true;
      emit(st, { type: 'skip', actor: (1 - p) as 0 | 1, text: p === 0 ? 'Flow reverses — go again!' : 'Flow reverses — you are skipped!' });
    } else if (t === 12) {
      drawCards(st, (1 - p) as 0 | 1, 2);
      again = true;
      emit(st, { type: 'skip', actor: (1 - p) as 0 | 1, text: p === 0 ? 'R3X is skipped — your move again.' : 'You are skipped — R3X goes again.' });
    } else if (cardId === 53) {
      drawCards(st, (1 - p) as 0 | 1, 4);
      again = true;
      emit(st, { type: 'skip', actor: (1 - p) as 0 | 1, text: p === 0 ? 'R3X is skipped — your move again.' : 'You are skipped — R3X goes again.' });
    }
    st.turn = again ? p : (1 - p) as 0 | 1;
  }

  st.legal = computeLegal(st);
  if (st.actions >= 200 && st.status === 'playing') {
    finish(st, -1);
    emit(st, { type: 'notice', text: 'Turn limit reached — the duel ends in a draw.' });
    emit(st, { type: 'game_over', winner: -1, text: 'Draw — turn limit.' });
  }
  const last = st.events[st.events.length - 1];
  recordFrame(st, last && 'text' in last ? (last as any).text : '');
}

/** Resolves every consecutive bot turn until control returns to the human. */
export function runBot(st: Engine) {
  let guard = 0;
  while (st.status === 'playing' && st.turn === 1 && guard++ < 300) {
    applyAction(st, 1, botPolicy(st));
  }
}

export function applyActionChecked(st: Engine, p: 0 | 1, action: number) {
  if (st.status !== 'playing') throw new Error('This match is already over.');
  if (st.turn !== p) throw new Error('It is not your turn.');
  if (!st.legal.includes(action)) throw new Error('That action is not legal right now.');
  applyAction(st, p, action);
}

export function saveOnFinish(st: Engine, code: string) {
  if (st.status === 'over') saveReplayLocal(st, code);
}