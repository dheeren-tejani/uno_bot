import type { Difficulty, PublicGameState, ReplayPayload } from '../../types';
import { sleep } from '../../cards/constants';
import { applyActionChecked, createEngine, runBot, saveOnFinish, Engine } from './engine';

interface MockGame { id: string; code: string; engine: Engine; }
const games = new Map<string, MockGame>();

const CODE_CHARS = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
function genCode(): string {
  let c = '';
  do { c = ''; for (let i = 0; i < 6; i++) c += CODE_CHARS[Math.floor(Math.random() * CODE_CHARS.length)]; }
  while (loadDB()[c]);
  return c;
}
const genId = () =>
  (crypto as any)?.randomUUID?.() ?? `g-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;

function loadDB(): Record<string, ReplayPayload> {
  try { return JSON.parse(localStorage.getItem('uno3d_replays_v1') || '{}'); } catch { return {}; }
}

function publicState(g: MockGame): PublicGameState {
  const st = g.engine;
  const over = st.status === 'over';
  const base: PublicGameState = {
    game_id: g.id, replay_code: g.code, status: st.status,
    phase: st.phase, current_player: st.turn,
    turn_count: Math.min(st.actions + 1, 200),
    top_card: st.topCard, active_color: st.activeColor,
    hand: [...st.hands[0]], bot_card_count: st.hands[1].length, deck_count: st.deck.length,
    legal_actions: over ? [] : [...st.legal],
    drawn_card: st.drawnCard,
    animation_queue: st.events.splice(0),
  };
  if (over) { base.winner = st.winner ?? -1; base.duration_seconds = Math.round(st.duration); }
  return base;
}

export async function mockStartGame(difficulty: Difficulty): Promise<PublicGameState> {
  await sleep(140 + Math.random() * 200);   // simulated network + inference latency
  const g: MockGame = { id: genId(), code: genCode(), engine: createEngine(difficulty) };
  games.set(g.id, g);
  if (g.engine.turn === 1) runBot(g.engine);  // bot-led opener resolved before returning
  saveOnFinish(g.engine, g.code);
  return publicState(g);
}

export async function mockPlayAction(gameId: string, action: number): Promise<PublicGameState> {
  await sleep(160 + Math.random() * 260);
  const g = games.get(gameId);
  if (!g) throw new Error('Game not found — it may have expired.');
  applyActionChecked(g.engine, 0, action);
  runBot(g.engine);
  saveOnFinish(g.engine, g.code);
  return publicState(g);
}

export async function mockGetReplay(code: string): Promise<ReplayPayload | null> {
  await sleep(140);
  return loadDB()[code.toUpperCase()] ?? null;
}

export async function mockGetGameState(gameId: string): Promise<PublicGameState> {
  await sleep(80);
  const g = games.get(gameId);
  if (!g) throw new Error('Game not found — it may have expired.');
  return publicState(g);
}