import { API_BASE, USE_MOCK } from '../config';
import type { Difficulty, PublicGameState, ReplayPayload } from '../types';
import { mockStartGame, mockPlayAction, mockGetReplay, mockGetGameState } from './mock/server';

// ── Cold-start tolerant fetch (Render free tier sleeps → first call pays a wake penalty) ──
type WakeListener = (waking: boolean) => void;
let wakeListener: WakeListener | null = null;
export const onServerWake = (fn: WakeListener) => { wakeListener = fn; return () => { wakeListener = null; }; };

const RETRY_STATUS = new Set([502, 503, 504]);
const backoff = (attempt: number) => new Promise(r => setTimeout(r, 4000 + attempt * 6000));

async function fetchWithWake(path: string, init: RequestInit,
                             opts: { timeoutMs?: number; retries?: number } = {}): Promise<Response> {
  const { timeoutMs = 75000, retries = 0 } = opts;
  for (let attempt = 0; ; attempt++) {
    const ctl = new AbortController();
    const tm = setTimeout(() => ctl.abort(), timeoutMs);
    try {
      const res = await fetch(`${API_BASE}${path}`, { ...init, signal: ctl.signal });
      clearTimeout(tm);
      if (RETRY_STATUS.has(res.status) && attempt < retries) {
        wakeListener?.(true);
        await backoff(attempt);
        continue;
      }
      wakeListener?.(false);
      return res;
    } catch (e: any) {
      clearTimeout(tm);
      // startGame / GETs are safe to retry. playAction is NOT (could double-apply) —
      // the hook recovers it via getGameState instead.
      if (attempt < retries && (e?.name === 'AbortError' || e?.name === 'TypeError')) {
        wakeListener?.(true);
        await backoff(attempt);
        continue;
      }
      wakeListener?.(false);
      throw e;
    }
  }
}

async function post<T>(path: string, body: unknown, retries = 0): Promise<T> {
  const res = await fetchWithWake(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }, { retries });
  if (!res.ok) {
    const detail = await res.json().then((j: any) => j?.detail).catch(() => null);
    throw new Error(detail ?? `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

// ── Swap to the real FastAPI backend by setting VITE_USE_MOCK=false ──

export const startGame = (difficulty: Difficulty): Promise<PublicGameState> =>
  USE_MOCK ? mockStartGame(difficulty)
           : post<PublicGameState>('/game/start', { difficulty }, 2);   // retryable: worst case orphans a session (swept)

export const playAction = (gameId: string, action: number): Promise<PublicGameState> =>
  USE_MOCK ? mockPlayAction(gameId, action)
           : post<PublicGameState>('/game/action', { game_id: gameId, action }, 0);  // no auto-retry (unsafe)

export const getGameState = async (gameId: string): Promise<PublicGameState> => {
  if (USE_MOCK) return mockGetGameState(gameId);
  const res = await fetchWithWake(`/game/${encodeURIComponent(gameId)}`,
                                  { method: 'GET' }, { retries: 1 });
  if (!res.ok) {
    const detail = await res.json().then((j: any) => j?.detail).catch(() => null);
    throw new Error(detail ?? `Request failed (${res.status})`);
  }
  return res.json() as Promise<PublicGameState>;
};

export const getReplay = async (code: string): Promise<ReplayPayload | null> => {
  if (USE_MOCK) return mockGetReplay(code);
  const res = await fetchWithWake(`/replay/${encodeURIComponent(code)}`, { method: 'GET' },
                                  { timeoutMs: 90000, retries: 2 });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`Replay fetch failed (${res.status})`);
  return res.json() as Promise<ReplayPayload>;
};

// Fire-and-forget: ping on lobby mount so the server wakes WHILE the user reads the menu.
export const prewarmServer = () => {
  if (USE_MOCK) return;
  fetch(`${API_BASE}/health`).catch(() => {});
};