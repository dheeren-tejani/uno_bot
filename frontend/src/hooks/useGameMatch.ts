import { useCallback, useEffect, useRef, useState } from 'react';
import { cardName, sleep } from '../cards/constants';
import { getGameState, onServerWake, playAction, startGame } from '../services/unoService';
import { useSceneStore } from '../three/store';
import type { AnimEvent, Difficulty, PublicGameState } from '../types';

/** Per-event pacing so each 3D transition is seen before the next begins. */
const STEP_MS: Record<AnimEvent['type'], number> = {
  deal: 2300, play_card: 750, draw: 520, skip: 800,
  pass: 800, reshuffle: 1200, notice: 800, game_over: 1000,
};

export function useGameMatch() {
  const [gs, setGs] = useState<PublicGameState | null>(null);
  const [busy, setBusy] = useState(false);
  const [botActive, setBotActive] = useState(false);
  const [toast, setToast] = useState<{ id: number; text: string } | null>(null);
  const [wild, setWild] = useState<{ base: number } | null>(null);

  const gsRef = useRef<PublicGameState | null>(null);
  const busyRef = useRef(false);
  const runRef = useRef(0);
  const toastId = useRef(0);
  const actRef = useRef<(a: number) => void>(() => {});

  const setBusyBoth = (v: boolean) => { busyRef.current = v; setBusy(v); };
  const say = (text: string) => setToast({ id: ++toastId.current, text });

  // Toast auto-clear.
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 3400);
    return () => clearTimeout(t);
  }, [toast]);

  // Cold-start wake notification (insurance for the free tier).
  useEffect(() => onServerWake(waking => {
    if (waking) say("R3X's server is waking from its nap — this can take a minute…");
  }), []);

  const processEvent = useCallback(async (ev: AnimEvent) => {
    const store = useSceneStore.getState();
    if (ev.type === 'deal') {
      store.applyDeal(ev);
      if (ev.text) say(ev.text);
      await sleep(STEP_MS.deal);
      return;
    }
    store.applyEvent(ev);
    if (ev.text) say(ev.text);
    if ('actor' in ev) setBotActive(ev.actor === 1);
    // Multi-card draws stagger at 120ms/card + flight time — wait for them.
    const wait = ev.type === 'draw'
      ? 520 + Math.max(0, ev.count - 1) * 240
      : STEP_MS[ev.type] ?? 600;
    await sleep(wait);
  }, []);

  const finishSync = useCallback((resp: PublicGameState) => {
    useSceneStore.getState().syncFinal(resp);
    gsRef.current = resp;
    setGs(resp);
    setBusyBoth(false);
    setBotActive(false);
  }, []);

  const start = useCallback(async (difficulty: Difficulty) => {
    const run = ++runRef.current;
    setBusyBoth(true); setBotActive(false);
    setGs(null); gsRef.current = null;
    useSceneStore.getState().gather();
    try {
      await sleep(650);                       // let the gather flourish land
      if (runRef.current !== run) return;
      const resp = await startGame(difficulty);
      for (const ev of resp.animation_queue) {
        if (runRef.current !== run) return;
        await processEvent(ev);
      }
      if (runRef.current !== run) return;
      finishSync(resp);
    } catch (e: any) {
      if (runRef.current === run) {
        say(e?.message ?? 'Could not reach the table server.');
        setBusyBoth(false); setBotActive(false);
      }
    }
  }, [processEvent, finishSync]);

  const act = useCallback(async (action: number) => {
    const st = gsRef.current;
    if (!st || st.status !== 'playing' || busyRef.current) return;
    if (!st.legal_actions.includes(action)) return;
    setWild(null);
    const run = ++runRef.current;
    setBusyBoth(true);
    // NOTE: no setBotActive(true) here — event actors drive the indicator, so
    // "R3X is plotting…" appears exactly during HIS events, not during your slam.
    try {
      const resp = await playAction(st.game_id, action);
      for (const ev of resp.animation_queue) {
        if (runRef.current !== run) return;
        await processEvent(ev);
      }
      if (runRef.current !== run) return;
      finishSync(resp);
    } catch (e: any) {
      if (runRef.current !== run) return;
      const msg: string = e?.message ?? '';
      // If the request may have landed server-side (or was rejected because
      // the move already applied), re-sync from the authoritative state.
      if (st.status === 'playing' &&
          (e?.name === 'TypeError' || /not your turn|already over/i.test(msg))) {
        try {
          const fresh = await getGameState(st.game_id);
          if (runRef.current !== run) return;
          finishSync(fresh);
          say('Connection hiccup — re-synced with the table.');
          return;
        } catch { /* fall through */ }
      }
      say(msg || 'Could not reach the table server.');
      setBusyBoth(false); setBotActive(false);
    }
  }, [processEvent, finishSync]);
  actRef.current = act;

  // Input translation: clicks → discrete action ids (wilds open the wheel).
  useEffect(() => {
    useSceneStore.getState().setCallbacks({
      onCardClick: (id: number, iid?: number) => {
        // Remember WHICH physical card body was clicked, so the right
        // duplicate flies to the pile.
        useSceneStore.getState().setPendingPlayIid(iid ?? null);
        const st = gsRef.current;
        if (!st || st.status !== 'playing' || busyRef.current) return;
        if (id === 52 && st.legal_actions.includes(52)) { setWild({ base: 52 }); return; }
        if (id === 53) {
          if (st.legal_actions.includes(56)) { setWild({ base: 56 }); return; }
          const holdsColor = st.hand.some(c => c < 52 && Math.floor(c / 13) === st.active_color);
          say(holdsColor
            ? 'Wild +4 is only legal when you hold no cards of the active color.'
            : 'No match — watch for the glowing edges.');
          return;
        }
        if (st.legal_actions.includes(id)) { actRef.current(id); return; }
        say(st.phase === 1
          ? 'Post-draw: only the card you just drew can be played.'
          : 'No match — watch for the glowing edges.');
      },
      onDeckClick: () => { actRef.current(60); },
    });
    return () => { useSceneStore.getState().setCallbacks({}); };
  }, []);

  const chooseWildColor = useCallback((c: number) => {
    if (!wild) return;
    actRef.current(wild.base + c);
  }, [wild]);

  const playDrawn = useCallback(() => {
    // Clear pending so the NEWEST body of that face (the drawn card) is the
    // one that flies — even if you already held a copy.
    useSceneStore.getState().setPendingPlayIid(null);
    const st = gsRef.current;
    if (!st || st.drawn_card == null) return;
    const c = st.drawn_card;
    if (c === 52) setWild({ base: 52 });
    else if (c === 53) setWild({ base: 56 });
    else actRef.current(c);
  }, []);

  return { gs, busy, botActive, toast, wild, start, act, chooseWildColor, cancelWild: () => setWild(null), playDrawn };
}