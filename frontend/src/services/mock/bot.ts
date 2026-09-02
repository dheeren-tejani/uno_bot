import { clamp, typeOf } from '../../cards/constants';
import type { Difficulty } from '../../types';
import type { Engine } from './engine';

const countColor = (hand: number[], c: number) => hand.filter(x => x < 52 && Math.floor(x / 13) === c).length;

/** Heuristic critic in [-1, +1] from the bot's perspective. */
export function evalValue(st: Engine): number {
  const h0 = st.hands[0].length, h1 = st.hands[1].length;
  let v = (h0 - h1) * 0.13;
  v += countColor(st.hands[1], st.activeColor) * 0.035;
  v -= countColor(st.hands[0], st.activeColor) * 0.03;
  if (h0 === 1) v -= 0.3;
  if (h1 === 1) v += 0.3;
  if (st.hands[1].includes(53)) v += 0.04;
  return clamp(v, -1, 1);
}

function chooseWildColor(st: Engine): number {
  const hand = st.hands[1];
  const counts = [0, 0, 0, 0];
  hand.forEach(c => { if (c < 52) counts[Math.floor(c / 13)]++; });
  if (st.difficulty !== 'hard') {
    let best = 0;
    for (let i = 1; i < 4; i++) if (counts[i] > counts[best]) best = i;
    return best;
  }
  // Hard: card-count the unseen pool and infer the human's holdings.
  const seen = [...st.discard, ...hand];
  const unseen = 54 - seen.length;
  const H = st.hands[0].length;
  let best = 0, bestS = -Infinity;
  for (let c = 0; c < 4; c++) {
    const seenC = seen.filter(x => x < 52 && Math.floor(x / 13) === c).length;
    const k = 13 - seenC;
    const pOpp = unseen > 0 ? 1 - Math.pow(Math.max(0, (unseen - k) / unseen), H) : 1;
    const s = counts[c] * 0.55 - pOpp * 1.1 + Math.random() * 0.05;
    if (s > bestS) { bestS = s; best = c; }
  }
  return best;
}

/** 3-tier policy: random / heuristic / card-counting. Returns action id 0-61. */
export function botPolicy(st: Engine): number {
  const legal = st.legal;
  if (st.phase === 1) {
    const plays = legal.filter(a => a !== 61);
    if (!plays.length) return 61;
    if (st.difficulty === 'easy' && Math.random() < 0.45) return 61;
    const a0 = plays[0];
    if (a0 >= 52) { const base = a0 <= 55 ? 52 : 56; return base + chooseWildColor(st); }
    return a0;
  }
  const plays = legal.filter(a => a !== 60);
  if (!plays.length) return legal.includes(60) ? 60 : 61;
  if (st.difficulty === 'easy') {
    if (legal.includes(60) && Math.random() < 0.22) return 60;
    return plays[Math.floor(Math.random() * plays.length)];
  }
  const hand = st.hands[1], oppLen = st.hands[0].length;
  const saveWild = st.difficulty === 'hard' ? 0.5 : 0.3;
  const uniq = plays.filter(a => a <= 51 || a === 52 || a === 56);
  let bestA = uniq[0], bestS = -Infinity;
  for (const a of uniq) {
    const card = a <= 51 ? a : (a === 52 ? 52 : 53);
    const t = card >= 52 ? (card === 52 ? 13 : 14) : typeOf(card);
    let s = 1;
    if (t <= 9) s = 1 + 0.3 * (card < 52 ? countColor(hand, Math.floor(card / 13)) : 0);
    else if (t === 10 || t === 11) s = 1.2 + (oppLen <= 2 ? 1.5 : 0);
    else if (t === 12) s = 1.05 + (oppLen <= 2 ? 1.6 : 0);
    else if (card === 52) s = 1 - saveWild + (oppLen <= 2 ? 1.2 : 0);
    else s = 0.8 - saveWild + (oppLen <= 2 ? 1.9 : 0);
    s += Math.random() * 0.12;
    if (s > bestS) { bestS = s; bestA = a; }
  }
  if (bestA >= 52) return bestA + chooseWildColor(st);
  return bestA;
}