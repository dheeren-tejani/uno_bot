import { jitOf } from '../cards/constants';

export type Zone = 'deck' | 'hand' | 'bot' | 'discard';
export interface Slot { pos: [number, number, number]; rot: [number, number, number]; zone: Zone; top?: boolean; }
/** A physical card: iid = unique instance identity, cid = face (-1 = face-down unknown). */
export interface CardInst { iid: number; cid: number; }
export interface View { hand: CardInst[]; discard: CardInst[]; deck: CardInst[]; bot: CardInst[]; }

export const DECK_X = -2.45, DECK_Z = 0.3, DISC_X = 2.15, DISC_Z = 0.3;
export const TOTAL_CARDS = 108;

/** Discard pile physically grows, then compresses past ~22 cards. */
export const pileY = (i: number) => 0.016 + (i <= 22 ? i * 0.013 : 0.286 + (i - 22) * 0.0022);
/** The draw pile can hold up to 107 cards — compress past 40 so it stays a sane tower. */
export const deckY = (i: number) => 0.016 + (i <= 40 ? i * 0.0125 : 0.516 + (i - 40) * 0.0028);

export const deckStackSlot = (iid: number, i: number): Slot => {
  const j = jitOf(iid);
  return { pos: [DECK_X + j.x, deckY(i), DECK_Z + j.y], rot: [Math.PI / 2, 0, j.z], zone: 'deck' };
};
export const botFanSlot = (i: number, n: number): Slot => {
  const o = i - (n - 1) / 2;
  const spacing = n > 1 ? Math.min(0.62, 4.9 / (n - 1)) : 0;
  return { pos: [o * spacing * 0.92, 0.88 + i * 0.004, -2.5 + i * 0.004], rot: [-0.35, Math.PI, o * 0.05], zone: 'bot' };
};

export function buildLayout(view: View, revealBot: boolean, god: boolean): Map<number, Slot> {
  const m = new Map<number, Slot>();
  view.deck.forEach((inst, i) => m.set(inst.iid, { ...deckStackSlot(inst.iid, i), top: i === view.deck.length - 1 }));
  view.discard.forEach((inst, i) => {
    const j = jitOf(inst.iid);
    m.set(inst.iid, { pos: [DISC_X + j.x * 0.7, pileY(i), DISC_Z + j.y * 0.7], rot: [-Math.PI / 2, 0, j.z], zone: 'discard' });
  });
  const n = view.hand.length, c = (n - 1) / 2;
  const sp = n > 1 ? Math.min(0.62, 4.9 / (n - 1)) : 0;
  view.hand.forEach((inst, i) => {
    const o = i - c;
    m.set(inst.iid, { pos: [o * sp, 0.9 + i * 0.004, 2.94 + i * 0.005], rot: [-0.38, 0, -o * 0.055], zone: 'hand' });
  });
  const bn = view.bot.length, bc = (bn - 1) / 2;
  const bsp = bn > 1 ? Math.min(0.62, 4.9 / (bn - 1)) : 0;
  view.bot.forEach((inst, i) => {
    const o = i - bc;
    if (god && revealBot) {
      m.set(inst.iid, { pos: [o * bsp, 0.82 + i * 0.004, -2.4 + i * 0.004], rot: [-0.5, 0, -o * 0.05], zone: 'bot' });
    } else {
      m.set(inst.iid, botFanSlot(i, bn));
    }
  });
  return m;
}

export const slotKey = (s: Slot) =>
  `${s.pos[0].toFixed(3)},${s.pos[1].toFixed(3)},${s.pos[2].toFixed(3)}|${s.rot[0].toFixed(3)},${s.rot[1].toFixed(3)},${s.rot[2].toFixed(3)}`;