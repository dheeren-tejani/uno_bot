export const COL_HEX = ['#EF4444', '#FACC15', '#22C55E', '#3B82F6'];
export const COL_DARK = ['#9f1d1d', '#a16207', '#166534', '#1e40af'];
export const COL_NAMES = ['RED', 'YELLOW', 'GREEN', 'BLUE'];
export const ALL_IDS: number[] = Array.from({ length: 54 }, (_, i) => i);
/** Placeholder ids for face-down cards whose identity is hidden by the server. */
export const ANON_IDS: number[] = Array.from({ length: 54 }, (_, i) => 100 + i);

export const colorOf = (id: number) => Math.floor(id / 13);
export const typeOf = (id: number) => id % 13;
export const isAnon = (id: number) => id >= 100;

export function cardName(id: number): string {
  if (id === 52) return 'Wild';
  if (id === 53) return 'Wild +4';
  const c = ['Red', 'Yellow', 'Green', 'Blue'][Math.floor(id / 13)];
  const t = id % 13;
  return `${c} ${t <= 9 ? t : ['Skip', 'Reverse', '+2'][t - 10]}`;
}

export const sleep = (ms: number) => new Promise<void>(r => setTimeout(r, ms));
export const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
export const easeOutCubic = (t: number) => 1 - Math.pow(1 - t, 3);
export const fmtTime = (seconds: number) => {
  const s = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
};

export function shuffle<T>(a: T[]): T[] {
  const r = a.slice();
  for (let i = r.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [r[i], r[j]] = [r[j], r[i]];
  }
  return r;
}

/** Deterministic ±12° landing jitter, shared by layout and teleport targets. */
export const JIT: { x: number; y: number; z: number }[] = (() => {
  let seed = 20240707;
  const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  return ALL_IDS.map(() => ({ x: (rnd() - 0.5) * 0.05, y: (rnd() - 0.5) * 0.05, z: (rnd() - 0.5) * 0.42 }));
})();
/** Deterministic jitter for any id (card face or growing instance id):
 *  wraps into the 54-entry JIT table safely, any magnitude, any sign. */
export const jitOf = (id: number) => JIT[((id % 54) + 54) % 54];