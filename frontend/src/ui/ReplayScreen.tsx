import { useEffect, useMemo, useState } from 'react';
import { cardName, clamp, COL_HEX } from '../cards/constants';
import { getReplay } from '../services/unoService';
import { useSceneStore } from '../three/store';
import type { ReplayPayload } from '../types';
import { Icon } from './primitives';

function ValueMeter({ v }: { v: number }) {
  const pct = Math.round(((v + 1) / 2) * 100);
  const col = pct >= 58 ? '#f87171' : pct <= 42 ? '#34d399' : '#fbbf24';
  return (
    <div className="w-56 max-w-[45vw]">
      <div className="flex justify-between text-[10px] font-bold tracking-[0.16em] text-slate-400 mb-1">
        <span>BOT WIN PROBABILITY</span>
        <span style={{ color: col }}>{pct}%</span>
      </div>
      <div className="relative h-2.5 rounded-full bg-white/10 overflow-hidden">
        <div className="absolute inset-y-0 left-0 rounded-full transition-all duration-300" style={{ width: `${pct}%`, background: col }} />
        <div className="absolute inset-y-0 left-1/2 w-px bg-white/30" />
      </div>
    </div>
  );
}

export function ReplayScreen({ code, onExit }: { code: string; onExit: () => void }) {
  const [data, setData] = useState<ReplayPayload | null | undefined>(undefined);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [reveal, setReveal] = useState(true);

  useEffect(() => {
    setPlaying(false); setIdx(0); setData(undefined);
    getReplay(code).then(d => setData(d)).catch(() => setData(null));
  }, [code]);

  const frames = useMemo(() => data?.frames ?? [], [data]);
  const f = frames[Math.min(idx, Math.max(0, frames.length - 1))];

  useEffect(() => {
    if (!data || !frames.length) return;
    const fr = frames[Math.min(idx, frames.length - 1)];
    useSceneStore.getState().applyReplayFrame(fr, reveal);
  }, [data, idx, reveal, frames]);

  useEffect(() => {
    if (!playing || !frames.length) return;
    if (idx >= frames.length - 1) { setPlaying(false); return; }
    const t = setTimeout(() => setIdx(i => Math.min(i + 1, frames.length - 1)), 1500 / speed);
    return () => clearTimeout(t);
  }, [playing, speed, idx, frames]);

  if (data === undefined) {
    return (
      <div className="absolute inset-0 z-10 flex items-center justify-center pointer-events-none">
        <div className="glass rounded-full px-6 py-3 text-xs font-bold tracking-[0.2em] text-slate-400">LOADING REPLAY…</div>
      </div>
    );
  }
  if (data === null) {
    return (
      <div className="absolute inset-0 z-10 flex items-center justify-center p-4">
        <div className="glass rounded-3xl p-8 text-center modal-in">
          <h3 className="text-xl font-black">REPLAY NOT FOUND</h3>
          <p className="text-[12px] text-slate-400 font-semibold mt-1.5">Code "{code}" isn't in the archive.</p>
          <button onClick={onExit} className="mt-5 glass rounded-2xl px-6 py-3 text-xs font-black tracking-widest">BACK TO LOBBY</button>
        </div>
      </div>
    );
  }

  const step = (d: number) => { setPlaying(false); setIdx(i => clamp(i + d, 0, frames.length - 1)); };
  const togglePlay = () => {
    if (!playing && idx >= frames.length - 1) setIdx(0);
    setPlaying(p => !p);
  };

  return (
    <>
      <div className="absolute top-0 inset-x-0 p-3 flex items-start justify-between gap-3 pointer-events-none z-10">
        <div className="glass rounded-2xl px-3 py-2 flex items-center gap-3 pointer-events-auto">
          <button className="ctl" onClick={onExit} title="Back to lobby"><Icon name="back" /></button>
          <div className="text-sm font-black tracking-[0.18em] hidden sm:block">REPLAY STUDIO</div>
          <span className="px-2 py-1 rounded-lg bg-white/5 border border-white/10 text-[11px] font-bold tracking-[0.2em] text-amber-300">{code}</span>
          {f && (
            <div className="flex items-center gap-2 ml-1">
              <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: COL_HEX[f.active_color] }} />
              <span className="text-[11px] font-bold text-slate-300 hidden md:block">{cardName(f.top_card)}</span>
            </div>
          )}
        </div>
        <div className="glass rounded-2xl px-4 py-2.5 flex items-center gap-5 pointer-events-auto">
          <ValueMeter v={f?.bot_value ?? 0} />
          <button onClick={() => setReveal(r => !r)} title="Reveal bot hand"
            className={`ctl ctl-wide ${reveal ? 'text-amber-300' : 'text-slate-500'}`}>
            <Icon name="eye" className="w-4 h-4" />
            <span className="text-[9px] font-black tracking-widest hidden sm:inline">{reveal ? 'BOT SHOWN' : 'BOT HIDDEN'}</span>
          </button>
        </div>
      </div>

      <div className="absolute bottom-0 inset-x-0 p-3 pointer-events-none z-10">
        <div className="glass rounded-2xl px-4 py-3 pointer-events-auto max-w-4xl mx-auto">
          <div className="flex items-center gap-3 flex-wrap">
            <div className="flex items-center gap-2">
              <button className="ctl" onClick={() => step(-1)} title="Step back"><Icon name="prev" /></button>
              <button className="ctl ctl-gold" onClick={togglePlay} title="Play / pause">
                <Icon name={playing ? 'pause' : 'play'} className="w-5 h-5" />
              </button>
              <button className="ctl" onClick={() => step(1)} title="Step forward"><Icon name="next" /></button>
            </div>
            <div className="w-px h-8 bg-white/10 hidden sm:block" />
            <div className="flex items-center gap-1">
              {[0.5, 1, 2].map(s => (
                <button key={s} onClick={() => setSpeed(s)}
                  className={`px-2.5 py-1.5 rounded-lg text-xs font-black transition-colors ${speed === s ? 'bg-amber-400 text-slate-950' : 'text-slate-400 hover:text-slate-200'}`}>
                  {s}×
                </button>
              ))}
            </div>
            <div className="flex-1 min-w-[180px] text-right">
              <div className="text-[10px] font-bold tracking-[0.18em] text-slate-500">
                FRAME {Math.min(idx + 1, frames.length)} / {frames.length} — TURN {f?.turn ?? 0}
              </div>
              <div className="text-[12px] text-slate-300 font-semibold truncate">{f?.event ?? '—'}</div>
            </div>
          </div>
          <input type="range" min={0} max={Math.max(0, frames.length - 1)}
            value={Math.min(idx, frames.length - 1)}
            onChange={e => { setPlaying(false); setIdx(+e.target.value); }}
            className="uno-range mt-2 w-full" />
        </div>
      </div>
    </>
  );
}   