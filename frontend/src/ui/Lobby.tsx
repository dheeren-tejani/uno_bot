import { useEffect, useState } from 'react';
import { getReplay } from '../services/unoService';
import { useSceneStore } from '../three/store';
import type { Difficulty } from '../types';
import { Icon, UnoLogo } from './primitives';
import { RulesButton } from './RulesModal';

const DIFFS: { id: Difficulty; name: string; tag: string; desc: string }[] = [
  { id: 'easy', name: 'EASY', tag: 'CASUAL', desc: 'R3X plays on pure instinct. Mistakes included.' },
  { id: 'normal', name: 'NORMAL', tag: 'BALANCED', desc: 'Solid heuristics, sensible color calls.' },
  { id: 'hard', name: 'HARD', tag: 'RL TABULA RASA', desc: 'Card counting and a value net. No mercy.' },
];

export function Lobby({ difficulty, setDifficulty, onStart, onWatch }: {
  difficulty: Difficulty; setDifficulty: (d: Difficulty) => void;
  onStart: (d: Difficulty) => void; onWatch: (code: string) => void;
}) {
  const [code, setCode] = useState('');
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { useSceneStore.getState().showLobby(); }, []);

  const watch = async () => {
    const c = code.trim().toUpperCase();
    if (!/^[A-Z0-9]{6,8}$/.test(c)) {
      setErr('Replay codes are 6–8 letters and digits, like W7X9KM.');
      return;
    }
    setBusy(true); setErr(null);
    const d = await getReplay(c);
    setBusy(false);
    if (!d) { setErr('No replay found for that code.'); return; }
    onWatch(c);
  };

  return (
    <div className="absolute inset-0 pointer-events-none z-10 flex items-start md:items-center">
      <div className="pointer-events-auto panel-in glass rounded-3xl p-6 m-4 md:ml-10 w-full md:w-[404px] max-h-[88vh] overflow-y-auto">
        <UnoLogo className="w-36 drop-shadow-[0_6px_20px_rgba(217,58,48,0.35)]" />
        <h1 className="text-4xl font-black italic tracking-tight mt-3 leading-none">TABLE <span className="text-amber-300">DUEL</span></h1>
        <p className="text-[13px] text-slate-400 font-semibold mt-2 leading-relaxed">
          A tactile 3D UNO match against <span className="text-slate-200">R3X</span>, our reinforcement-learning card
          shark. Slam cards, call colors, steal the replay.
        </p>

        <div className="text-[10px] font-bold tracking-[0.22em] text-slate-500 mt-6 mb-2.5">SELECT RIVAL DIFFICULTY</div>
        <div className="space-y-2">
          {DIFFS.map(d => (
            <button key={d.id} onClick={() => setDifficulty(d.id)}
              className={`w-full text-left rounded-2xl border p-3.5 transition-all ${
                difficulty === d.id ? 'border-amber-300/60 bg-amber-300/10' : 'border-white/10 bg-white/[0.03] hover:border-white/25'}`}>
              <div className="flex items-center gap-3">
                <span className={`w-4 h-4 rounded-full border-2 flex items-center justify-center shrink-0 ${difficulty === d.id ? 'border-amber-300' : 'border-slate-500'}`}>
                  {difficulty === d.id && <span className="w-2 h-2 rounded-full bg-amber-300" />}
                </span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-baseline gap-2">
                    <span className="font-black tracking-wide text-sm">{d.name}</span>
                    <span className="text-[9px] font-bold tracking-[0.2em] text-slate-500">{d.tag}</span>
                  </div>
                  <div className="text-[11px] text-slate-400 font-semibold mt-0.5">{d.desc}</div>
                </div>
              </div>
            </button>
          ))}
        </div>

        <button onClick={() => onStart(difficulty)}
          className="mt-5 w-full rounded-2xl bg-amber-400 hover:bg-amber-300 active:scale-[0.98] text-slate-950 font-black tracking-[0.15em] text-sm py-4 transition-all shadow-[0_8px_30px_rgba(251,191,36,0.25)]">
          SHUFFLE UP & DEAL
        </button>
        
        <RulesButton variant="lobby" />

        <div className="mt-5 pt-5 border-t border-white/10">
          <div className="text-[10px] font-bold tracking-[0.22em] text-slate-500 mb-2">HAVE A REPLAY CODE?</div>
          <div className="flex gap-2">
            <input value={code}
              onChange={e => { setCode(e.target.value.toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 8)); setErr(null); }}
              onKeyDown={e => { if (e.key === 'Enter') watch(); }}
              placeholder="W7X9KM" maxLength={8} spellCheck={false}
              className="flex-1 min-w-0 rounded-xl bg-white/5 border border-white/15 px-3.5 py-2.5 text-sm font-bold tracking-[0.25em] placeholder:text-slate-600 focus:outline-none focus:border-amber-300/60 select-text" />
            <button onClick={watch} disabled={busy}
              className="rounded-xl border border-white/15 hover:border-white/30 px-4 py-2.5 text-xs font-extrabold tracking-widest flex items-center gap-2 disabled:opacity-50 transition-colors">
              <Icon name="play" className="w-3.5 h-3.5" />WATCH
            </button>
          </div>
          {err && <div className="text-[11px] text-rose-300 font-semibold mt-2">{err}</div>}
        </div>

        <p className="text-[10px] text-slate-500 font-semibold mt-5 leading-relaxed">
          Runs offline on the bundled mock engine — set VITE_USE_MOCK=false to wire the FastAPI bot.
        </p>
      </div>
    </div>
  );
}