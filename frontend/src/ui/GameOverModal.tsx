import { fmtTime } from '../cards/constants';
import type { Difficulty, PublicGameState } from '../types';
import { Icon } from './primitives';
import { useCopy } from './primitives';

export function GameOverModal({ gs, difficulty, onWatchReplay, onRestart }: {
  gs: PublicGameState; difficulty: Difficulty;
  onWatchReplay: (code: string) => void; onRestart: () => void;
}) {
  const [copied, copy] = useCopy();
  const winner = gs.winner ?? -1;
  return (
    <div className="absolute inset-0 z-30 flex items-center justify-center p-4 bg-slate-950/55 backdrop-blur-sm">
      <div className="glass rounded-3xl w-full max-w-md p-8 text-center modal-in">
        <div className="w-16 h-16 rounded-full bg-amber-400/15 border border-amber-300/30 flex items-center justify-center text-amber-300 mx-auto">
          <Icon name={winner === 0 ? 'trophy' : 'clock'} className="w-8 h-8" />
        </div>
        <h2 className="text-4xl font-black italic tracking-tight mt-4">
          {winner === 0 ? 'YOU WIN!' : winner === 1 ? 'R3X WINS' : 'DRAW'}
        </h2>
        <p className="text-[12.5px] text-slate-400 font-semibold mt-1.5">
          {winner === 0 ? 'Clean hand. The table is yours tonight.' :
            winner === 1 ? 'The machine out-drew you. Run it back.' :
              'Two hundred turns, no give. Honors even.'}
        </p>

        <div className="grid grid-cols-3 gap-2 mt-6">
          {[
            { icon: 'cards', label: 'TURNS', value: String(gs.turn_count) },
            { icon: 'clock', label: 'TIME', value: fmtTime(gs.duration_seconds ?? 0) },
            { icon: 'bot', label: 'RIVAL', value: difficulty.toUpperCase() },
          ].map(s => (
            <div key={s.label} className="rounded-2xl bg-white/[0.04] border border-white/10 py-3">
              <Icon name={s.icon} className="w-4 h-4 mx-auto text-slate-500" />
              <div className="text-lg font-black mt-1">{s.value}</div>
              <div className="text-[9px] font-bold tracking-[0.2em] text-slate-500">{s.label}</div>
            </div>
          ))}
        </div>

        <div className="mt-5 rounded-2xl border border-dashed border-white/25 bg-white/[0.03] px-4 py-3.5 flex items-center gap-3">
          <div className="flex-1 text-left">
            <div className="text-[9px] font-bold tracking-[0.22em] text-slate-500">REPLAY CODE</div>
            <div className="text-xl font-black tracking-[0.3em] text-amber-300">{gs.replay_code}</div>
          </div>
          <button onClick={() => copy(gs.replay_code)} className="ctl ctl-wide" title="Copy code">
            <Icon name={copied ? 'check' : 'copy'} className="w-4 h-4" />
            <span className="text-[10px] font-black tracking-widest">{copied ? 'COPIED' : 'COPY'}</span>
          </button>
        </div>

        <div className="flex gap-3 mt-6">
          <button onClick={() => onWatchReplay(gs.replay_code)}
            className="flex-1 glass rounded-2xl py-3.5 text-xs font-black tracking-widest text-slate-300 hover:text-white transition-colors flex items-center justify-center gap-2">
            <Icon name="play" className="w-3.5 h-3.5" />WATCH REPLAY
          </button>
          <button onClick={onRestart}
            className="flex-1 rounded-2xl bg-amber-400 hover:bg-amber-300 text-slate-950 py-3.5 text-xs font-black tracking-widest transition-all">
            PLAY AGAIN
          </button>
        </div>
      </div>
    </div>
  );
}