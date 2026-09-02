import { useEffect, useState } from 'react';
import { cardName } from '../cards/constants';
import { useGameMatch } from '../hooks/useGameMatch';
import type { Difficulty } from '../types';
import { ColorRing, ColorWheel, Icon, UnoLogo } from './primitives';
import { GameOverModal } from './GameOverModal';
import { RulesButton } from './RulesModal';

const diffBadge: Record<Difficulty, string> = {
  easy: 'bg-emerald-400/15 text-emerald-300 border border-emerald-300/30',
  normal: 'bg-amber-400/15 text-amber-300 border-amber-300/30',
  hard: 'bg-rose-400/15 text-rose-300 border-rose-300/30',
};

export function GameScreen({ difficulty, onExit, onWatchReplay, onRestart }: {
  difficulty: Difficulty;
  onExit: () => void;
  onWatchReplay: (code: string) => void;
  onRestart: () => void;
}) {
  const { gs, busy, botActive, toast, wild, start, act, chooseWildColor, cancelWild, playDrawn } = useGameMatch();
  const [showOver, setShowOver] = useState(false);

  useEffect(() => { start(difficulty); }, [difficulty]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (gs?.status === 'over') {
      const t = setTimeout(() => setShowOver(true), 1100);
      return () => clearTimeout(t);
    }
    setShowOver(false);
  }, [gs?.status]);

  const myTurn = !busy && !!gs && gs.status === 'playing' && gs.current_player === 0;
  const canDraw = !!(gs?.legal_actions.includes(60));
  const forcedPass = !!(gs && myTurn && gs.phase === 0 && !canDraw && gs.legal_actions.includes(61));

  return (
    <>
      {/* top bar */}
      <div className="absolute top-0 inset-x-0 p-3 flex items-start justify-between gap-3 pointer-events-none z-10">
        <div className="glass rounded-2xl px-4 py-2.5 flex items-center gap-4 pointer-events-auto">
          <UnoLogo className="w-[74px] shrink-0 hidden sm:block" />
          <div className="w-px h-9 bg-white/10 hidden sm:block" />
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-slate-800/90 border border-white/10 flex items-center justify-center text-amber-300">
              <Icon name="bot" className="w-5 h-5" />
            </div>
            <div className="leading-tight">
              <div className="text-sm font-extrabold flex items-center gap-2">
                R3X
                <span className={`text-[9px] font-black px-1.5 py-0.5 rounded ${diffBadge[difficulty]}`}>
                  {difficulty.toUpperCase()}
                </span>
              </div>
              <div className="text-[11px] text-slate-400 font-bold flex items-center gap-1 h-4">
                {botActive ? (
                  <span className="text-amber-300/90">ANALYZING<span className="dots"><i /><i /><i /></span></span>
                ) : (
                  <><Icon name="cards" className="w-3.5 h-3.5" />{gs ? `${gs.bot_card_count} cards` : '—'}</>
                )}
              </div>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2.5 pointer-events-auto">
          <div className="glass rounded-xl px-3.5 py-2 text-[11px] font-extrabold tracking-[0.18em] text-slate-300">
            TURN {Math.min(gs?.turn_count ?? 1, 200)}<span className="text-slate-500"> / 200</span>
          </div>
          <ColorRing color={gs?.active_color ?? 0} />
          <RulesButton />
          <button className="ctl" title="Leave match" onClick={onExit}><Icon name="exit" /></button>
        </div>
      </div>

      {/* toast narration (one per animation-queue step) */}
      {toast && (
        <div key={toast.id} className="toast-in absolute top-[84px] inset-x-0 z-40 flex justify-center pointer-events-none px-4">
          <div className="glass rounded-full px-5 py-2 text-[12.5px] font-bold text-slate-200 flex items-center gap-2 max-w-full">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-300 shrink-0" />
            <span className="truncate">{toast.text}</span>
          </div>
        </div>
      )}

      {/* bottom prompts */}
      {gs && gs.status === 'playing' && myTurn && gs.phase === 1 && (
        <div className="absolute bottom-5 inset-x-0 z-20 flex justify-center gap-3 pointer-events-none px-4">
          <button onClick={playDrawn}
            className="pointer-events-auto rounded-2xl bg-amber-400 hover:bg-amber-300 active:scale-[0.97] text-slate-950 font-black tracking-widest text-xs px-6 py-3.5 transition-all shadow-[0_8px_26px_rgba(251,191,36,0.3)]">
            PLAY DRAWN — {cardName(gs.drawn_card ?? 0).toUpperCase()}
          </button>
          <button onClick={() => act(61)}
            className="pointer-events-auto glass rounded-2xl px-6 py-3.5 text-xs font-black tracking-widest text-slate-300 hover:text-white border-white/15 transition-colors">
            KEEP & PASS
          </button>
        </div>
      )}
      {forcedPass && (
        <div className="absolute bottom-5 inset-x-0 z-20 flex justify-center pointer-events-none px-4">
          <button onClick={() => act(61)} className="pointer-events-auto glass-amber rounded-2xl px-6 py-3.5 text-xs font-black tracking-widest text-amber-300">
            NO PLAYS, NO DRAWS — PASS
          </button>
        </div>
      )}
      {gs && gs.status === 'playing' && myTurn && gs.phase === 0 && !forcedPass && (
        <div className="absolute bottom-4 inset-x-0 z-20 flex justify-center gap-2.5 pointer-events-none px-4">
          <div className="glass rounded-full px-4 py-2 text-[12px] font-bold text-slate-300">
            Your move — glowing cards are playable.
          </div>
          {canDraw && (
            <button onClick={() => act(60)}
              className="pointer-events-auto glass-amber rounded-full px-4 py-2 text-[12px] font-black tracking-widest text-amber-300 flex items-center gap-1.5 hover:bg-amber-300/10 transition-colors">
              <Icon name="cards" className="w-3.5 h-3.5" />DRAW
            </button>
          )}
        </div>
      )}
      {busy && botActive && !(gs && gs.status === 'over') && (
        <div className="absolute bottom-4 inset-x-0 z-20 flex justify-center pointer-events-none px-4">
          <div className="glass rounded-full px-4 py-2 text-[12px] font-bold text-slate-400">R3X is plotting…</div>
        </div>
      )}

      {busy && !botActive && gs?.status === 'playing' && (
        <div className="absolute bottom-4 inset-x-0 z-20 flex justify-center pointer-events-none px-4">
          <div className="glass rounded-full px-4 py-2 text-[12px] font-bold text-slate-400">THE TABLE IS SETTLING…</div>
        </div>
      )}

      {/* wild color selector */}
      {wild && (
        <div className="absolute inset-0 z-30 flex items-center justify-center bg-slate-950/60 backdrop-blur-sm">
          <div className="glass rounded-3xl p-8 flex flex-col items-center modal-in">
            <h3 className="text-lg font-black tracking-wide">CALL THE COLOR</h3>
            <p className="text-xs text-slate-400 font-semibold mb-4">
              {wild.base === 52 ? 'Wild' : 'Wild +4'} — pick what R3X must match
            </p>
            <ColorWheel onPick={chooseWildColor} />
            <button onClick={cancelWild}
              className="mt-4 text-[11px] font-bold tracking-widest text-slate-400 hover:text-slate-200">
              CANCEL
            </button>
          </div>
        </div>
      )}

      {/* game over */}
      {showOver && gs?.status === 'over' && (
        <GameOverModal gs={gs} difficulty={difficulty}
          onWatchReplay={onWatchReplay} onRestart={onRestart} />
      )}
    </>
  );
}