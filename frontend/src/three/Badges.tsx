import { Html } from '@react-three/drei';
import { cardName, colorOf, COL_HEX, COL_NAMES } from '../cards/constants';
import { DECK_X, DECK_Z, DISC_X, DISC_Z } from './layout';
import { useSceneStore } from './store';
import { Icon } from '../ui/primitives';

export function DeckBadge() {
  const count = useSceneStore(s => s.view.deck.length);
  const clickable = useSceneStore(s => s.deckClickable);
  return (
    <Html position={[DECK_X, 1.15, DECK_Z]} center distanceFactor={10} zIndexRange={[8, 0]}>
      <div className={`px-3 py-1 rounded-full text-[11px] font-extrabold tracking-[0.18em] border backdrop-blur-md pointer-events-none select-none transition-colors ${
        clickable ? 'text-amber-300 border-amber-300/50 bg-slate-950/80'
                   : 'text-slate-400 border-white/10 bg-slate-950/70'}`}>
        DECK · {count}
      </div>
    </Html>
  );
}

export function TopCardBadge() {
  const discard = useSceneStore(s => s.view.discard);
  const activeColor = useSceneStore(s => s.activeColor);
  const topId = discard.length ? discard[discard.length - 1].cid : -1;
  const wild = topId >= 52;
  const hex = topId < 0 ? '#94a3b8' : wild ? COL_HEX[activeColor] ?? '#94a3b8' : COL_HEX[colorOf(topId)];
  const label = topId < 0 ? 'DISCARD'
    : wild ? `WILD${topId === 53 ? ' +4' : ''} → ${COL_NAMES[activeColor] ?? ''}`
    : cardName(topId).toUpperCase();
  return (
    <Html position={[DISC_X, 1.26, DISC_Z]} center distanceFactor={10} zIndexRange={[8, 0]}>
      <div key={topId}
        className="pop-in px-3 py-1 rounded-full text-[11px] font-extrabold tracking-[0.18em] border backdrop-blur-md pointer-events-none select-none flex items-center gap-2 bg-slate-950/80 border-white/10 text-slate-200">
        <span className="w-2 h-2 rounded-full shrink-0" style={{ background: hex, boxShadow: `0 0 10px ${hex}` }} />
        TOP · {label}
      </div>
    </Html>
  );
}

export function BotHandLabel() {
  const bot = useSceneStore(s => s.view.bot);
  const mode = useSceneStore(s => s.mode);
  if (mode === 'replay' || bot.length === 0) return null;
  const uno = bot.length === 1;
  return (
    <Html position={[0, 2.0, -2.62]} center distanceFactor={10} zIndexRange={[8, 0]}>
      <div className={`px-3 py-1 rounded-full text-[11px] font-extrabold tracking-[0.18em] border backdrop-blur-md pointer-events-none select-none flex items-center gap-2 ${
        uno ? 'bg-rose-950/85 border-rose-400/40 text-rose-300'
            : 'bg-slate-950/75 border-white/10 text-slate-400'}`}>
        <Icon name="bot" className="w-3.5 h-3.5" />
        {uno ? 'R3X · UNO!' : `R3X · ${bot.length} CARDS`}
      </div>
    </Html>
  );
}