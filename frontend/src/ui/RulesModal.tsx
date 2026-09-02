import { useState } from 'react';
import { Icon } from './primitives';

interface RuleItem { text: string; hot?: boolean }
interface RuleSection { icon: string; title: string; items: RuleItem[] }

/** Single source of truth for the house rules this engine enforces.
 *  If you ever change UNO_INITIAL_HAND_SIZE / max turns on the backend,
 *  update this array to match. */
const SECTIONS: RuleSection[] = [
  {
    icon: 'cards',
    title: 'THE DECK & THE DEAL',
    items: [
      { text: 'Official 108-card deck — every color has one 0 and two of each 1–9, Skip, Reverse and +2, plus four Wilds and four Wild +4s. Duplicates are real: there are two Red 7s.' },
      { text: 'Both duelists start with 10 cards — this bot trained on 10-card duels, not the casual 7.', hot: true },
      { text: 'The starting player is chosen at random — R3X may lead.' },
      { text: 'The opening discard is always a plain number card.' },
    ],
  },
  {
    icon: 'play',
    title: 'PLAYING A CARD',
    items: [
      { text: 'A card matches if it shares the active color, or the same symbol/number as the top discard. On a Wild top, only the declared color counts.' },
      { text: 'Plain Wilds are always playable — Wild +4 has its own restriction below.' },
      { text: 'Exactly one action per turn: play a card, or draw. Only the cards with glowing gold edges in your hand are legal right now.' },
      { text: 'No jump-ins and no playing out of turn — ever.' },
    ],
  },
  {
    icon: 'zap',
    title: 'ACTION CARDS IN A DUEL',
    items: [
      { text: 'Skip: your opponent loses their turn — you play again.' },
      { text: 'Reverse: in a two-player duel it works exactly like a Skip — you play again.' },
      { text: '+2 and Wild +4: the victim draws and also loses their turn.', hot: true },
      { text: 'No stacking — you cannot answer a +2 with another +2, or pile +4s. The penalty resolves and the turn moves on.', hot: true },
    ],
  },
  {
    icon: 'droplet',
    title: 'WILDS & COLOR CALLS',
    items: [
      { text: 'Wild: play it any time and pick the new active color.' },
      { text: 'Wild +4 is strict — only legal when you hold no cards of the current active color. It simply won\'t glow otherwise.', hot: true },
      { text: 'One exception: a Wild +4 you just drew can always be played during the post-draw decision, even if you hold the active color.' },
      { text: 'After a wild, matching is against the declared color — the color picker opens when you click one.' },
    ],
  },
  {
    icon: 'copy',
    title: 'DRAWING',
    items: [
      { text: 'DRAW takes exactly one card — there is no "draw until you can play".', hot: true },
      { text: 'If the drawn card is playable, you get the Post-Draw Decision: play that card, or keep it and pass. You can\'t play anything else from your hand in that phase.' },
      { text: 'If the drawn card isn\'t playable, your turn simply ends.' },
      { text: 'When the draw pile empties, all discards except the top card are shuffled back in. If nothing can be drawn, you pass.' },
    ],
  },
  {
    icon: 'trophy',
    title: 'WINNING & LIMITS',
    items: [
      { text: 'Empty your hand to win — the instant it hits zero.' },
      { text: 'No UNO call, no penalty for forgetting to announce. R3X never announces either — the "R3X · UNO!" badge is a courtesy warning for you.', hot: true },
      { text: '200-turn limit: at the limit, the player with fewer cards wins; an exact tie is a draw.' },
      { text: 'Every match earns an 8-character replay code you can rewatch frame-by-frame in the Replay Studio.' },
    ],
  },
];

function RulesModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="absolute inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/65 backdrop-blur-sm"
         onClick={onClose}>
      <div className="glass rounded-3xl w-full max-w-lg max-h-[86vh] flex flex-col modal-in"
           onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between px-6 pt-6 pb-4 border-b border-white/10 shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-amber-400/15 border border-amber-300/30 flex items-center justify-center text-amber-300">
              <Icon name="book" className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-lg font-black tracking-wide leading-tight">TABLE RULES</h3>
              <p className="text-[9px] font-bold tracking-[0.22em] text-slate-500">EXACTLY WHAT THIS BOT ENFORCES</p>
            </div>
          </div>
          <button className="ctl shrink-0" title="Close" onClick={onClose}><Icon name="exit" /></button>
        </div>

        <div className="overflow-y-auto px-6 py-5 space-y-6">
          {SECTIONS.map(sec => (
            <section key={sec.title}>
              <h4 className="flex items-center gap-2 text-[10px] font-black tracking-[0.22em] text-amber-300/90 mb-2.5">
                <Icon name={sec.icon} className="w-3.5 h-3.5" />{sec.title}
              </h4>
              <ul className="space-y-2">
                {sec.items.map((it, i) => (
                  <li key={i} className="flex gap-2.5 text-[12.5px] font-semibold leading-relaxed">
                    <span className="mt-[7px] w-1.5 h-1.5 rounded-full shrink-0"
                          style={{ background: it.hot ? '#fbbf24' : '#475569' }} />
                    <span className={it.hot ? 'text-slate-200' : 'text-slate-400'}>
                      {it.text}
                      {it.hot && (
                        <span className="ml-2 inline-block whitespace-nowrap align-middle text-[8px] font-black tracking-[0.15em] px-1.5 py-0.5 rounded bg-amber-400/15 text-amber-300 border border-amber-300/25">
                          ≠ HOUSE RULES
                        </span>
                      )}
                    </span>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>

        <div className="px-6 py-4 border-t border-white/10 shrink-0 flex items-center gap-4">
          <p className="flex-1 text-[10px] font-semibold text-slate-500 leading-relaxed">
            These mirror the trained policy's environment exactly and can't be customized mid-match.
            Amber tags mark where common house rules differ.
          </p>
          <button onClick={onClose}
            className="rounded-2xl bg-amber-400 hover:bg-amber-300 text-slate-950 px-6 py-3 text-xs font-black tracking-widest transition-all shrink-0">
            GOT IT
          </button>
        </div>
      </div>
    </div>
  );
}

/** Self-contained: renders the button AND owns the modal state. */
export function RulesButton({ variant = 'topbar' }: { variant?: 'topbar' | 'lobby' }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      {variant === 'lobby' ? (
        <button onClick={() => setOpen(true)}
          className="mt-3 w-full rounded-2xl border border-white/10 bg-white/[0.03] hover:border-amber-300/40 py-3 text-[11px] font-black tracking-[0.2em] text-slate-300 hover:text-amber-200 flex items-center justify-center gap-2 transition-all">
          <Icon name="book" className="w-4 h-4" />HOW THIS TABLE PLAYS
        </button>
      ) : (
        <button className="ctl" title="How this table plays" onClick={() => setOpen(true)}>
          <Icon name="info" />
        </button>
      )}
      {open && <RulesModal onClose={() => setOpen(false)} />}
    </>
  );
}