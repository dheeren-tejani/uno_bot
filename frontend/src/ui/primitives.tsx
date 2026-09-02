import React, { useCallback, useState } from 'react';
import { COL_HEX } from '../cards/constants';

export const ICONS: Record<string, React.ReactNode> = {
  bot: (<><rect x="5" y="8" width="14" height="10" rx="3" /><circle cx="9.5" cy="13" r="1.3" fill="currentColor" stroke="none" /><circle cx="14.5" cy="13" r="1.3" fill="currentColor" stroke="none" /><path d="M12 8V5" /><circle cx="12" cy="4" r="1.2" /></>),
  cards: (<><rect x="3" y="7" width="10" height="14" rx="2" /><path d="M9 3h8a2 2 0 0 1 2 2v12" /></>),
  copy: (<><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15V5a2 2 0 0 1 2-2h10" /></>),
  check: <path d="M5 13l4 4 10-10" />,
  play: <path d="M8 5l11 7-11 7z" fill="currentColor" stroke="none" />,
  pause: (<><rect x="7" y="5" width="3.5" height="14" rx="1" fill="currentColor" stroke="none" /><rect x="13.5" y="5" width="3.5" height="14" rx="1" fill="currentColor" stroke="none" /></>),
  next: (<><path d="M6 5l9 7-9 7z" fill="currentColor" stroke="none" /><rect x="17" y="5" width="2.5" height="14" rx="1" fill="currentColor" stroke="none" /></>),
  prev: (<><path d="M18 5l-9 7 9 7z" fill="currentColor" stroke="none" /><rect x="4.5" y="5" width="2.5" height="14" rx="1" fill="currentColor" stroke="none" /></>),
  eye: (<><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6z" /><circle cx="12" cy="12" r="2.5" /></>),
  trophy: (<><path d="M8 4h8v5a4 4 0 0 1-8 0z" /><path d="M8 5H5a3 3 0 0 0 3 4M16 5h3a3 3 0 0 1-3 4" /><path d="M12 13v4M8 20h8" /></>),
  clock: (<><circle cx="12" cy="12" r="8" /><path d="M12 8v4l3 2" /></>),
  exit: (<><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" /><path d="M16 17l5-5-5-5" /><path d="M21 12H9" /></>),
  back: (<><path d="M15 19l-7-7 7-7" /><path d="M8 12h13" /></>),
  info: (<><circle cx="12" cy="12" r="10" /><path d="M12 16v-4" /><path d="M12 8h.01" /></>),
  book: (<><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" /><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" /></>),
  zap: <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />,
  droplet: <path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z" />,
};

export function Icon({ name, className = 'w-4 h-4' }: { name: string; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      {ICONS[name]}
    </svg>
  );
}

export function UnoLogo({ className = '' }: { className?: string }) {
  return (
    <svg viewBox="0 0 150 100" className={className}>
      <ellipse cx="75" cy="50" rx="70" ry="42" fill="#d93a30" transform="rotate(-7 75 50)" />
      <ellipse cx="75" cy="50" rx="57" ry="30" fill="#f3c73e" transform="rotate(-7 75 50)" />
      <text x="75" y="62" textAnchor="middle" fontFamily="Nunito, sans-serif" fontWeight="900" fontStyle="italic" fontSize="40" fill="#17181c">UNO</text>
    </svg>
  );
}

export function ColorRing({ color }: { color: number }) {
  const hex = COL_HEX[color] ?? '#facc15';
  return (
    <div className="glass rounded-xl p-2 w-12 h-12 relative flex items-center justify-center">
      <span className="absolute inset-1.5 rounded-full animate-ping opacity-25" style={{ background: hex }} />
      <span className="w-6 h-6 rounded-full border-2 border-white/70" style={{ background: hex, boxShadow: `0 0 14px ${hex}` }} />
    </div>
  );
}

export function ColorWheel({ onPick }: { onPick: (c: number) => void }) {
  const segPath = (i: number) => {
    const start = -90 + i * 90 + 5, end = -90 + (i + 1) * 90 - 5;
    const p = (r: number, aDeg: number) => {
      const a = aDeg * Math.PI / 180;
      return [100 + r * Math.cos(a), 100 + r * Math.sin(a)];
    };
    const [x1, y1] = p(78, start), [x2, y2] = p(78, end);
    const [x3, y3] = p(44, end), [x4, y4] = p(44, start);
    return `M ${x1} ${y1} A 78 78 0 0 1 ${x2} ${y2} L ${x3} ${y3} A 44 44 0 0 0 ${x4} ${y4} Z`;
  };
  return (
    <svg viewBox="0 0 200 200" className="w-64 h-64 max-w-full">
      {[0, 1, 2, 3].map(i => (
        <path key={i} className="wheel-seg" d={segPath(i)} fill={COL_HEX[i]} onClick={() => onPick(i)} />
      ))}
      <circle cx="100" cy="100" r="34" fill="#0b111c" stroke="rgba(255,255,255,.12)" strokeWidth="2" />
      <text x="100" y="97" textAnchor="middle" fontFamily="Nunito, sans-serif" fontWeight="900" fontSize="13" fill="#e2e8f0">CALL</text>
      <text x="100" y="114" textAnchor="middle" fontFamily="Nunito, sans-serif" fontWeight="900" fontSize="13" fill="#e2e8f0">A COLOR</text>
    </svg>
  );
}

export function useCopy() {
  const [copied, setCopied] = useState(false);
  const copy = useCallback(async (t: string) => {
    try { await navigator.clipboard.writeText(t); }
    catch {
      const ta = document.createElement('textarea');
      ta.value = t; document.body.appendChild(ta); ta.select();
      document.execCommand('copy'); ta.remove();
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  }, []);
  return [copied, copy] as const;
}