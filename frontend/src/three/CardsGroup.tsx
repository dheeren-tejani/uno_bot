import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { buildLayout, CardInst, Zone } from './layout';
import { Card3D } from './Card3D';
import { useSceneStore } from './store';

export function CardsGroup() {
  const view = useSceneStore(s => s.view);
  const mode = useSceneStore(s => s.mode);
  const revealBot = useSceneStore(s => s.revealBot);
  const interactive = useSceneStore(s => s.interactive);
  const legalIds = useSceneStore(s => s.legalIds);
  const deckClickable = useSceneStore(s => s.deckClickable);
  const delays = useSceneStore(s => s.delays);

  const [hovered, setHovered] = useState<number | null>(null);
  const layout = useMemo(
    () => buildLayout(view, revealBot, mode === 'replay'),
    [view, revealBot, mode],
  );
  // Every physical card body in one flat list — React keys are the iids.
  const insts = useMemo<CardInst[]>(() => [
    ...view.hand, ...view.bot, ...view.deck, ...view.discard,
  ], [view]);
  const byIid = useMemo(() => new Map(insts.map(i => [i.iid, i])), [insts]);

  const onHover = useCallback((iid: number, on: boolean) => {
    setHovered(prev => (on ? iid : (prev === iid ? null : prev)));
  }, []);

  // Cursor feedback: pointer only over clickable things.
  useEffect(() => {
    let pointer = false;
    if (hovered != null) {
      const slot = layout.get(hovered);
      const inst = byIid.get(hovered);
      if (slot && inst) {
        if (slot.zone === 'hand' && interactive && inst.cid >= 0 && legalIds.has(inst.cid)) pointer = true;
        else if (slot.zone === 'deck' && slot.top && deckClickable) pointer = true;
        else if (mode !== 'game') pointer = true;
      }
    }
    document.body.style.cursor = pointer ? 'pointer' : 'default';
    return () => { document.body.style.cursor = 'default'; };
  }, [hovered, layout, byIid, interactive, legalIds, deckClickable, mode]);

  const handle = useCallback((iid: number, zone: Zone, top: boolean | undefined, cardId: number) => {
    const s = useSceneStore.getState();
    if (zone === 'deck' && top) { s.onDeckClick?.(); return; }
    if (zone === 'hand' && cardId >= 0) s.onCardClick?.(cardId, iid);
  }, []);

  return (
    <group>
      {insts.map(inst => {
        const slot = layout.get(inst.iid);
        if (!slot) return null;                       // defensive: never crash on a stray body
        const legal = interactive && inst.cid >= 0 && legalIds.has(inst.cid);
        const lift = hovered === inst.iid && (
          legal || (slot.zone === 'deck' && slot.top && deckClickable) || mode !== 'game'
        );
        return (
          <Card3D key={inst.iid} iid={inst.iid} cardId={inst.cid} slot={slot}
            legal={legal} lift={lift} delay={delays.get(inst.iid)}
            onClick={handle} onHover={onHover} />
        );
      })}
    </group>
  );
}