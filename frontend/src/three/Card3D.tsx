import React, { memo, useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { CARD_BASE, FACE_GEO, BACK_MAT, getFaceMaterial } from './geometry';
import { Slot, slotKey, Zone } from './layout';
import { clamp, easeOutCubic, lerp } from '../cards/constants';

const _v = new THREE.Vector3();
function bez(a: THREE.Vector3, c: THREE.Vector3, b: THREE.Vector3, t: number, out: THREE.Vector3) {
  const u = 1 - t;
  return out.set(
    u * u * a.x + 2 * u * t * c.x + t * t * b.x,
    u * u * a.y + 2 * u * t * c.y + t * t * b.y,
    u * u * a.z + 2 * u * t * c.z + t * t * b.z,
  );
}

export interface Card3DProps {
  iid: number;
  cardId: number;          // -1 = face-down / unknown
  slot: Slot;
  legal: boolean;
  lift: boolean;
  delay?: number;
  onClick: (iid: number, zone: Zone, top: boolean | undefined, cardId: number) => void;
  onHover: (iid: number, on: boolean) => void;
}

export const Card3D = memo(function Card3D({ iid, cardId, slot, legal, lift, delay, onClick, onHover }: Card3DProps) {
  const grp = useRef<THREE.Group>(null!);
  const baseMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#f5f4ef', roughness: 0.55, metalness: 0.04,
    emissive: new THREE.Color('#f6c453'), emissiveIntensity: 0,
  }), []);
  const frontMat = useMemo(() => getFaceMaterial(cardId), [cardId]);

  // Per-instance material freed manually; the SHARED geometry/materials are
  // protected by dispose={null} below so no single unmount can kill them.
  useEffect(() => () => { baseMat.dispose(); }, [baseMat]);

  const anim = useRef({
    pos: new THREE.Vector3(), rot: new THREE.Euler(),
    flight: null as null | {
      from: THREE.Vector3; ctrl: THREE.Vector3; to: THREE.Vector3;
      fr: [number, number, number]; tr: [number, number, number];
      t: number; dur: number; delay: number;
    },
    wobble: 0, lastKey: '', init: false,
  });

  useFrame((st, dtRaw) => {
    const dt = Math.min(dtRaw, 0.05);
    const a = anim.current;
    const [px, py, pz] = slot.pos;
    const [rx, ry, rz] = slot.rot;

    if (!a.init) { a.init = true; a.pos.set(px, py, pz); a.rot.set(rx, ry, rz); a.lastKey = slotKey(slot); }

    const k = slotKey(slot);
    if (k !== a.lastKey) {
      a.lastKey = k;
      const to = new THREE.Vector3(px, py, pz);
      const dist = a.pos.distanceTo(to);
      if (dist > 0.02) {
        const mid = a.pos.clone().lerp(to, 0.5);
        mid.y += Math.min(0.6 + dist * 0.22, 2.0);
        a.flight = {
          from: a.pos.clone(), ctrl: mid, to,
          fr: [a.rot.x, a.rot.y, a.rot.z], tr: [rx, ry, rz],
          t: 0, dur: clamp(0.34 + dist * 0.06, 0.4, 0.9), delay: delay || 0,
        };
      } else { a.pos.set(px, py, pz); a.rot.set(rx, ry, rz); }
    }

    if (a.flight) {
      const f = a.flight;
      if (f.delay > 0) {
        f.delay -= dt;
      } else {
        f.t = Math.min(1, f.t + dt / f.dur);
        const e = easeOutCubic(f.t);
        bez(f.from, f.ctrl, f.to, e, a.pos);
        a.rot.set(lerp(f.fr[0], f.tr[0], e), lerp(f.fr[1], f.tr[1], e), lerp(f.fr[2], f.tr[2], e));
        if (f.t >= 1) { a.flight = null; a.wobble = 0.34; a.pos.copy(f.to); }
      }
    } else {
      _v.set(px, py, pz);
      if (lift) { _v.y += 0.26; if (slot.zone === 'hand' || slot.zone === 'bot') _v.z += 0.18; }
      const kk = 1 - Math.exp(-dt * 14);
      a.pos.lerp(_v, kk);
      a.rot.x += (rx - a.rot.x) * kk;
      a.rot.y += (ry - a.rot.y) * kk;
      a.rot.z += (rz - a.rot.z) * kk;
    }

    let rzW = 0;
    if (a.wobble > 0) { a.wobble -= dt; rzW = Math.sin((0.34 - a.wobble) * 26) * a.wobble * 0.5; }

    if (grp.current) {
      grp.current.position.copy(a.pos);
      grp.current.rotation.set(a.rot.x, a.rot.y, a.rot.z + rzW);
    }
    baseMat.emissiveIntensity = legal
      ? 0.26 + 0.32 * (0.5 + 0.5 * Math.sin(st.clock.elapsedTime * 3.4 + iid))
      : 0;
  });

  return (
    <group
      ref={grp}
      onPointerOver={(e) => { e.stopPropagation(); onHover(iid, true); }}
      onPointerOut={(e) => { e.stopPropagation(); onHover(iid, false); }}
      onClick={(e) => { e.stopPropagation(); onClick(iid, slot.zone, slot.top, cardId); }}
    >
      {/* dispose={null}: opts out of R3F auto-disposal so unmounting one card
          can never destroy the shared geometry/materials used by all 108. */}
      <mesh geometry={CARD_BASE} material={baseMat} castShadow receiveShadow dispose={null} />
      <mesh geometry={FACE_GEO} material={frontMat} position={[0, 0, 0.0115]} dispose={null} />
      <mesh geometry={FACE_GEO} material={BACK_MAT} position={[0, 0, -0.0115]} rotation={[0, Math.PI, 0]} dispose={null} />
    </group>
  );
});