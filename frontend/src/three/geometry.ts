import * as THREE from 'three';
import { getBackTexture, getFaceTexture } from '../cards/textures';

/** Rounded extruded card body with a bevel — gives cards physical thickness. */
export const CARD_BASE: THREE.ExtrudeGeometry = (() => {
  const w = 1.2, h = 1.8, r = 0.12;
  const s = new THREE.Shape();
  s.moveTo(-w / 2 + r, -h / 2);
  s.lineTo(w / 2 - r, -h / 2);
  s.absarc(w / 2 - r, -h / 2 + r, r, -Math.PI / 2, 0, false);
  s.lineTo(w / 2, h / 2 - r);
  s.absarc(w / 2 - r, h / 2 - r, r, 0, Math.PI / 2, false);
  s.lineTo(-w / 2 + r, h / 2);
  s.absarc(-w / 2 + r, h / 2 - r, r, Math.PI / 2, Math.PI, false);
  s.lineTo(-w / 2, -h / 2 + r);
  s.absarc(-w / 2 + r, -h / 2 + r, r, Math.PI, Math.PI * 1.5, false);
  const g = new THREE.ExtrudeGeometry(s, {
    depth: 0.012, bevelEnabled: true, bevelThickness: 0.004, bevelSize: 0.003,
    bevelSegments: 1, curveSegments: 8,
  });
  g.translate(0, 0, -0.006);
  return g;
})();

export const FACE_GEO = new THREE.PlaneGeometry(1.11, 1.7);
export const BACK_MAT = new THREE.MeshStandardMaterial({ map: getBackTexture(), roughness: 0.5, metalness: 0.03 });

const faceMatCache = new Map<number, THREE.MeshStandardMaterial>();

/** SHARED, cached material per card face. Safe to hand the same material to
 *  many meshes at once, and never disposed by any single card's unmount
 *  (the dispose={null} guard in Card3D protects it). */
export function getFaceMaterial(cid: number): THREE.MeshStandardMaterial {
  if (cid < 0) return BACK_MAT;                     // anonymous / face-down → card back
  let m = faceMatCache.get(cid);
  if (!m) {
    m = new THREE.MeshStandardMaterial({
      map: getFaceTexture(cid),
      roughness: cid >= 52 ? 0.34 : 0.42,
      metalness: cid >= 52 ? 0.22 : 0.02,
    });
    faceMatCache.set(cid, m);
  }
  return m;
}