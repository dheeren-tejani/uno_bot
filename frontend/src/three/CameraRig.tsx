import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame, useThree } from '@react-three/fiber';

export function CameraRig({ mode }: { mode: 'lobby' | 'game' | 'replay' }) {
  const camera = useThree(s => s.camera) as THREE.PerspectiveCamera;
  const size = useThree(s => s.size);
  const cfg = useMemo(() => {
    const a = size.width / Math.max(1, size.height);
    let pos: [number, number, number], fov = 45;
    const look = new THREE.Vector3(0, 0.35, 0.3);
    if (a < 0.8) { pos = [0, 10.4, 13.4]; fov = 52; }          // mobile portrait
    else if (a < 1.15) { pos = [0, 8.4, 10.6]; fov = 48; }
    else if (mode === 'replay') { pos = [0, 7.2, 9.2]; look.set(0, 0.65, 0.15); }
    else pos = [0, 6.9, 8.6];
    return { pos, fov, look };
  }, [size.width, size.height, mode]);

  const init = useRef(false);
  const tgt = useMemo(() => new THREE.Vector3(), []);
  useFrame((st, dt) => {
    if (!init.current) {
      init.current = true;
      camera.position.set(...cfg.pos);
      camera.fov = cfg.fov;
      camera.updateProjectionMatrix();
    }
    tgt.set(...cfg.pos);
    const k = 1 - Math.exp(-dt * 3);
    camera.position.lerp(tgt, k);
    if (mode === 'lobby') {
      camera.position.x += Math.sin(st.clock.elapsedTime * 0.22) * 0.55;
      camera.position.y += Math.cos(st.clock.elapsedTime * 0.17) * 0.28;
    }
    if (Math.abs(camera.fov - cfg.fov) > 0.05) {
      camera.fov += (cfg.fov - camera.fov) * k;
      camera.updateProjectionMatrix();
    }
    camera.lookAt(cfg.look);
  });
  return null;
}