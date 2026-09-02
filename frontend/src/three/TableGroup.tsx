import React, { memo, useRef } from 'react';
import * as THREE from 'three';
import { RoundedBox } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import { COL_HEX } from '../cards/constants';
import { DECK_X, DECK_Z, DISC_X, DISC_Z } from './layout';

export const TableGroup = memo(function TableGroup({ activeColor }: { activeColor: number }) {
  const ring = useRef<THREE.MeshBasicMaterial>(null!);
  const discRing = useRef<THREE.MeshBasicMaterial>(null!);
  const acc = useRef<THREE.PointLight>(null!);
  useFrame(st => {
    const t = st.clock.elapsedTime;
    if (ring.current) ring.current.opacity = 0.16 + 0.13 * (0.5 + 0.5 * Math.sin(t * 1.7));
    if (discRing.current) discRing.current.opacity = 0.16 + 0.14 * (0.5 + 0.5 * Math.sin(t * 2.2));
    if (acc.current) acc.current.intensity = 7 + 5 * (0.5 + 0.5 * Math.sin(t * 1.7));
  });
  const hex = COL_HEX[activeColor] ?? '#FACC15';
  return (
    <group>
      <RoundedBox args={[10.9, 0.6, 7.4]} radius={0.27} smoothness={4} position={[0, -0.31, 0]} castShadow receiveShadow>
        <meshStandardMaterial color="#5b3f28" roughness={0.62} metalness={0.06} />
      </RoundedBox>
      <RoundedBox args={[9.9, 0.42, 6.5]} radius={0.19} smoothness={4} position={[0, -0.21, 0]} receiveShadow>
        <meshStandardMaterial color="#0f172a" roughness={0.95} metalness={0.02} />
      </RoundedBox>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.62, 0]} receiveShadow>
        <circleGeometry args={[26, 48]} />
        <meshStandardMaterial color="#04060a" roughness={1} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.003, 0]}>
        <ringGeometry args={[1.85, 1.95, 80]} />
        <meshBasicMaterial color="#233752" transparent opacity={0.55} depthWrite={false} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.004, 0]}>
        <ringGeometry args={[2.3, 2.44, 96]} />
        <meshBasicMaterial ref={ring} color={hex} transparent opacity={0.2} toneMapped={false} depthWrite={false} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[DECK_X, 0.0025, DECK_Z]}>
        <circleGeometry args={[1.18, 40]} />
        <meshBasicMaterial color="#0a1322" transparent opacity={0.5} depthWrite={false} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[DISC_X, 0.0025, DISC_Z]}>
        <circleGeometry args={[1.18, 40]} />
        <meshBasicMaterial color="#0a1322" transparent opacity={0.5} depthWrite={false} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[DISC_X, 0.0045, DISC_Z]}>
        <ringGeometry args={[1.02, 1.16, 64]} />
        <meshBasicMaterial ref={discRing} color={hex} transparent opacity={0.22} toneMapped={false} depthWrite={false} />
      </mesh>
      <pointLight ref={acc} position={[0, 1.9, 0]} color={hex} intensity={9} distance={8} decay={2} />
    </group>
  );
});