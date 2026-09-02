import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { CameraRig } from './CameraRig';
import { TableGroup } from './TableGroup';
import { CardsGroup } from './CardsGroup';
import { DeckBadge, TopCardBadge, BotHandLabel } from './Badges';
import { useSceneStore } from './store';

export function SceneContents() {
  const mode = useSceneStore(s => s.mode);
  const activeColor = useSceneStore(s => s.activeColor);
  const spot = useRef<THREE.SpotLight>(null!);
  useEffect(() => {
    if (spot.current) {
      spot.current.target.position.set(0, 0, 0.4);
      spot.current.target.updateMatrixWorld();
    }
  }, []);
  return (
    <>
      <color attach="background" args={['#05070c']} />
      <fog attach="fog" args={['#05070c', 15, 34]} />
      <CameraRig mode={mode} />
      <ambientLight intensity={0.55} />
      <hemisphereLight args={['#3b4d6b', '#05070c', 0.45]} />
      <directionalLight position={[6, 5, 10]} intensity={0.8} color="#aac6ff" />
      <spotLight
        ref={spot} position={[0, 7.5, 0.9]} angle={0.62} penumbra={0.85}
        intensity={170} color="#ffe9c8" decay={2} castShadow
        shadow-mapSize={[2048, 2048]} shadow-bias={-0.0002} shadow-normalBias={0.02}
      />
      <TableGroup activeColor={activeColor} />
      <CardsGroup />
      <DeckBadge />
      <TopCardBadge />
      <BotHandLabel />
    </>
  );
}