import { useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { SceneContents } from './three/SceneContents';
import { Lobby } from './ui/Lobby';
import { GameScreen } from './ui/GameScreen';
import { ReplayScreen } from './ui/ReplayScreen';
import type { Difficulty } from './types';

export default function App() {
  const [screen, setScreen] = useState<'lobby' | 'game' | 'replay'>('lobby');
  const [difficulty, setDifficulty] = useState<Difficulty>('normal');
  const [gameKey, setGameKey] = useState(0);
  const [replayCode, setReplayCode] = useState('');

  return (
    <div className="fixed inset-0 overflow-hidden bg-[#05070c] text-slate-100 font-sans select-none fade-in">
      <div className="absolute inset-0">
        <Canvas
          shadows="soft"
          dpr={[1, 2]}
          gl={{ antialias: true, powerPreference: 'high-performance' }}
          camera={{ fov: 45, position: [0, 6.9, 8.6], near: 0.1, far: 80 }}
          onCreated={({ gl }) => { (gl as any).toneMappingExposure = 1.18; }}
        >
          <SceneContents />
        </Canvas>
      </div>
      <div className="vignette" />

      {screen === 'lobby' && (
        <Lobby
          difficulty={difficulty} setDifficulty={setDifficulty}
          onStart={d => { setDifficulty(d); setGameKey(k => k + 1); setScreen('game'); }}
          onWatch={c => { setReplayCode(c); setScreen('replay'); }}
        />
      )}
      {screen === 'game' && (
        <GameScreen
          key={gameKey} difficulty={difficulty}
          onExit={() => setScreen('lobby')}
          onWatchReplay={c => { setReplayCode(c); setScreen('replay'); }}
          onRestart={() => setGameKey(k => k + 1)}
        />
      )}
      {screen === 'replay' && (
        <ReplayScreen code={replayCode} onExit={() => setScreen('lobby')} />
      )}
    </div>
  );
}