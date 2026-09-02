import { createRoot } from 'react-dom/client';
import App from './App';
import './styles/index.css';

// Wait for Nunito so the procedural canvas card art renders with the right face.
(async () => {
  try {
    const fd = (document as any).fonts;
    if (fd) {
      await Promise.race([
        Promise.all([fd.load('900 40px Nunito'), fd.load('italic 900 40px Nunito'), fd.load('700 20px Nunito')]),
        new Promise(r => setTimeout(r, 2500)),
      ]);
    }
  } catch { /* fall back to system fonts */ }
  // NOTE: no StrictMode — its double effect invocation would start two matches.
  createRoot(document.getElementById('root')!).render(<App />);
  document.getElementById('boot')?.remove();
})();