export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://localhost:8000/api';

// Single switch between the bundled mock engine and the real FastAPI server.
export const USE_MOCK: boolean =
  ((import.meta.env.VITE_USE_MOCK as string | undefined) ?? 'true') !== 'false';