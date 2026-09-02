export type Difficulty = 'easy' | 'normal' | 'hard';

/** Sequential transitions the frontend plays one-by-one (400–800ms each). */
export type AnimEvent =
  | { type: 'deal'; human_cards: number[]; bot_count: number; starter_card: number; text?: string }
  | { type: 'play_card'; actor: 0 | 1; card_id: number; active_color: number; text: string }
  | { type: 'draw'; actor: 0 | 1; count: number; card_ids?: number[]; text: string }
  | { type: 'skip'; actor: 0 | 1; text: string }
  | { type: 'pass'; actor: 0 | 1; text: string }
  | { type: 'reshuffle'; deck_count: number; text: string }
  | { type: 'notice'; text: string }
  | { type: 'game_over'; winner: 0 | 1 | -1; text: string };

/** Public view of the game — bot hand & deck contents are NEVER included. */
export interface PublicGameState {
  game_id: string;
  replay_code: string;
  status: 'playing' | 'over';
  phase: 0 | 1;
  current_player: 0 | 1;
  turn_count: number;
  top_card: number;
  active_color: number;
  hand: number[];
  bot_card_count: number;
  deck_count: number;
  legal_actions: number[];
  animation_queue: AnimEvent[];
  drawn_card?: number | null;
  winner?: 0 | 1 | -1;
  duration_seconds?: number;
}

export interface ReplayFrame {
  turn: number; p0_hand: number[]; p1_hand: number[]; top_card: number;
  active_color: number; bot_value: number; deck_count: number; event: string;
  deck?: number[];   // optional extras (mock & our backend provide them)
  discard?: number[];
}

export interface ReplayPayload {
  code: string;
  difficulty: Difficulty;
  winner: 0 | 1 | -1;
  total_turns: number;
  duration_seconds: number;
  created_at: string;
  frames: ReplayFrame[];
}