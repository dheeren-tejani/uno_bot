"""
config.py
Configuration and Hyperparameters for High-Throughput Tabula Rasa UNO Self-Play Engine.
Updated for: 176-dim card-counting observation space, 62 discrete actions, and PFSP league training.
"""

from dataclasses import dataclass, field
from typing import Tuple
import os
import torch


@dataclass
class HardwareConfig:
    """Hardware acceleration and resource allocation settings."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    use_compile: bool = True          # Enable TorchInductor compilation
    compile_mode: str = "default"     # "default" for low memory footprint
    pin_memory: bool = True
    num_threads: int = 8
    vram_budget_gb: float = 14.0


@dataclass
class UnoCardConfig:
    """Card definition dimensions and action space mappings."""
    colors: Tuple[str, ...] = ("RED", "YELLOW", "GREEN", "BLUE")
    num_colors: int = 4
    cards_per_color: int = 13
    num_colored_cards: int = 52
    num_wild_types: int = 2
    total_unique_cards: int = 54      # 52 colored + 2 wild
    total_deck_size: int = 108
    
    # 62 Total Actions:
    # 00..51 : Play colored card
    # 52..55 : Play regular Wild + declare color (Red, Yellow, Green, Blue)
    # 56..59 : Play Wild Draw Four + declare color (Red, Yellow, Green, Blue)
    # 60     : DRAW (Main Phase only)
    # 61     : PASS (Post-Draw Phase only)
    num_actions: int = 62
    
    # Official-rule fidelity: Wild Draw Four is only playable when the hand
    # holds zero cards matching the active declared color.
    strict_wild_draw_four: bool = True


@dataclass
class EnvironmentConfig:
    """Vectorized Environment and Observation Space Parameters."""
    num_envs: int = 1024
    max_turns_per_game: int = 200
    
    # Observation Vector (176 dimensions):
    # - [0:54]   : Own Hand counts (normalized by /4.0) [54]
    # - [54:108] : Full Discard counts (normalized by /4.0) [54] (Card Counting)
    # - [108:162]: Top Discard Card One-Hot [54]
    # - [162:166]: Active Declared Color One-Hot [4]
    # - [166:176]: Public Context, Color-Void Clues, and Phase Flags [10]
    obs_dim: int = 176
    
    history_length: int = 8
    # Turn Event Representation: [Player (1), Action ID (1), Color Declared (1), Cards Drawn (1)] -> 4 dims
    history_feature_dim: int = 4


@dataclass
class ModelConfig:
    """Masked Recurrent Actor-Critic Architecture Parameters."""
    obs_dim: int = 176
    history_length: int = 8
    history_feature_dim: int = 4
    
    # Encoders
    obs_hidden_dim: int = 384
    history_embed_dim: int = 64
    history_gru_dim: int = 128
    
    # Core Recurrent Memory (Belief State Tracking)
    core_gru_dim: int = 512
    
    # Actor & Critic Heads
    head_hidden_dim: int = 256
    num_actions: int = 62


@dataclass
class PPOConfig:
    """Proximal Policy Optimization Hyperparameters."""
    rollout_steps: int = 256
    total_rollout_steps: int = 64 * 128  # 8,192 steps per iteration
    ppo_epochs: int = 4
    num_minibatches: int = 8
    minibatch_size: int = 2048
    
    lr: float = 3e-4
    lr_end: float = 1e-5
    max_grad_norm: float = 0.5
    
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    clip_epsilon: float = 0.2
    clip_vloss: bool = True
    value_loss_coef: float = 0.5
    
    # Slower, more controlled entropy schedule to prevent premature policy collapse
    entropy_coef_start: float = 0.01
    entropy_coef_end: float = 0.001
    entropy_anneal_steps: int = 200_000_000


@dataclass
class LeagueConfig:
    """Self-Play League and Checkpointing Configuration."""
    self_play_prob: float = 0.30        # 30% against active policy
    historical_pool_prob: float = 0.40  # 40% against prioritized historical snapshots (PFSP)
    archetype_prob: float = 0.30        # 30% against multi-archetype heuristic bots
    
    snapshot_interval_games: int = 10_000
    win_rate_snapshot_threshold: float = 0.58
    max_league_capacity: int = 40
    
    # Prioritized Fictitious Self-Play power factor
    pfsp_alpha: float = 1.5
    
    initial_elo: float = 1200.0
    elo_k_factor: float = 8.0  # Small K for per-game updates (archetypes fixed at 1300)


@dataclass
class LoggingConfig:
    """Directories, TensorBoard, and Terminal Telemetry."""
    experiment_name: str = "uno_production_50m_ppo"
    base_dir: str = "./runs"
    checkpoint_dir: str = "./checkpoints"
    league_dir: str = "./checkpoints/league"
    
    log_interval_iterations: int = 5
    save_interval_iterations: int = 25
    eval_interval_iterations: int = 50
    eval_episodes: int = 200
    
    def __post_init__(self):
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.league_dir, exist_ok=True)


@dataclass
class TrainingConfig:
    """Master Training Orchestration Container."""
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    cards: UnoCardConfig = field(default_factory=UnoCardConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    league: LeagueConfig = field(default_factory=LeagueConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    total_timesteps: int = 200_000_000
    seed: int = 42