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
    device: str = "cpu"
    use_amp: bool = True
    use_compile: bool = True
    compile_mode: str = "default"
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
    total_unique_cards: int = 54
    total_deck_size: int = 108

    num_actions: int = 62

    strict_wild_draw_four: bool = True


@dataclass
class EnvironmentConfig:
    """Vectorized Environment and Observation Space Parameters."""
    num_envs: int = 1024
    max_turns_per_game: int = 200

    obs_dim: int = 176

    history_length: int = 8
    history_feature_dim: int = 4


@dataclass
class ModelConfig:
    """Masked Recurrent Actor-Critic Architecture Parameters."""
    obs_dim: int = 176
    history_length: int = 8
    history_feature_dim: int = 4

    obs_hidden_dim: int = 384
    history_embed_dim: int = 64
    history_gru_dim: int = 128

    core_gru_dim: int = 512

    head_hidden_dim: int = 256
    num_actions: int = 62


@dataclass
class PPOConfig:
    """Proximal Policy Optimization Hyperparameters."""
    rollout_steps: int = 256
    ppo_epochs: int = 4
    num_minibatches: int = 8
    minibatch_size: int = 8192

    lr: float = 3e-4
    lr_end: float = 1e-5
    max_grad_norm: float = 0.5

    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_epsilon: float = 0.2
    clip_vloss: bool = True
    value_loss_coef: float = 0.5

    entropy_coef_start: float = 0.01
    entropy_coef_end: float = 0.001
    entropy_anneal_steps: int = 200_000_000


@dataclass
class LeagueConfig:
    """Self-Play League and Checkpointing Configuration."""
    self_play_prob: float = 0.30
    historical_pool_prob: float = 0.40
    archetype_prob: float = 0.30

    # T7: raised from 10_000 to 50_000 to avoid snapshot every iteration
    snapshot_interval_games: int = 50_000
    win_rate_snapshot_threshold: float = 0.58
    max_league_capacity: int = 40

    pfsp_alpha: float = 1.5

    initial_elo: float = 1200.0
    elo_k_factor: float = 8.0


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