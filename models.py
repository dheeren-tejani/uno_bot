"""
models.py
Masked Recurrent Actor-Critic neural network architecture for imperfect-information card games.
Updated with 176-dimensional card-counting observation encoder, sequence embedding GRU,
and numerical stabilization for 62 discrete masked actions.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from config import ModelConfig


def layer_init(layer: nn.Linear, std: float = 1.414, bias_const: float = 0.0) -> nn.Linear:
    """Applies orthogonal initialization with custom gain and bias constant."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MaskedCategorical:
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        self.mask = mask
        inf_mask = (1.0 - mask.float()) * -1e4
        self.masked_logits = logits + inf_mask
        self.dist = Categorical(logits=self.masked_logits)

    def sample(self) -> torch.Tensor:
        return self.dist.sample()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(action)

    def entropy(self) -> torch.Tensor:
        # FIX: use log_softmax (the actual log-probabilities), not raw logits
        log_probs = F.log_softmax(self.masked_logits, dim=-1)
        probs = self.dist.probs
        p_log_p = log_probs * probs
        p_log_p = torch.where(self.mask, p_log_p, torch.zeros_like(p_log_p))
        return -p_log_p.sum(dim=-1)


class MaskedRecurrentActorCritic(nn.Module):
    """
    Dual-head Recurrent Actor-Critic architecture with temporal move history encoder
    and 176-dim card-counting input support.
    """
    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else ModelConfig()
        
        # 1. Observation State Encoder (MLP) -> Encodes hand, discard counts, top card, and context
        self.obs_encoder = nn.Sequential(
            layer_init(nn.Linear(self.cfg.obs_dim, self.cfg.obs_hidden_dim)),
            nn.LayerNorm(self.cfg.obs_hidden_dim),
            nn.ReLU(),
            layer_init(nn.Linear(self.cfg.obs_hidden_dim, self.cfg.obs_hidden_dim)),
            nn.LayerNorm(self.cfg.obs_hidden_dim),
            nn.ReLU()
        )
        
        # 2. Public Move History Sequence Encoder (GRU)
        self.history_fc = nn.Sequential(
            layer_init(nn.Linear(self.cfg.history_feature_dim, self.cfg.history_embed_dim)),
            nn.ReLU()
        )
        self.history_gru = nn.GRU(
            input_size=self.cfg.history_embed_dim,
            hidden_size=self.cfg.history_gru_dim,
            batch_first=True
        )
        
        # 3. Core Recurrent Memory (Belief State Tracking)
        fused_dim = self.cfg.obs_hidden_dim + self.cfg.history_gru_dim
        self.fused_fc = nn.Sequential(
            layer_init(nn.Linear(fused_dim, self.cfg.core_gru_dim)),
            nn.ReLU()
        )
        self.core_gru = nn.GRU(
            input_size=self.cfg.core_gru_dim,
            hidden_size=self.cfg.core_gru_dim,
            batch_first=True
        )
        
        # 4. Policy (Actor) Head (Outputs 62 action logits)
        self.actor_head = nn.Sequential(
            layer_init(nn.Linear(self.cfg.core_gru_dim, self.cfg.head_hidden_dim)),
            nn.LayerNorm(self.cfg.head_hidden_dim),
            nn.ReLU(),
            layer_init(nn.Linear(self.cfg.head_hidden_dim, self.cfg.num_actions), std=0.01)
        )
        
        # 5. Value (Critic) Head (Expected Return in [-1.0, 1.0])
        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(self.cfg.core_gru_dim, self.cfg.head_hidden_dim)),
            nn.LayerNorm(self.cfg.head_hidden_dim),
            nn.ReLU(),
            layer_init(nn.Linear(self.cfg.head_hidden_dim, 1), std=1.0),
            # REMOVED: nn.Tanh() — let the critic learn unconstrained.
            # The rewards are in [-1, 1], so the value will naturally stay near that range.
            # If needed, clip values during inference, not during training.
        )

    def _extract_features(
        self, 
        obs: torch.Tensor, 
        history: torch.Tensor, 
        hidden_state: Optional[torch.Tensor] = None,
        done_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        is_sequential = obs.dim() == 3
        
        if not is_sequential:
            # Step-by-step rollout pass
            obs_feats = self.obs_encoder(obs)
            
            hist_emb = self.history_fc(history)
            hist_out, _ = self.history_gru(hist_emb)
            hist_feats = hist_out[:, -1, :]
            
            fused = torch.cat([obs_feats, hist_feats], dim=-1)
            fused = self.fused_fc(fused).unsqueeze(1)
            
            if hidden_state is None:
                hidden_state = torch.zeros(1, obs.size(0), self.cfg.core_gru_dim, device=obs.device)
            elif done_mask is not None:
                # Proper 3D broadcasting: (1, num_envs, 1) against (1, num_envs, hidden_dim)
                hidden_state = hidden_state * (1.0 - done_mask)
                
            gru_out, new_hidden = self.core_gru(fused, hidden_state)
            features = gru_out.squeeze(1)
            return features, new_hidden
        else:
            # Batched mini-batch training pass
            b_size, seq_len, _ = obs.shape
            obs_flat = obs.view(b_size * seq_len, -1)
            obs_feats = self.obs_encoder(obs_flat).view(b_size, seq_len, -1)
            
            hist_flat = history.view(b_size * seq_len, self.cfg.history_length, -1)
            hist_emb = self.history_fc(hist_flat)
            hist_out, _ = self.history_gru(hist_emb)
            hist_feats = hist_out[:, -1, :].view(b_size, seq_len, -1)
            
            fused = torch.cat([obs_feats, hist_feats], dim=-1)
            fused = self.fused_fc(fused)
            
            if hidden_state is None:
                hidden_state = torch.zeros(1, b_size, self.cfg.core_gru_dim, device=obs.device)
                
            gru_out, new_hidden = self.core_gru(fused, hidden_state)
            return gru_out, new_hidden

    def forward(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        history: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        done_mask: Optional[torch.Tensor] = None
    ) -> Tuple[MaskedCategorical, torch.Tensor, torch.Tensor]:
        features, new_hidden = self._extract_features(obs, history, hidden_state, done_mask)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)  # Ensure 1D shape (batch_size,)
        dist = MaskedCategorical(logits=logits, mask=mask)
        return dist, value, new_hidden

    def get_value(
        self,
        obs: torch.Tensor,
        history: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        done_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        features, _ = self._extract_features(obs, history, hidden_state, done_mask)
        return self.critic_head(features).squeeze(-1)  # Ensure 1D shape (batch_size,)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        history: torch.Tensor,
        actions: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates actions for PPO update. Handles both 2D (single-step) and
        3D (sequential chunk) inputs by flattening after the heads.
        """
        features, _ = self._extract_features(obs, history, hidden_state)
    
        if features.dim() == 3:
            # Sequential path: (batch, seq_len, hidden) -> flatten to (batch * seq_len, hidden)
            b, s, h = features.shape
            features_flat = features.reshape(b * s, h)
            logits = self.actor_head(features_flat)              # (batch * seq_len, num_actions)
            values = self.critic_head(features_flat).squeeze(-1) # (batch * seq_len,)
            mask_flat = mask.reshape(b * s, -1)                   # (batch * seq_len, num_actions)
        else:
            # Single-step path: (batch, hidden)
            logits = self.actor_head(features)
            values = self.critic_head(features).squeeze(-1)
            mask_flat = mask
    
        dist = MaskedCategorical(logits=logits, mask=mask_flat)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
    
        return log_probs, values, entropy