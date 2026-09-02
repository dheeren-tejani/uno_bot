"""
ppo.py
Proximal Policy Optimization (PPO) training engine with GAE, AMP, and
sequence-aware recurrent minibatch sampling for proper BPTT.
"""

from typing import Dict, Generator, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from config import PPOConfig, HardwareConfig
from models import MaskedRecurrentActorCritic


class RolloutBuffer:
    """
    Pinned trajectory storage buffer for vectorized environment rollouts.
    Supports sequence-aware minibatch sampling for recurrent BPTT.
    T3: prepare() hoists permute/reshape out of the epoch loop.
    """
    def __init__(self, num_envs: int, rollout_steps: int, obs_dim: int,
                 history_len: int, history_dim: int, num_actions: int,
                 hidden_dim: int, device: str = "cpu"):
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.total_size = num_envs * rollout_steps
        self.device = device
        self.history_len = history_len
        self.history_dim = history_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        # State & observation buffers
        self.obs = torch.zeros((rollout_steps, num_envs, obs_dim), dtype=torch.float32, device=device)
        self.masks = torch.zeros((rollout_steps, num_envs, num_actions), dtype=torch.bool, device=device)
        self.histories = torch.zeros((rollout_steps, num_envs, history_len, history_dim), dtype=torch.float32, device=device)
        self.hidden_states = torch.zeros((rollout_steps, num_envs, hidden_dim), dtype=torch.float32, device=device)

        # Action & execution buffers
        self.actions = torch.zeros((rollout_steps, num_envs), dtype=torch.int64, device=device)
        self.log_probs = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((rollout_steps, num_envs), dtype=torch.bool, device=device)

        # Computed GAE and Target Returns
        self.advantages = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)

        self.step = 0

        # T3: Pre-computed chunked buffers (set by prepare(), cleared by reset())
        self._c_obs = None
        self._c_masks = None
        self._c_hist = None
        self._c_init_h = None
        self._c_act = None
        self._c_logp = None
        self._c_adv = None
        self._c_ret = None
        self._c_val = None
        self._c_dones = None
        self._n_chunks = 0
        self._c_seq_len = 0

    def insert(self, obs, mask, history, hidden_state, action, log_prob, value, reward, done):
        self.obs[self.step].copy_(obs)
        self.masks[self.step].copy_(mask)
        self.histories[self.step].copy_(history)
        self.hidden_states[self.step].copy_(hidden_state.squeeze(0))
        self.actions[self.step].copy_(action)
        self.log_probs[self.step].copy_(log_prob)
        self.values[self.step].copy_(value.squeeze(-1))
        self.rewards[self.step].copy_(reward)
        self.dones[self.step].copy_(done)
        self.step += 1

    def compute_gae(self, next_value, next_done, gamma=0.99, gae_lambda=0.95):
        """
        Computes Generalized Advantage Estimation (GAE) backwards in time.
        Uses dones[t] (not dones[t+1]) because our convention stores
        "done after action at step t" meaning state[t+1] is a fresh start.
        """
        last_gae_lam = 0.0
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_non_terminal = 1.0 - next_done.float()
                next_values = next_value.squeeze(-1)
            else:
                next_non_terminal = 1.0 - self.dones[t].float()
                next_values = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam

        self.returns = self.advantages + self.values

    def prepare(self, seq_len: int = 8):
        """
        T3: Pre-computes chunked tensors ONCE per iteration.
        Replaces the permute/reshape that get_generator used to do 4x per iteration.
        """
        n_chunks = (self.rollout_steps // seq_len) * self.num_envs

        self._c_obs   = self.obs.permute(1, 0, 2).reshape(n_chunks, seq_len, -1)
        self._c_masks = self.masks.permute(1, 0, 2).reshape(n_chunks, seq_len, -1)
        self._c_hist  = self.histories.permute(1, 0, 2, 3).reshape(
            n_chunks, seq_len, self.history_len, -1
        )

        b_hid = self.hidden_states.permute(1, 0, 2).reshape(n_chunks, seq_len, -1)
        self._c_init_h = b_hid[:, 0, :].unsqueeze(0).contiguous()

        self._c_act   = self.actions.permute(1, 0).reshape(n_chunks, seq_len)
        self._c_logp  = self.log_probs.permute(1, 0).reshape(n_chunks, seq_len)
        self._c_adv   = self.advantages.permute(1, 0).reshape(n_chunks, seq_len)
        self._c_ret   = self.returns.permute(1, 0).reshape(n_chunks, seq_len)
        self._c_val   = self.values.permute(1, 0).reshape(n_chunks, seq_len)
        self._c_dones = self.dones.permute(1, 0).reshape(n_chunks, seq_len)

        self._n_chunks = n_chunks
        self._c_seq_len = seq_len

    def get_generator(self, minibatch_samples: int, seq_len: int = 8) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        T3+B2: Yields mini-batches from PRE-COMPUTED chunked tensors.
        Each chunk is `seq_len` consecutive timesteps from the same environment.
        Also yields dones for the B2 recurrent reset fix.
        """
        assert self._c_obs is not None, "Must call buffer.prepare() before get_generator()"
        assert seq_len == self._c_seq_len, \
            f"seq_len mismatch: prepare({self._c_seq_len}) vs get_generator({seq_len})"

        chunk_batch = max(1, minibatch_samples // seq_len)
        indices = torch.randperm(self._n_chunks, device=self.device)

        for start in range(0, self._n_chunks, chunk_batch):
            end = start + chunk_batch
            mb_idx = indices[start:end]

            yield {
                "obs":            self._c_obs[mb_idx],
                "masks":          self._c_masks[mb_idx],
                "histories":      self._c_hist[mb_idx],
                "hidden_states":  self._c_init_h[:, mb_idx, :],
                "actions":        self._c_act[mb_idx].reshape(-1),
                "old_log_probs":  self._c_logp[mb_idx].reshape(-1),
                "advantages":     self._c_adv[mb_idx].reshape(-1),
                "returns":        self._c_ret[mb_idx].reshape(-1),
                "old_values":     self._c_val[mb_idx].reshape(-1),
                "dones":          self._c_dones[mb_idx],
            }

    def reset(self):
        self.step = 0
        # T3: Free chunked buffers from previous prepare() call
        self._c_obs = None
        self._c_masks = None
        self._c_hist = None
        self._c_init_h = None
        self._c_act = None
        self._c_logp = None
        self._c_adv = None
        self._c_ret = None
        self._c_val = None
        self._c_dones = None
        self._n_chunks = 0
        self._c_seq_len = 0


class PPOTrainer:
    """
    PPO Optimization engine with mixed-precision arithmetic and entropy scheduling.
    """
    def __init__(self, model: MaskedRecurrentActorCritic, ppo_cfg: PPOConfig, hw_cfg: HardwareConfig):
        self.model = model
        self.cfg = ppo_cfg
        self.hw = hw_cfg

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.scaler = GradScaler("cuda", enabled=(self.hw.use_amp and self.hw.device == "cuda"))

        self.current_entropy_coef = self.cfg.entropy_coef_start

    def update_learning_rate(self, current_step: int, total_steps: int):
        frac = 1.0 - (current_step / max(1, total_steps))
        lr = max(self.cfg.lr_end, self.cfg.lr_end + frac * (self.cfg.lr - self.cfg.lr_end))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        ent_frac = 1.0 - min(1.0, current_step / max(1, self.cfg.entropy_anneal_steps))
        self.current_entropy_coef = self.cfg.entropy_coef_end + ent_frac * (
            self.cfg.entropy_coef_start - self.cfg.entropy_coef_end
        )

    def train_epoch(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Executes PPO optimization across mini-batches of contiguous trajectory chunks.
        T3: Calls buffer.prepare() once to hoist permute/reshape out of the epoch loop.
        """
        buffer.prepare(seq_len=8)

        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []
        clip_fractions = []

        for _ in range(self.cfg.ppo_epochs):
            for batch in buffer.get_generator(self.cfg.minibatch_size):
                obs = batch["obs"]
                masks = batch["masks"]
                histories = batch["histories"]
                hidden_states = batch["hidden_states"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["old_values"]

                # Normalize advantages across the flattened minibatch
                adv_mean = advantages.mean()
                adv_std = advantages.std() + 1e-8
                norm_advantages = (advantages - adv_mean) / adv_std

                with autocast("cuda", enabled=(self.hw.use_amp and self.hw.device == "cuda")):
                    # B2: pass dones to evaluate_actions for recurrent reset
                    log_probs, values, entropy = self.model.evaluate_actions(
                        obs=obs,
                        mask=masks,
                        history=histories,
                        actions=actions,
                        hidden_state=hidden_states,
                        dones=batch["dones"],
                    )

                    # PPO Clipped Objective
                    log_ratio = log_probs - old_log_probs
                    ratio = torch.exp(log_ratio)

                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                        clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_epsilon).float().mean().item()
                        approx_kls.append(approx_kl)
                        clip_fractions.append(clip_frac)

                    surr1 = ratio * norm_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon) * norm_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Clipped Value Loss
                    if self.cfg.clip_vloss:
                        v_loss_unclipped = (values - returns) ** 2
                        v_clipped = old_values + torch.clamp(
                            values - old_values, -self.cfg.clip_epsilon, self.cfg.clip_epsilon
                        )
                        v_loss_clipped = (v_clipped - returns) ** 2
                        value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        value_loss = 0.5 * ((values - returns) ** 2).mean()

                    entropy_loss = -entropy.mean()

                    total_loss = (
                        policy_loss
                        + self.cfg.value_loss_coef * value_loss
                        + self.current_entropy_coef * entropy_loss
                    )

                # Backward pass
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.mean().item())

        return {
            "loss/policy": float(np.mean(policy_losses)),
            "loss/value": float(np.mean(value_losses)),
            "loss/entropy": float(np.mean(entropy_losses)),
            "loss/approx_kl": float(np.mean(approx_kls)),
            "loss/clip_frac": float(np.mean(clip_fractions)),
            "entropy_coef": self.current_entropy_coef,
        }