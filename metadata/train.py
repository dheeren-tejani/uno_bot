"""
train.py
Master training orchestration engine for Tabula Rasa UNO Bot.
Features PFSP Matchmaking, randomized starting player, AMP, TorchInductor,
pinned memory transfers, and proper done handling for auto-resetting envs.
"""

import os
import time
import gc
import argparse
from typing import Optional
from collections import deque
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.table import Table

from config import TrainingConfig
from uno_engine import VectorizedUnoEnv
from models import MaskedRecurrentActorCritic
from ppo import RolloutBuffer, PPOTrainer
from league import LeagueManager, HeuristicArchetypes


def get_vram_usage_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0.0


def main(cfg: Optional[TrainingConfig] = None):
    parser = argparse.ArgumentParser(description="Production UNO RL Training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training")
    args = parser.parse_args()

    cfg = cfg if cfg is not None else TrainingConfig()

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    device = torch.device(cfg.hardware.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    console = Console()
    writer = SummaryWriter(
        log_dir=os.path.join(cfg.logging.base_dir, cfg.logging.experiment_name)
    )

    console.print(f"[bold green]Starting Production Tabula Rasa UNO Training on: {device}[/bold green]")
    console.print(f"[cyan]Target: {cfg.total_timesteps:,} steps | "
                  f"Vectorized Envs: {cfg.env.num_envs} | Obs Dim: {cfg.env.obs_dim}[/cyan]\n")

    # === 1. INITIALIZE COMPONENTS ===
    env = VectorizedUnoEnv(env_cfg=cfg.env, card_cfg=cfg.cards, seed=cfg.seed)
    agent_net = MaskedRecurrentActorCritic(cfg.model).to(device)

    if cfg.hardware.use_compile and hasattr(torch, "compile"):
        console.print("[bold cyan]Compiling Actor-Critic Network with TorchInductor...[/bold cyan]")
        try:
            agent_net = torch.compile(agent_net, mode=cfg.hardware.compile_mode)
        except Exception as e:
            console.print(f"[bold red]torch.compile failed, using Eager mode: {e}[/bold red]")

    ppo_trainer = PPOTrainer(model=agent_net, ppo_cfg=cfg.ppo, hw_cfg=cfg.hardware)
    league = LeagueManager(league_cfg=cfg.league, model_cfg=cfg.model, hw_cfg=cfg.hardware)

    total_timesteps = 0
    total_games_completed = 0
    iteration = 0

    # === OPTIONAL RESUME ===
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        target_model = getattr(agent_net, "_orig_mod", agent_net)
        target_model.load_state_dict(league._clean_state_dict(ckpt["model_state_dict"]))
        ppo_trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        ppo_trainer.scaler.load_state_dict(ckpt["scaler_state_dict"])
        total_timesteps = int(ckpt.get("total_timesteps", 0))
        total_games_completed = int(ckpt.get("total_games", 0))
        iteration = int(ckpt.get("iteration", 0))
        league.main_agent_elo = float(ckpt.get("elo", league.main_agent_elo))
        league.last_snapshot_games = total_games_completed
        restored = league.restore_pool_from_disk(cfg.logging.league_dir)
        console.print(
            f"[bold green]Resumed from {args.resume} | iter {iteration} | "
            f"{total_timesteps:,} steps | {total_games_completed:,} games | "
            f"pool restored: {restored}[/bold green]"
        )

    # === BUFFER ===
    buffer = RolloutBuffer(
        num_envs=cfg.env.num_envs,
        rollout_steps=cfg.ppo.rollout_steps,
        obs_dim=cfg.env.obs_dim,
        history_len=cfg.env.history_length,
        history_dim=cfg.env.history_feature_dim,
        num_actions=cfg.cards.num_actions,
        hidden_dim=cfg.model.core_gru_dim,
        device=cfg.hardware.device
    )

    # === PINNED MEMORY BUFFERS (async CPU<->GPU) ===
    obs_pinned = torch.empty(cfg.env.num_envs, cfg.env.obs_dim,
                             dtype=torch.float32, pin_memory=True)
    mask_pinned = torch.empty(cfg.env.num_envs, cfg.cards.num_actions,
                              dtype=torch.bool, pin_memory=True)
    hist_pinned = torch.empty(cfg.env.num_envs, cfg.env.history_length,
                              cfg.env.history_feature_dim,
                              dtype=torch.float32, pin_memory=True)
    opp_obs_pinned = torch.empty(cfg.env.num_envs, cfg.env.obs_dim,
                                 dtype=torch.float32, pin_memory=True)
    opp_mask_pinned = torch.empty(cfg.env.num_envs, cfg.cards.num_actions,
                                  dtype=torch.bool, pin_memory=True)
    opp_hist_pinned = torch.empty(cfg.env.num_envs, cfg.env.history_length,
                                  cfg.env.history_feature_dim,
                                  dtype=torch.float32, pin_memory=True)
    act_pinned = torch.empty(cfg.env.num_envs, dtype=torch.int64, pin_memory=True)
    opp_act_pinned = torch.empty(cfg.env.num_envs, dtype=torch.int64, pin_memory=True)
    # T4: pinned buffers for rewards/dones H2D
    reward_pinned = torch.empty(cfg.env.num_envs, dtype=torch.float32, pin_memory=True)
    done_pinned = torch.empty(cfg.env.num_envs, dtype=torch.bool, pin_memory=True)

    # === INITIAL RESET ===
    obs_np, mask_np, hist_np = env.reset()
    obs_pinned.copy_(torch.from_numpy(obs_np))
    mask_pinned.copy_(torch.from_numpy(mask_np))
    hist_pinned.copy_(torch.from_numpy(hist_np))
    obs_t = obs_pinned.to(device, non_blocking=True)
    mask_t = mask_pinned.to(device, non_blocking=True)
    hist_t = hist_pinned.to(device, non_blocking=True)

    agent_hidden = torch.zeros(1, cfg.env.num_envs, cfg.model.core_gru_dim, device=device)
    opp_hidden = torch.zeros(1, cfg.env.num_envs, cfg.model.core_gru_dim, device=device)

    # B5: only add iter-0 snapshot on fresh start (not resume)
    if not args.resume:
        league.add_snapshot(agent_net, iteration=0, save_dir=cfg.logging.league_dir)

    # B8: use deque with maxlen instead of unbounded list
    rolling_rewards = deque(maxlen=1000)
    rolling_lengths = deque(maxlen=1000)

    # ==================================================================
    # MASTER TRAINING LOOP
    # ==================================================================
    while total_timesteps < cfg.total_timesteps:
        iteration += 1
        iter_start_time = time.time()
        iter_env_steps = 0
        buffer.reset()

        # Sample opponent for this iteration
        opp_category, active_archetype = league.sample_matchup()
        opp_net, opp_pool_idx = league.get_opponent_model(agent_net, opp_category)

        # Reset opponent hidden state when opponent changes
        opp_hidden = torch.zeros(1, cfg.env.num_envs,
                                 cfg.model.core_gru_dim, device=device)

        agent_net.eval()
        if opp_net is not None and opp_net is not agent_net:
            opp_net.eval()

        # ==============================================================
        # 1. ROLLOUT COLLECTION
        # ==============================================================
        for step in range(cfg.ppo.rollout_steps):
            total_timesteps += cfg.env.num_envs

            # --- Per-step accumulators ---
            step_dones = np.zeros(cfg.env.num_envs, dtype=bool)
            step_rewards = np.zeros(cfg.env.num_envs, dtype=np.float32)
            step_infos = {
                "winners": -np.ones(cfg.env.num_envs, dtype=np.int8),
                "turn_counts": np.zeros(cfg.env.num_envs, dtype=np.int32)
            }
            game_outcomes = []

            # ==========================================================
            # 1a. RESOLVE OPPONENT TURNS BEFORE AGENT ACTS
            # ==========================================================
            opp_safety = 0
            while np.any(env.current_player == 1) and opp_safety < 500:
                opp_safety += 1
                active_opp_mask = env.current_player == 1
                opp_env_indices = np.where(active_opp_mask)[0]
                opp_actions = np.zeros(cfg.env.num_envs, dtype=np.int64)

                if opp_category == "archetype" and active_archetype is not None:
                    # T5: only build mask (archetypes don't need obs/history for forward)
                    fwd_mask = env.get_action_masks()
                    for e in opp_env_indices:
                        opp_actions[e] = HeuristicArchetypes.get_action(
                            active_archetype, env, e, 1, fwd_mask[e]
                        )
                else:
                    # T5: build all three manually (not inside env.step)
                    fwd_obs = env.get_observations()
                    fwd_mask = env.get_action_masks()
                    fwd_hist = env._get_histories()

                    # T2: always forward the FULL num_envs batch (fixed shape, no recompiles)
                    opp_obs_pinned.copy_(torch.from_numpy(fwd_obs))
                    opp_mask_pinned.copy_(torch.from_numpy(fwd_mask))
                    opp_hist_pinned.copy_(torch.from_numpy(fwd_hist))
                    opp_obs_t = opp_obs_pinned.to(device, non_blocking=True)
                    opp_mask_t = opp_mask_pinned.to(device, non_blocking=True)
                    opp_hist_t = opp_hist_pinned.to(device, non_blocking=True)

                    with torch.no_grad():
                        dist_opp, _, opp_hidden_full = opp_net(
                            obs=opp_obs_t, mask=opp_mask_t,
                            history=opp_hist_t, hidden_state=opp_hidden
                        )
                        opp_act = dist_opp.sample()

                    # Write back hidden ONLY for envs that actually moved
                    idx_t = torch.from_numpy(opp_env_indices).to(device, non_blocking=True)
                    opp_hidden[:, idx_t, :] = opp_hidden_full[:, idx_t, :]

                    # T4: async D2H to pinned memory + single sync
                    opp_act_pinned.copy_(opp_act, non_blocking=True)
                    torch.cuda.synchronize()
                    sampled = opp_act_pinned.numpy()
                    opp_actions[opp_env_indices] = sampled[opp_env_indices]

                # T5: skip output building inside env.step
                _, _, _, opp_rewards, opp_dones, opp_infos = env.step(
                    opp_actions, env_mask=active_opp_mask, build_outputs=False
                )
                iter_env_steps += int(np.count_nonzero(active_opp_mask))

                step_rewards = step_rewards + (-opp_rewards)
                step_dones = step_dones | opp_dones
                for k in ["winners", "turn_counts"]:
                    if k in opp_infos:
                        step_infos[k] = np.where(
                            opp_dones, opp_infos[k], step_infos[k]
                        )
                for e in range(cfg.env.num_envs):
                    if opp_dones[e]:
                        game_outcomes.append((
                            int(opp_infos["winners"][e]),
                            int(opp_infos["turn_counts"][e])
                        ))
                opp_done_t = torch.from_numpy(opp_dones).to(
                    device, non_blocking=True
                ).float().view(1, -1, 1)
                opp_hidden = opp_hidden * (1.0 - opp_done_t)

            # ==========================================================
            # 1b. BUILD OBSERVATIONS FOR AGENT
            #     (T5: build manually after opponent loop, not inside env.step)
            # ==========================================================
            obs_np = env.get_observations()
            mask_np = env.get_action_masks()
            hist_np = env._get_histories()
            obs_pinned.copy_(torch.from_numpy(obs_np))
            mask_pinned.copy_(torch.from_numpy(mask_np))
            hist_pinned.copy_(torch.from_numpy(hist_np))
            obs_t = obs_pinned.to(device, non_blocking=True)
            mask_t = mask_pinned.to(device, non_blocking=True)
            hist_t = hist_pinned.to(device, non_blocking=True)

            # Reset agent_hidden for envs where games ended during
            # pre-agent opponent resolution
            pre_agent_done_t = torch.from_numpy(step_dones).to(
                device, non_blocking=True
            ).float().view(1, -1, 1)
            agent_hidden = agent_hidden * (1.0 - pre_agent_done_t)

            # ==========================================================
            # 1c. AGENT TAKES ACTION
            # ==========================================================
            with torch.no_grad():
                dist_main, val_main, next_agent_hidden = agent_net(
                    obs=obs_t, mask=mask_t, history=hist_t,
                    hidden_state=agent_hidden
                )
                act_main = dist_main.sample()
                logp_main = dist_main.log_prob(act_main)

            # T4: async D2H to pinned memory + single sync (was: .cpu() sync + double copy)
            act_pinned.copy_(act_main, non_blocking=True)
            torch.cuda.synchronize()
            act_main_np = act_pinned.numpy()

            # Step ONLY envs where it's the agent's turn
            agent_active = (~step_dones) & (env.current_player == 0)
            if np.any(agent_active):
                # T5: skip output building (loop 1d or post-loop will build manually)
                _, _, _, agent_rewards, agent_dones, agent_infos = env.step(
                    act_main_np, env_mask=agent_active, build_outputs=False
                )
                iter_env_steps += int(np.count_nonzero(agent_active))
                step_rewards = step_rewards + agent_rewards
                step_dones = step_dones | agent_dones
                for k in ["winners", "turn_counts"]:
                    if k in agent_infos:
                        step_infos[k] = np.where(
                            agent_dones, agent_infos[k], step_infos[k]
                        )
                for e in range(cfg.env.num_envs):
                    if agent_dones[e]:
                        game_outcomes.append((
                            int(agent_infos["winners"][e]),
                            int(agent_infos["turn_counts"][e])
                        ))

            # ==========================================================
            # 1d. RESOLVE OPPONENT TURNS AFTER AGENT ACTS
            # ==========================================================
            while np.any((env.current_player == 1) & (~step_dones)):
                active_opp_mask = (env.current_player == 1) & (~step_dones)
                opp_env_indices = np.where(active_opp_mask)[0]
                opp_actions = np.zeros(cfg.env.num_envs, dtype=np.int64)

                if opp_category == "archetype" and active_archetype is not None:
                    # T5: only build mask
                    fwd_mask = env.get_action_masks()
                    for e in opp_env_indices:
                        opp_actions[e] = HeuristicArchetypes.get_action(
                            active_archetype, env, e, 1, fwd_mask[e]
                        )
                else:
                    # T5: build all three manually
                    fwd_obs = env.get_observations()
                    fwd_mask = env.get_action_masks()
                    fwd_hist = env._get_histories()

                    # T2: full-batch forward
                    opp_obs_pinned.copy_(torch.from_numpy(fwd_obs))
                    opp_mask_pinned.copy_(torch.from_numpy(fwd_mask))
                    opp_hist_pinned.copy_(torch.from_numpy(fwd_hist))
                    opp_obs_t = opp_obs_pinned.to(device, non_blocking=True)
                    opp_mask_t = opp_mask_pinned.to(device, non_blocking=True)
                    opp_hist_t = opp_hist_pinned.to(device, non_blocking=True)

                    with torch.no_grad():
                        dist_opp, _, opp_hidden_full = opp_net(
                            obs=opp_obs_t, mask=opp_mask_t,
                            history=opp_hist_t, hidden_state=opp_hidden
                        )
                        opp_act = dist_opp.sample()

                    idx_t = torch.from_numpy(opp_env_indices).to(device, non_blocking=True)
                    opp_hidden[:, idx_t, :] = opp_hidden_full[:, idx_t, :]

                    opp_act_pinned.copy_(opp_act, non_blocking=True)
                    torch.cuda.synchronize()
                    sampled = opp_act_pinned.numpy()
                    opp_actions[opp_env_indices] = sampled[opp_env_indices]

                # T5: skip output building
                _, _, _, opp_rewards, opp_dones, opp_infos = env.step(
                    opp_actions, env_mask=active_opp_mask, build_outputs=False
                )
                iter_env_steps += int(np.count_nonzero(active_opp_mask))

                step_rewards = step_rewards + (-opp_rewards)
                step_dones = step_dones | opp_dones
                for k in ["winners", "turn_counts"]:
                    if k in opp_infos:
                        step_infos[k] = np.where(
                            opp_dones, opp_infos[k], step_infos[k]
                        )
                for e in range(cfg.env.num_envs):
                    if opp_dones[e]:
                        game_outcomes.append((
                            int(opp_infos["winners"][e]),
                            int(opp_infos["turn_counts"][e])
                        ))
                opp_done_t = torch.from_numpy(opp_dones).to(
                    device, non_blocking=True
                ).float().view(1, -1, 1)
                opp_hidden = opp_hidden * (1.0 - opp_done_t)

            # ==========================================================
            # 1e. RECORD ALL GAME OUTCOMES
            # ==========================================================
            for winner, turn_count in game_outcomes:
                total_games_completed += 1
                league.record_match_outcome(
                    winner=winner, opp_pool_idx=opp_pool_idx,
                    arch_name=active_archetype
                )
                rolling_lengths.append(turn_count)
                rolling_rewards.append(
                    1.0 if winner == 0 else (-1.0 if winner == 1 else 0.0)
                )

            # ==========================================================
            # 1f. STORE TRANSITION IN BUFFER
            #     (T5: build outputs manually for next step)
            # ==========================================================
            # T4: use pinned buffers for truly async H2D
            reward_pinned.copy_(torch.from_numpy(step_rewards))
            done_pinned.copy_(torch.from_numpy(step_dones))
            rewards_t = reward_pinned.to(device, non_blocking=True)
            dones_t = done_pinned.to(device, non_blocking=True)

            buffer.insert(
                obs=obs_t, mask=mask_t, history=hist_t,
                hidden_state=agent_hidden, action=act_main,
                log_prob=logp_main, value=val_main,
                reward=rewards_t, done=dones_t
            )

            # Reset hidden states for envs that ended this step
            done_mask = dones_t.float().view(1, -1, 1)
            agent_hidden = next_agent_hidden * (1.0 - done_mask)
            opp_hidden = opp_hidden * (1.0 - done_mask)

            # T5: build outputs for the next outer step
            next_obs_np = env.get_observations()
            next_mask_np = env.get_action_masks()
            next_hist_np = env._get_histories()

            obs_np = next_obs_np
            mask_np = next_mask_np
            hist_np = next_hist_np
            obs_pinned.copy_(torch.from_numpy(obs_np))
            mask_pinned.copy_(torch.from_numpy(mask_np))
            hist_pinned.copy_(torch.from_numpy(hist_np))
            obs_t = obs_pinned.to(device, non_blocking=True)
            mask_t = mask_pinned.to(device, non_blocking=True)
            hist_t = hist_pinned.to(device, non_blocking=True)

        # ==============================================================
        # 2. GAE COMPUTATION
        # ==============================================================
        with torch.no_grad():
            next_val = agent_net.get_value(
                obs=obs_t, history=hist_t, hidden_state=agent_hidden
            )
            buffer.compute_gae(
                next_value=next_val,
                next_done=dones_t,
                gamma=cfg.ppo.gamma,
                gae_lambda=cfg.ppo.gae_lambda
            )

        # ==============================================================
        # 3. PPO UPDATE
        # ==============================================================
        agent_net.train()
        ppo_trainer.update_learning_rate(total_timesteps, cfg.total_timesteps)
        train_metrics = ppo_trainer.train_epoch(buffer)

        # ==============================================================
        # 4. LEAGUE SNAPSHOT PROMOTION
        # ==============================================================
        if league.should_save_snapshot(total_games_completed):
            saved_path = league.add_snapshot(
                agent_net, iteration, cfg.logging.league_dir,
                total_games=total_games_completed
            )
            console.print(
                f"[bold yellow]=> Promoted to League Pool: {saved_path} "
                f"(Elo: {int(league.main_agent_elo)})[/bold yellow]"
            )

        # ==============================================================
        # 5. LOGGING & VRAM GC
        # ==============================================================
        iter_time = time.time() - iter_start_time
        fps = int(iter_env_steps / max(1e-6, iter_time))
        vram_used = get_vram_usage_gb()

        if vram_used > cfg.hardware.vram_budget_gb:
            gc.collect()
            torch.cuda.empty_cache()

        if iteration % cfg.logging.log_interval_iterations == 0:
            league_stats = league.get_stats()
            mean_reward = (float(np.mean(rolling_rewards))
                           if len(rolling_rewards) > 0 else 0.0)
            mean_len = (float(np.mean(rolling_lengths))
                        if len(rolling_lengths) > 0 else 0.0)

            writer.add_scalar("Performance/FPS", fps, total_timesteps)
            writer.add_scalar("Performance/Mean_Reward", mean_reward, total_timesteps)
            writer.add_scalar("Performance/Mean_Game_Length", mean_len, total_timesteps)
            writer.add_scalar("League/Main_Agent_Elo",
                              league_stats["league/main_elo"], total_timesteps)
            writer.add_scalar("League/Pool_Size",
                              league_stats["league/pool_size"], total_timesteps)
            writer.add_scalar("League/Cycle_Win_Rate",
                              league_stats["league/cycle_win_rate"], total_timesteps)
            for k, v in train_metrics.items():
                writer.add_scalar(k, v, total_timesteps)

            opp_label = (f"Archetype ({active_archetype})"
                         if opp_category == "archetype" else opp_category)
            table = Table(title=f"UNO Production Training | Iteration {iteration}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Total Timesteps", f"{total_timesteps:,} / {cfg.total_timesteps:,}")
            table.add_row("Total Games Played", f"{total_games_completed:,}")
            table.add_row("Throughput", f"{fps:,} env-steps/sec")
            table.add_row("Active Opponent", opp_label)
            table.add_row("Main Agent Elo", f"{league_stats['league/main_elo']:.1f}")
            table.add_row("League Pool Size", str(int(league_stats['league/pool_size'])))
            table.add_row("Cycle Win Rate", f"{league_stats['league/cycle_win_rate'] * 100:.1f}%")
            table.add_row("Avg Game Length", f"{mean_len:.1f} turns")
            table.add_row("Policy Loss", f"{train_metrics['loss/policy']:.4f}")
            table.add_row("Value Loss", f"{train_metrics['loss/value']:.4f}")
            table.add_row("Entropy Loss", f"{train_metrics['loss/entropy']:.4f}")
            table.add_row("VRAM Used", f"{vram_used:.2f} GB / {cfg.hardware.vram_budget_gb} GB")

            console.print(table)

        # ==============================================================
        # 6. PERIODIC CHECKPOINTING
        # ==============================================================
        if iteration % cfg.logging.save_interval_iterations == 0:
            ckpt_path = os.path.join(cfg.logging.checkpoint_dir, "latest.pt")
            raw_model = getattr(agent_net, "_orig_mod", agent_net)
            clean_state = league._clean_state_dict(raw_model.state_dict())
            torch.save({
                "iteration": iteration,
                "total_timesteps": total_timesteps,
                "total_games": total_games_completed,
                "model_state_dict": clean_state,
                "optimizer_state_dict": ppo_trainer.optimizer.state_dict(),
                "scaler_state_dict": ppo_trainer.scaler.state_dict(),
                "elo": league.main_agent_elo
            }, ckpt_path)

    writer.close()
    console.print("[bold green]Training Complete![/bold green]")


if __name__ == "__main__":
    main()