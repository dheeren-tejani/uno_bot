"""
evaluate.py
Interactive CLI play against trained agent and automated benchmark suite
supporting 176-dim card-counting observations, 62 discrete actions, and 
multi-archetype baseline benchmarking.
"""

import os
import math
import random
import time
import argparse
from typing import List, Tuple, Optional
import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from config import TrainingConfig, UnoCardConfig, EnvironmentConfig
from models import MaskedRecurrentActorCritic
from uno_engine import VectorizedUnoEnv, UnoCardMapper
from league import HeuristicArchetypes
from mcts_baseline import MCTSUnoState, ismcts_select_action


console = Console()


class UnoCardFormatter:
    """Helper class to pretty-print UNO cards and action selections."""
    COLOR_NAMES = ["RED", "YELLOW", "GREEN", "BLUE"]
    COLOR_STYLES = ["bold red", "bold yellow", "bold green", "bold blue"]

    @classmethod
    def format_card(cls, card_id: int) -> str:
        if card_id == 52:
            return "[bold magenta]WILD[/bold magenta]"
        elif card_id == 53:
            return "[bold magenta]WILD +4[/bold magenta]"
        
        color_idx = card_id // 13
        card_type = card_id % 13
        color_str = cls.COLOR_NAMES[color_idx]
        style = cls.COLOR_STYLES[color_idx]

        if card_type <= 9:
            name = str(card_type)
        elif card_type == 10:
            name = "SKIP"
        elif card_type == 11:
            name = "REVERSE"
        elif card_type == 12:
            name = "+2"
        else:
            name = "UNKNOWN"

        return f"[{style}]{color_str} {name}[/{style}]"

    @classmethod
    def format_action(cls, action_id: int) -> str:
        if action_id == 60:
            return "[bold cyan]DRAW (Draw 1 card)[/bold cyan]"
        elif action_id == 61:
            return "[bold yellow]PASS (Keep drawn card and end turn)[/bold yellow]"
        elif action_id < 52:
            return f"Play {cls.format_card(action_id)}"
        elif 52 <= action_id <= 55:
            dec_color = cls.COLOR_NAMES[action_id - 52]
            return f"Play [bold magenta]WILD[/bold magenta] (Declare [{cls.COLOR_STYLES[action_id - 52]}]{dec_color}[/{cls.COLOR_STYLES[action_id - 52]}])"
        elif 56 <= action_id <= 59:
            dec_color = cls.COLOR_NAMES[action_id - 56]
            return f"Play [bold magenta]WILD +4[/bold magenta] (Declare [{cls.COLOR_STYLES[action_id - 56]}]{dec_color}[/{cls.COLOR_STYLES[action_id - 56]}])"
        return f"Action {action_id}"


def clean_state_dict(state_dict: dict) -> dict:
    """Strips '_orig_mod.' prefix added by torch.compile if present."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def pick_action(dist, greedy: bool) -> torch.Tensor:
    """Deterministic argmax over masked logits for eval, or stochastic sampling."""
    if greedy:
        return dist.masked_logits.argmax(dim=-1)
    return dist.sample()


def load_trained_agent(model_path: str, cfg: TrainingConfig) -> MaskedRecurrentActorCritic:
    """Instantiates and loads checkpointed weights cleanly."""
    model = MaskedRecurrentActorCritic(cfg.model).to(cfg.hardware.device)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=cfg.hardware.device, weights_only=True)
        raw_state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        clean_state = clean_state_dict(raw_state)
        model.load_state_dict(clean_state)
        console.print(f"[bold green]Loaded model successfully from: {model_path}[/bold green]")
    else:
        console.print(f"[bold red]Checkpoint '{model_path}' not found![/bold red]")
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    model.eval()
    return model


def run_benchmark(model_path: str, num_games: int = 2000, stochastic: bool = False):
    """
    Evaluates the trained model against Random and all Heuristic Archetypes.
    Defaults to greedy (argmax) action selection for a true strength measurement;
    pass stochastic=True for sampling-based parity checks.
    """
    cfg = TrainingConfig()
    cfg.env.num_envs = 1  # Single environment stream for evaluation tracking
    env = VectorizedUnoEnv(env_cfg=cfg.env, card_cfg=cfg.cards, seed=int(time.time()))
    agent = load_trained_agent(model_path, cfg)

    opponents = [
        ("Random Bot", "random"),
        ("Aggro Blitzer (Greedy)", "aggro"),
        ("Card Hoarder (Endgame Trapper)", "hoarder"),
        ("Color Manipulator", "color_manipulator"),
        ("Erratic (Human-Noise)", "erratic")
    ]

    summary_table = Table(title=f"Benchmark Evaluation Summary ({num_games:,} Games per Opponent)")
    summary_table.add_column("Opponent Archetype", style="cyan")
    summary_table.add_column("Win Rate (%)", style="bold green")
    summary_table.add_column("Wins / Games", style="magenta")
    summary_table.add_column("Draws", style="yellow")
    summary_table.add_column("Avg Turns", style="white")

    for opp_label, opp_type in opponents:
        wins = 0
        draws = 0
        total_turns = []

        console.print(f"\n[bold yellow]Starting Benchmark vs {opp_label}...[/bold yellow]")

        for g in range(num_games):
            agent_seat = g % 2   # alternate seats to cancel first-mover advantage
            obs, mask, hist = env.reset()
            hidden = torch.zeros(1, 1, cfg.model.core_gru_dim, device=cfg.hardware.device)
            done = False

            while not done:
                p = int(env.current_player[0])
                if p == agent_seat:
                    # Model Turn
                    obs_t = torch.from_numpy(obs).to(cfg.hardware.device)
                    mask_t = torch.from_numpy(mask).to(cfg.hardware.device)
                    hist_t = torch.from_numpy(hist).to(cfg.hardware.device)

                    with torch.no_grad():
                        dist, _, hidden = agent(obs=obs_t, mask=mask_t, history=hist_t, hidden_state=hidden)
                        action = pick_action(dist, greedy=not stochastic).cpu().numpy()[0]
                else:
                    # Baseline Turn
                    if opp_type == "random":
                        legal = np.where(mask[0])[0]
                        action = int(np.random.choice(legal))
                    else:
                        action = HeuristicArchetypes.get_action(opp_type, env, 0, p, mask[0])

                obs, mask, hist, _, dones, infos = env.step(np.array([action]))
                done = dones[0]

                if done:
                    winner = infos["winners"][0]
                    total_turns.append(infos["turn_counts"][0])
                    if winner == agent_seat:
                        wins += 1
                    elif winner == -1:
                        draws += 1

        win_rate = (wins / num_games) * 100
        avg_len = float(np.mean(total_turns)) if len(total_turns) > 0 else 0.0
        
        summary_table.add_row(
            opp_label,
            f"{win_rate:.2f}%",
            f"{wins} / {num_games}",
            str(draws),
            f"{avg_len:.1f}"
        )

    console.print("\n")
    console.print(summary_table)


def run_mcts_benchmark(
    model_path: str,
    budgets: List[int],
    num_games: int = 200,
    playout_depth: int = 48,
    seed: int = 67,
):
    """
    The definitive strength measurement: trained agent (greedy) vs ISMCTS.
    Seats alternate every game to cancel first-mover advantage. Score uses
    win=1, draw=0.5. Reports binomial 95% confidence intervals.
    """
    cfg = TrainingConfig()
    cfg.env.num_envs = 1
    env = VectorizedUnoEnv(env_cfg=cfg.env, card_cfg=cfg.cards, seed=67)
    agent = load_trained_agent(model_path, cfg)
    rng = random.Random(seed)

    summary_table = Table(title="ISMCTS Strength Benchmark (Agent Greedy vs ISMCTS)")
    summary_table.add_column("Sims/Move", style="cyan")
    summary_table.add_column("Agent Score", style="bold green")
    summary_table.add_column("W / D / L", style="magenta")
    summary_table.add_column("95% CI", style="yellow")
    summary_table.add_column("Avg Turns", style="white")

    for sims in budgets:
        score = 0.0
        wins = draws = losses = 0
        total_turns: List[int] = []

        # NEW: Rich Progress Bar Setup
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("• {task.completed}/{task.total} games"),
            TextColumn("• Score: {task.fields[score]:.3f}"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            mcts_task = progress.add_task(f"Playing vs ISMCTS ({sims} sims/move)", total=num_games, score=0.0)
            
            for g in range(num_games):
                agent_seat = g % 2
                obs, mask, hist = env.reset()
                hidden = torch.zeros(1, 1, cfg.model.core_gru_dim, device=cfg.hardware.device)
                done = False

                while not done:
                    p = int(env.current_player[0])
                    if p == agent_seat:
                        obs_t = torch.from_numpy(obs).to(cfg.hardware.device)
                        mask_t = torch.from_numpy(mask).to(cfg.hardware.device)
                        hist_t = torch.from_numpy(hist).to(cfg.hardware.device)
                        with torch.no_grad():
                            dist, _, hidden = agent(obs=obs_t, mask=mask_t, history=hist_t, hidden_state=hidden)
                            action = int(dist.masked_logits.argmax(dim=-1).item())
                    else:
                        st = MCTSUnoState.from_env(env, 0)
                        action = ismcts_select_action(
                            st, simulations=sims, rng=rng, playout_depth=playout_depth
                        )
                        # --- DEBUG BLOCK ---
                        if not mask[0][action]:
                            console.print("\n[bold red]--- DEBUG: ILLEGAL ACTION DETECTED ---[/bold red]")
                            console.print(f"Action Chosen: {action} ({UnoCardFormatter.format_action(action)})")
                            console.print(f"Phase: {int(env.current_phase[0])}, Top Card: {int(env.top_card[0])}, Active Color: {int(env.active_color[0])}")
                            console.print(f"MCTS Hand: {st.hands[st.current_player]}")
                            console.print(f"MCTS Legal: {st.legal_actions()}")
                            console.print(f"Engine Legal: {np.where(mask[0])[0]}")
                            console.print(f"MCTS Draw Pile Len: {len(st.draw_pile)}")
                            raise AssertionError("search chose an illegal action")
                        # -------------------
    
                    obs, mask, hist, _, dones, infos = env.step(np.array([action]))
                    done = dones[0]

                winner = int(infos["winners"][0])
                total_turns.append(int(infos["turn_counts"][0]))
                if winner == -1:
                    draws += 1
                    score += 0.5
                elif winner == agent_seat:
                    wins += 1
                    score += 1.0
                else:
                    losses += 1

                # NEW: Update the progress bar and running score
                p_hat = score / (g + 1)
                progress.update(mcts_task, advance=1, score=p_hat)

        p_hat = score / num_games
        ci = 1.96 * math.sqrt(max(p_hat * (1.0 - p_hat), 1e-9) / num_games)
        avg_len = float(np.mean(total_turns)) if total_turns else 0.0
        summary_table.add_row(
            str(sims),
            f"{p_hat * 100:.2f}%",
            f"{wins} / {draws} / {losses}",
            f"+/-{ci * 100:.2f}%",
            f"{avg_len:.1f}"
        )

    console.print("\n")
    console.print(summary_table)


def run_interactive(model_path: str, greedy: bool = False):
    """
    Interactive CLI human player (Player 0) vs. Trained Bot (Player 1).
    Pass greedy=True for deterministic bot play instead of sampling.
    """
    cfg = TrainingConfig()
    cfg.env.num_envs = 1
    env = VectorizedUnoEnv(env_cfg=cfg.env, card_cfg=cfg.cards, seed=int(time.time()))
    agent = load_trained_agent(model_path, cfg)

    obs, mask, hist = env.reset()
    hidden = torch.zeros(1, 1, cfg.model.core_gru_dim, device=cfg.hardware.device)

    console.print(Panel.fit(
        "[bold cyan]Interactive 2-Player UNO Arena[/bold cyan]\n"
        "You are [bold green]Player 0 (Human)[/bold green]. The AI is [bold magenta]Player 1 (Bot)[/bold magenta].\n"
        "Features: Discard tracking, 2-phase post-draw actions, and Wild Color declarations."
    ))

    done = False
    while not done:
        p = env.current_player[0]
        phase = env.current_phase[0]
        top_card_str = UnoCardFormatter.format_card(env.top_card[0])
        active_color_str = UnoCardFormatter.COLOR_NAMES[env.active_color[0]]
        active_color_style = UnoCardFormatter.COLOR_STYLES[env.active_color[0]]

        console.print("\n" + "─" * 60)
        phase_label = "[bold cyan]MAIN PHASE[/bold cyan]" if phase == 0 else "[bold yellow]POST-DRAW DECISION PHASE[/bold yellow]"
        console.print(f"Top Discard: {top_card_str}  │  Active Color: [{active_color_style}]{active_color_str}[/{active_color_style}]  │  {phase_label}")
        console.print(
            f"Human Hand: [bold green]{int(np.sum(env.hands[0, 0]))}[/bold green] cards  │  "
            f"AI Hand: [bold magenta]{int(np.sum(env.hands[0, 1]))}[/bold magenta] cards  │  "
            f"Draw Deck: {int(env.draw_pile_pos[0])} cards"
        )

        if p == 0:
            # Human Turn
            
            # --- NEW: Print the human's full hand ---
            hand_counts = env.hands[0, 0]
            hand_cards = []
            for card_id, count in enumerate(hand_counts):
                if count > 0:
                    for _ in range(int(count)):
                        hand_cards.append(UnoCardFormatter.format_card(card_id))
            console.print(f"\n[bold green]Your Hand:[/bold green] {', '.join(hand_cards)}")
            
            # Print legal choices
            legal_indices = np.where(mask[0])[0]
            console.print("[bold]Legal Available Choices:[/bold]")
            for idx, act in enumerate(legal_indices):
                console.print(f"  ({idx + 1}) {UnoCardFormatter.format_action(act)}")

            choice = -1
            while choice < 1 or choice > len(legal_indices):
                try:
                    user_input = input(f"\nEnter choice [1-{len(legal_indices)}]: ")
                    choice = int(user_input)
                except (ValueError, EOFError):
                    choice = -1

            action = legal_indices[choice - 1]
            console.print(f"[green]You performed:[/green] {UnoCardFormatter.format_action(action)}")

        else:
            # AI Turn
            obs_t = torch.from_numpy(obs).to(cfg.hardware.device)
            mask_t = torch.from_numpy(mask).to(cfg.hardware.device)
            hist_t = torch.from_numpy(hist).to(cfg.hardware.device)

            with torch.no_grad():
                dist, value, hidden = agent(obs=obs_t, mask=mask_t, history=hist_t, hidden_state=hidden)
                action = pick_action(dist, greedy=greedy).cpu().numpy()[0]

            console.print(f"\n[magenta]AI performed:[/magenta] {UnoCardFormatter.format_action(action)} (Confidence/Value: {value.item():+.2f})")

        obs, mask, hist, _, dones, infos = env.step(np.array([action]))
        done = dones[0]

        if done:
            winner = infos["winners"][0]
            turns = infos["turn_counts"][0]
            if winner == 0:
                console.print(f"\n[bold green]🏆 CONGRATULATIONS! You won in {turns} turns![/bold green]")
            elif winner == 1:
                console.print(f"\n[bold red]💀 GAME OVER! The AI bot won in {turns} turns![/bold red]")
            else:
                console.print(f"\n[bold yellow]🤝 Match ended in a DRAW after {turns} turns![/bold yellow]")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production UNO RL Evaluation & Arena")
    parser.add_argument("--model", type=str, default="./checkpoints/latest.pt", help="Path to model checkpoint")
    parser.add_argument("--interactive", action="store_true", help="Play interactively in terminal against the AI")
    parser.add_argument("--benchmark", action="store_true", help="Run full multi-archetype benchmark")
    parser.add_argument("--games", type=int, default=2000, help="Number of games per baseline archetype")
    parser.add_argument("--greedy", action="store_true", help="Interactive mode: bot plays argmax instead of sampling")
    parser.add_argument("--sample", action="store_true", help="Benchmark mode: sample actions instead of greedy argmax")
    parser.add_argument("--mcts", action="store_true", help="Run ISMCTS strength benchmark instead of archetype benchmark")
    parser.add_argument("--mcts-sims", type=str, default="100,400", help="Comma-separated ISMCTS simulation budgets")
    parser.add_argument("--mcts-games", type=int, default=200, help="Games per ISMCTS budget")
    parser.add_argument("--playout-depth", type=int, default=48, help="Random-playout truncation depth for ISMCTS")

    args = parser.parse_args()

    if args.interactive:
        run_interactive(args.model, greedy=args.greedy)
    elif args.mcts:
        budgets = [int(b) for b in args.mcts_sims.split(",") if b.strip()]
        run_mcts_benchmark(
            args.model,
            budgets=budgets,
            num_games=args.mcts_games,
            playout_depth=args.playout_depth,
        )
    else:
        run_benchmark(args.model, num_games=args.games, stochastic=args.sample)