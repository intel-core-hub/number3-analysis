"""
src.analysis.stress_test — モンテカルロ・ストレステスト

目的:
    - 予測勝率とKelly基準で資産推移をシミュレーション
    - 最大ドローダウン、破産確率、期待収益分布を算出
"""
from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config import TICKET_COST
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StressTestConfig:
    win_probability: float
    rounds: int = 1000
    simulations: int = 3000
    initial_bankroll: float = 100_000.0
    ticket_cost: int = TICKET_COST
    prize: float = 15_000.0
    kelly_fraction: float = 0.25
    min_bet: float = float(TICKET_COST)
    max_bet: float = 5_000.0
    seed: int | None = 42
    workers: int | None = None


@dataclass
class StressTestResult:
    summary: dict[str, float]
    distribution: pd.DataFrame


def _simulate_chunk(
    n_sims: int,
    cfg: StressTestConfig,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    final_bankroll = np.zeros(n_sims, dtype=float)
    max_drawdowns = np.zeros(n_sims, dtype=float)
    ruined = np.zeros(n_sims, dtype=bool)

    b = (cfg.prize - cfg.ticket_cost) / cfg.ticket_cost
    if b <= 0:
        return final_bankroll, max_drawdowns, ruined

    p = float(np.clip(cfg.win_probability, 0.0, 1.0))
    q = 1.0 - p
    base_kelly = (p * b - q) / b

    for i in range(n_sims):
        bankroll = cfg.initial_bankroll
        peak = bankroll
        max_dd = 0.0
        ruined_flag = False

        for _ in range(cfg.rounds):
            if bankroll < cfg.min_bet:
                ruined_flag = True
                break

            if base_kelly <= 0:
                continue

            fraction = base_kelly * cfg.kelly_fraction
            bet = bankroll * fraction
            bet = max(cfg.min_bet, min(cfg.max_bet, bet))
            bet = (bet // cfg.ticket_cost) * cfg.ticket_cost
            if bet <= 0 or bet > bankroll:
                continue

            if rng.random() < p:
                bankroll += bet * b
            else:
                bankroll -= bet

            if bankroll > peak:
                peak = bankroll
            drawdown = peak - bankroll
            if drawdown > max_dd:
                max_dd = drawdown

        final_bankroll[i] = bankroll
        max_drawdowns[i] = max_dd
        ruined[i] = ruined_flag

    return final_bankroll, max_drawdowns, ruined


def run_stress_test(cfg: StressTestConfig) -> StressTestResult:
    if cfg.simulations <= 0 or cfg.rounds <= 0:
        raise ValueError("simulations and rounds must be positive")

    workers = cfg.workers or max(1, (os.cpu_count() or 2) - 1)
    chunks = np.array_split(np.arange(cfg.simulations), workers)

    seeds = None
    if cfg.seed is not None:
        rng = np.random.default_rng(cfg.seed)
        seeds = rng.integers(0, 2**32 - 1, size=len(chunks)).tolist()

    tasks = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        for idx, chunk in enumerate(chunks):
            chunk_seed = seeds[idx] if seeds is not None else None
            tasks.append(
                pool.apply_async(_simulate_chunk, (len(chunk), cfg, chunk_seed))
            )
        outputs = [task.get() for task in tasks]

    final_bankroll = np.concatenate([o[0] for o in outputs])
    max_drawdowns = np.concatenate([o[1] for o in outputs])
    ruined = np.concatenate([o[2] for o in outputs])

    distribution = pd.DataFrame(
        {
            "final_bankroll": final_bankroll,
            "max_drawdown": max_drawdowns,
            "ruined": ruined.astype(int),
        }
    )

    summary = {
        "simulations": float(cfg.simulations),
        "rounds": float(cfg.rounds),
        "win_probability": float(cfg.win_probability),
        "kelly_fraction": float(cfg.kelly_fraction),
        "mean_final_bankroll": float(np.mean(final_bankroll)),
        "median_final_bankroll": float(np.median(final_bankroll)),
        "p05_final_bankroll": float(np.percentile(final_bankroll, 5)),
        "p95_final_bankroll": float(np.percentile(final_bankroll, 95)),
        "mean_max_drawdown": float(np.mean(max_drawdowns)),
        "p95_max_drawdown": float(np.percentile(max_drawdowns, 95)),
        "ruination_probability": float(np.mean(ruined)),
    }

    logger.info(
        "Stress test done: win=%.3f rounds=%d sims=%d ruin=%.3f",
        cfg.win_probability,
        cfg.rounds,
        cfg.simulations,
        summary["ruination_probability"],
    )

    return StressTestResult(summary=summary, distribution=distribution)
