"""
main.py — Virtuals Protocol Graduation Arb Test Engine
Pipeline orchestrator.

Runs all four stages in sequence with checkpoint logic:
    Stage 1 — Graduation event discovery
    Stage 2 — Uniswap V3 pool seed extraction
    Stage 3 — Price dislocation + competition + exit routes (parallel enrichment)
    Stage 4 — P&L simulation at three sizes
    Final   — Summary stats + go/no-go gate evaluation

Resumes from the last checkpoint if the run is interrupted.

Usage:
    python main.py                 # Full run
    python main.py --reset-stage N # Clear checkpoint for stage N and re-run from there
    python main.py --stats-only    # Re-run summary stats on existing Stage 4 output
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# ── Bootstrap: ensure project root is on the import path ─────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Configure logging before any other imports ────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")

# ── Module imports (after path setup) ─────────────────────────────────────────
import config
from src.chain.checkpoint import (
    checkpoint_exists,
    clear_checkpoint,
    list_checkpoints,
)
from src.discovery.graduation_events import run as run_stage1
from src.pool.seed_extractor import run as run_stage2
from src.price.dislocation import run as run_stage3_prices
from src.competition.bot_detector import run as run_stage3_competition
from src.exit_routes.liquidity_checker import run as run_stage3_exits
from src.simulation.pnl_simulator import run as run_stage4
from src.analysis.summary_stats import run as run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Virtuals Protocol Graduation Arb Test Engine"
    )
    parser.add_argument(
        "--reset-stage",
        type=int,
        metavar="N",
        help="Clear checkpoint for stage N (1–4) and re-run from that stage.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Skip all extraction stages and re-run summary stats from stage4 checkpoint.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Set log level to DEBUG.",
    )
    return parser.parse_args()


_STAGE_NAMES = {
    1: "stage1_graduations",
    2: "stage2_seeds",
    3: ["stage3_prices", "stage3_competition", "stage3_exit_routes"],
    4: "stage4_pnl",
}


def reset_stage(n: int) -> None:
    """Clear checkpoints for stage N and all downstream stages."""
    stages_to_clear = []
    for stage_num in range(n, 5):
        name = _STAGE_NAMES.get(stage_num)
        if isinstance(name, list):
            stages_to_clear.extend(name)
        elif name:
            stages_to_clear.append(name)

    for name in stages_to_clear:
        if checkpoint_exists(name):
            clear_checkpoint(name)
            print(f"  Cleared checkpoint: {name}")
        else:
            print(f"  No checkpoint to clear: {name}")


def merge_stage3(df_prices: pd.DataFrame,
                 df_competition: pd.DataFrame,
                 df_exits: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the three Stage 3 enrichment DataFrames on event_id.

    All three share the same base columns (from Stage 2); their new columns
    are orthogonal so a simple column-union merge is correct.
    """
    # Start with prices (which already carries all Stage 1/2 columns).
    merged = df_prices.copy()

    # Competition columns to add (avoid duplicating base columns).
    comp_new = [c for c in df_competition.columns
                if c not in merged.columns or c == "event_id"]
    competition_cols = ["event_id"] + [c for c in comp_new if c != "event_id"]
    merged = merged.merge(
        df_competition[competition_cols],
        on="event_id",
        how="left",
        suffixes=("", "_comp"),
    )

    # Exit route columns to add.
    exit_new = [c for c in df_exits.columns
                if c not in merged.columns or c == "event_id"]
    exit_cols = ["event_id"] + [c for c in exit_new if c != "event_id"]
    merged = merged.merge(
        df_exits[exit_cols],
        on="event_id",
        how="left",
        suffixes=("", "_exit"),
    )

    return merged


def estimate_runtime(n_events: int) -> str:
    """Return a rough runtime estimate based on the number of graduation events."""
    # Rough benchmarks per event (seconds):
    #   Stage 2: ~2s/event (RPC calls)
    #   Stage 3a: ~3s/event (archive eth_call × 5 blocks + CoinGecko)
    #   Stage 3b: ~5s/event (Basescan API per wallet)
    #   Stage 3c: ~2s/event (archive eth_call)
    #   Stage 4:  ~1s/event (pure computation + a few RPC reads)
    per_event_sec = 2 + 3 + 5 + 2 + 1  # = 13 seconds per event
    total_sec = n_events * per_event_sec
    hours   = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    return f"~{hours}h {minutes}m" if hours > 0 else f"~{minutes}m"


def main() -> None:
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    print("\n" + "═" * 65)
    print("  VIRTUALS PROTOCOL GRADUATION ARB — TEST ENGINE")
    print(f"  Analysis window: July 1, 2025 → present")
    print("═" * 65 + "\n")

    # ── Handle --reset-stage ──────────────────────────────────────────────────
    if args.reset_stage:
        if args.reset_stage not in range(1, 5):
            print(f"ERROR: --reset-stage must be 1–4, got {args.reset_stage}")
            sys.exit(1)
        print(f"Resetting stage {args.reset_stage} and all downstream stages...")
        reset_stage(args.reset_stage)
        print()

    # ── Stats-only mode ───────────────────────────────────────────────────────
    if args.stats_only:
        if not checkpoint_exists("stage4_pnl"):
            print("ERROR: --stats-only requires a Stage 4 checkpoint. Run the full pipeline first.")
            sys.exit(1)
        from src.chain.checkpoint import load_checkpoint
        data = load_checkpoint("stage4_pnl")
        df   = pd.DataFrame(data)
        run_summary(df)
        return

    run_start = time.time()

    # ── Stage 1: Graduation event discovery ──────────────────────────────────
    print("─" * 65)
    print("STAGE 1 — Graduation Event Discovery")
    print("─" * 65)
    s1_start = time.time()
    df_stage1 = run_stage1()
    s1_elapsed = time.time() - s1_start

    n_events = len(df_stage1)
    print(f"[Stage 1] Completed in {s1_elapsed:.0f}s. Found {n_events:,} graduation events.\n")

    if n_events == 0:
        print("FATAL: No graduation events found. Check VIRTUALS_FACTORY_ADDRESS in .env.")
        sys.exit(1)

    # Estimate remaining runtime after stage 1.
    runtime_est = estimate_runtime(n_events)
    print(f"Estimated remaining runtime: {runtime_est}")
    print(f"(Based on {n_events:,} events × ~13 seconds/event per stage)\n")

    # ── Stage 2: Pool seed extraction ────────────────────────────────────────
    print("─" * 65)
    print("STAGE 2 — Uniswap V3 Pool Seed Extraction")
    print("─" * 65)
    s2_start = time.time()
    df_stage2 = run_stage2(df_stage1)
    s2_elapsed = time.time() - s2_start
    pools_found = df_stage2["uniswap_pool_address"].notna().sum()
    print(f"[Stage 2] Completed in {s2_elapsed:.0f}s. "
          f"Pool data: {pools_found}/{n_events} events.\n")

    # ── Stage 3: Parallel enrichment ─────────────────────────────────────────
    # Prices, competition, and exit routes are independent enrichment steps.
    # They all read the same base data (Stage 2 output) and produce orthogonal columns.
    # We run them sequentially here (no async) to stay within RPC rate limits.
    print("─" * 65)
    print("STAGE 3 — Price Dislocation + Competition + Exit Routes")
    print("─" * 65)

    print("  [3a] Price dislocation...")
    s3a_start = time.time()
    df_prices = run_stage3_prices(df_stage2)
    print(f"  [3a] Done in {time.time() - s3a_start:.0f}s.")

    print("  [3b] Competition density...")
    s3b_start = time.time()
    df_competition = run_stage3_competition(df_stage2)
    print(f"  [3b] Done in {time.time() - s3b_start:.0f}s.")

    print("  [3c] Exit route liquidity...")
    s3c_start = time.time()
    df_exits = run_stage3_exits(df_stage2)
    print(f"  [3c] Done in {time.time() - s3c_start:.0f}s.\n")

    # Merge all Stage 3 outputs into one DataFrame.
    df_merged = merge_stage3(df_prices, df_competition, df_exits)
    print(f"[Stage 3] Merged DataFrame: {len(df_merged)} rows × {len(df_merged.columns)} cols.\n")

    # ── Stage 4: P&L simulation ───────────────────────────────────────────────
    print("─" * 65)
    print("STAGE 4 — P&L Simulation (3 sizes × all events)")
    print("─" * 65)
    s4_start = time.time()
    df_stage4 = run_stage4(df_merged)
    print(f"[Stage 4] Completed in {time.time() - s4_start:.0f}s.\n")

    # ── Final: Summary stats + go/no-go gates ─────────────────────────────────
    print("─" * 65)
    print("FINAL — Summary Statistics & Go/No-Go Evaluation")
    print("─" * 65)
    run_summary(df_stage4)

    # ── Pipeline complete ─────────────────────────────────────────────────────
    total_elapsed = time.time() - run_start
    hours   = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    seconds = int(total_elapsed % 60)
    elapsed_str = f"{hours}h {minutes}m {seconds}s" if hours > 0 else f"{minutes}m {seconds}s"

    print(f"\nPipeline complete in {elapsed_str}.")
    print(f"Master dataset:  {Path(config.OUTPUT_DIR) / 'master_dataset.parquet'}")
    print(f"Summary stats:   {Path(config.OUTPUT_DIR) / 'summary_stats.csv'}")
    print(f"Pipeline log:    pipeline.log\n")


if __name__ == "__main__":
    main()
