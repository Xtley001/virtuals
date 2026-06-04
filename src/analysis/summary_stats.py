"""
src/analysis/summary_stats.py — Generate summary_stats.csv and evaluate go/no-go gates.

This is the answer sheet. Every metric is derived from the analysis window
(July 1, 2025 → present). Events outside the window are present in
master_dataset.parquet but excluded from gate evaluation.

Go / No-Go Gates (from gray paper spec):
    Gate 1: median spread at graduation ≥ 2.0%        → PASS or FAIL
    Gate 2: win rate at medium size ≥ 55%             → PASS or FAIL
    Gate 3: median bot count in first 3 blocks ≤ 5    → PASS or FAIL

All three must PASS to proceed with the full production build.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

import config

logger = logging.getLogger(__name__)

OUTPUT_PATH    = Path(config.OUTPUT_DIR) / "summary_stats.csv"
PARQUET_PATH   = Path(config.OUTPUT_DIR) / "master_dataset.parquet"


def run(stage4_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics and evaluate go/no-go gates.

    Args:
        stage4_df: Final merged DataFrame from Stage 4 (all columns included).

    Returns:
        Single-row DataFrame with all summary metrics (also written to CSV).
    """
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ── Write master_dataset.parquet (full dataset, all events) ───────────────
    stage4_df.to_parquet(PARQUET_PATH, index=False)
    logger.info("master_dataset.parquet written: %d rows, %d cols.",
                len(stage4_df), len(stage4_df.columns))

    # ── Apply analysis window filter ──────────────────────────────────────────
    window_end = config.ANALYSIS_WINDOW_END_TIMESTAMP or int(time.time())
    in_window  = (
        (stage4_df["block_timestamp"] >= config.ANALYSIS_WINDOW_START_TIMESTAMP) &
        (stage4_df["block_timestamp"] <= window_end)
    )
    df = stage4_df.loc[in_window].copy()

    import datetime
    window_start_str = datetime.datetime.utcfromtimestamp(
        config.ANALYSIS_WINDOW_START_TIMESTAMP
    ).strftime("%Y-%m-%d")
    window_end_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")

    logger.info(
        "Analysis window: %s → %s | %d events in window / %d total.",
        window_start_str, window_end_str, len(df), len(stage4_df),
    )

    if len(df) == 0:
        logger.error(
            "No events found in analysis window (%s → %s). "
            "Check ANALYSIS_WINDOW_START_TIMESTAMP in config. "
            "Total events in dataset: %d.",
            window_start_str, window_end_str, len(stage4_df),
        )
        raise RuntimeError(
            "No events in analysis window. Cannot compute summary stats or gates."
        )

    # ── Compute all metrics ───────────────────────────────────────────────────
    metrics = _compute_all_metrics(df, window_start_str, window_end_str)

    # ── Evaluate go/no-go gates ───────────────────────────────────────────────
    gate_results = _evaluate_gates(metrics)
    metrics.update(gate_results)

    # ── Write summary_stats.csv ───────────────────────────────────────────────
    stats_df = pd.DataFrame([metrics])
    stats_df.to_csv(OUTPUT_PATH, index=False)
    logger.info("summary_stats.csv written.")

    # ── Print results ─────────────────────────────────────────────────────────
    _print_report(metrics, gate_results, window_start_str, window_end_str)

    return stats_df


def _compute_all_metrics(df: pd.DataFrame, window_start: str, window_end: str) -> dict:
    """Compute every metric defined in the gray paper spec."""

    # Helper: safe median (returns None if column absent or all-null).
    def med(col):
        s = df.get(col)
        if s is None:
            return None
        v = s.dropna()
        return float(v.median()) if len(v) > 0 else None

    def pct(col):
        s = df.get(col)
        if s is None:
            return None
        v = s.dropna()
        return float(v.quantile(0.25)) if len(v) > 0 else None, \
               float(v.quantile(0.75)) if len(v) > 0 else None

    # ── Event counts ──────────────────────────────────────────────────────────
    total_events       = len(df)
    feasible_events    = int((df.get("best_exit_route", pd.Series(dtype=str)) != "none").sum()) \
                         if "best_exit_route" in df.columns else None

    # ── Date range ────────────────────────────────────────────────────────────
    timestamps = df["block_timestamp"].dropna()
    date_range_start = pd.to_datetime(timestamps.min(), unit="s").strftime("%Y-%m-%d") \
                       if len(timestamps) > 0 else None
    date_range_end   = pd.to_datetime(timestamps.max(), unit="s").strftime("%Y-%m-%d") \
                       if len(timestamps) > 0 else None

    # ── Spread metrics ────────────────────────────────────────────────────────
    spread_col = df.get("spread_at_graduation_pct", pd.Series(dtype=float))
    spread_vals = spread_col.dropna() if spread_col is not None else pd.Series(dtype=float)

    median_spread_pct = float(spread_vals.median()) if len(spread_vals) > 0 else None
    p25_spread_pct    = float(spread_vals.quantile(0.25)) if len(spread_vals) > 0 else None
    p75_spread_pct    = float(spread_vals.quantile(0.75)) if len(spread_vals) > 0 else None

    # ── Spread decay ─────────────────────────────────────────────────────────
    blocks_closed = df.get("blocks_until_spread_closed", pd.Series(dtype=float))
    blocks_closed_clean = blocks_closed.dropna() if blocks_closed is not None else pd.Series(dtype=float)
    median_blocks_until_spread_closed = float(blocks_closed_clean.median()) \
                                        if len(blocks_closed_clean) > 0 else None

    spread_b3 = df.get("spread_at_block_plus_3_pct", pd.Series(dtype=float))
    if spread_b3 is not None and len(spread_b3.dropna()) > 0:
        pct_spread_closed_by_block_3 = float(
            (spread_b3.dropna().abs() < config.MIN_SPREAD_PCT_THRESHOLD).mean() * 100
        )
    else:
        pct_spread_closed_by_block_3 = None

    # ── Competition metrics ───────────────────────────────────────────────────
    bot_col = df.get("bot_count_first_3_blocks", pd.Series(dtype=float))
    bot_vals = bot_col.dropna() if bot_col is not None else pd.Series(dtype=float)
    median_bot_count_first_3_blocks = float(bot_vals.median()) if len(bot_vals) > 0 else None

    w1_col = df.get("wallets_block_plus_1", pd.Series(dtype=float))
    if w1_col is not None and len(w1_col.dropna()) > 0:
        pct_events_zero_bots_block_1 = float(
            (w1_col.dropna() == 0).mean() * 100
        )
    else:
        pct_events_zero_bots_block_1 = None

    # ── Pool size ─────────────────────────────────────────────────────────────
    if "seed_virtual_amount" in df.columns and "virtual_usd_price_at_block" in df.columns:
        pool_size_usd = df["seed_virtual_amount"].fillna(0) * \
                        df["virtual_usd_price_at_block"].fillna(0)
        median_pool_size_usd = float(pool_size_usd[pool_size_usd > 0].median()) \
                               if (pool_size_usd > 0).any() else None
    else:
        median_pool_size_usd = None

    # ── Win rates ─────────────────────────────────────────────────────────────
    def win_rate(col):
        s = df.get(col)
        if s is None:
            return None
        feasible_mask = df.get("best_exit_route", pd.Series("none", index=df.index)) != "none"
        feasible_vals = s[feasible_mask].dropna()
        if len(feasible_vals) == 0:
            return None
        return float((feasible_vals > config.MIN_NET_PROFIT_USD).mean())

    win_rate_small  = win_rate("net_profit_small_after_fees")
    win_rate_medium = win_rate("net_profit_medium_after_fees")
    win_rate_large  = win_rate("net_profit_large_after_fees")

    # ── Median net profit at medium size (winning trades only) ────────────────
    med_profit_col = df.get("net_profit_medium_after_fees", pd.Series(dtype=float))
    profitable_medium = med_profit_col[
        (med_profit_col.notna()) & (med_profit_col > config.MIN_NET_PROFIT_USD)
    ] if med_profit_col is not None else pd.Series(dtype=float)
    median_net_profit_medium_usd = float(profitable_medium.median()) \
                                   if len(profitable_medium) > 0 else None

    # ── Graduation frequency ──────────────────────────────────────────────────
    if date_range_start and date_range_end:
        ts_min = timestamps.min()
        ts_max = timestamps.max()
        months_elapsed = (ts_max - ts_min) / (30.44 * 86400)
        avg_monthly_graduations = total_events / months_elapsed if months_elapsed > 0 else None
    else:
        avg_monthly_graduations = None

    # ── Monthly P&L estimate ──────────────────────────────────────────────────
    if all(v is not None for v in [median_net_profit_medium_usd, win_rate_medium,
                                    avg_monthly_graduations]):
        estimated_monthly_profit_medium = (
            median_net_profit_medium_usd * win_rate_medium * avg_monthly_graduations
        )
    else:
        estimated_monthly_profit_medium = None

    # ── Competition trend (linear regression of bot count over time) ──────────
    competition_trend_slope = None
    if "bot_count_first_3_blocks" in df.columns and "block_timestamp" in df.columns:
        trend_data = df[["block_timestamp", "bot_count_first_3_blocks"]].dropna()
        if len(trend_data) >= 5:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                trend_data["block_timestamp"],
                trend_data["bot_count_first_3_blocks"],
            )
            competition_trend_slope = float(slope)
            # Positive slope = competition increasing over time.
            logger.info(
                "Competition trend: slope=%.6f bots/sec (R²=%.3f, p=%.3f)",
                slope, r_value ** 2, p_value,
            )

    return {
        "analysis_window_start":              window_start,
        "analysis_window_end":                window_end,
        "total_graduations_analyzed":         total_events,
        "total_graduations_all_time":         None,  # filled by caller if needed
        "feasible_graduations":               feasible_events,
        "date_range_start":                   date_range_start,
        "date_range_end":                     date_range_end,
        "median_spread_at_graduation_pct":    median_spread_pct,
        "p25_spread_pct":                     p25_spread_pct,
        "p75_spread_pct":                     p75_spread_pct,
        "median_blocks_until_spread_closed":  median_blocks_until_spread_closed,
        "pct_spread_closed_by_block_3":       pct_spread_closed_by_block_3,
        "median_bot_count_first_3_blocks":    median_bot_count_first_3_blocks,
        "pct_events_zero_bots_block_1":       pct_events_zero_bots_block_1,
        "median_pool_size_usd":               median_pool_size_usd,
        "win_rate_small":                     win_rate_small,
        "win_rate_medium":                    win_rate_medium,
        "win_rate_large":                     win_rate_large,
        "median_net_profit_medium_usd":       median_net_profit_medium_usd,
        "avg_monthly_graduations":            avg_monthly_graduations,
        "estimated_monthly_profit_medium":    estimated_monthly_profit_medium,
        "competition_trend_slope":            competition_trend_slope,
    }


def _evaluate_gates(metrics: dict) -> dict:
    """
    Evaluate the three go/no-go gates defined in the gray paper.

    Gates:
        1: median_spread_at_graduation_pct >= 2.0%
        2: win_rate_medium >= 0.55
        3: median_bot_count_first_3_blocks <= 5

    Returns:
        Dict with gate_1_result, gate_2_result, gate_3_result, overall_verdict.
    """
    def _eval(value, threshold, op):
        if value is None:
            return "INSUFFICIENT_DATA"
        if op == "gte":
            return "PASS" if value >= threshold else "FAIL"
        if op == "lte":
            return "PASS" if value <= threshold else "FAIL"
        return "ERROR"

    g1 = _eval(metrics.get("median_spread_at_graduation_pct"), config.GATE_1_MEDIAN_SPREAD_PCT, "gte")
    g2 = _eval(metrics.get("win_rate_medium"), config.GATE_2_WIN_RATE_MEDIUM, "gte")
    g3 = _eval(metrics.get("median_bot_count_first_3_blocks"), config.GATE_3_MEDIAN_BOT_COUNT, "lte")

    all_pass = all(g == "PASS" for g in [g1, g2, g3])
    any_insufficient = any(g == "INSUFFICIENT_DATA" for g in [g1, g2, g3])

    if any_insufficient:
        verdict = "INSUFFICIENT_DATA — rerun with more events or check data quality"
    elif all_pass:
        verdict = "GO — all three gates pass. Proceed to full gray paper and production build."
    else:
        verdict = "NO-GO — one or more gates failed. Analyse root cause before proceeding."

    return {
        "gate_1_median_spread_gte_2pct":        g1,
        "gate_1_value":                         metrics.get("median_spread_at_graduation_pct"),
        "gate_1_threshold":                     config.GATE_1_MEDIAN_SPREAD_PCT,
        "gate_2_win_rate_medium_gte_55pct":     g2,
        "gate_2_value":                         metrics.get("win_rate_medium"),
        "gate_2_threshold":                     config.GATE_2_WIN_RATE_MEDIUM,
        "gate_3_median_bot_count_lte_5":        g3,
        "gate_3_value":                         metrics.get("median_bot_count_first_3_blocks"),
        "gate_3_threshold":                     config.GATE_3_MEDIAN_BOT_COUNT,
        "overall_verdict":                      verdict,
    }


def _print_report(metrics: dict, gate_results: dict, window_start: str, window_end: str) -> None:
    """Print the full analysis report to stdout."""
    sep = "═" * 65

    print(f"\n{sep}")
    print("  VIRTUALS PROTOCOL GRADUATION ARB — TEST ENGINE RESULTS")
    print(f"  Analysis Window: {window_start} → {window_end}")
    print(sep)

    print(f"\n  DATASET")
    print(f"    Events in window:         {metrics.get('total_graduations_analyzed'):>8}")
    print(f"    Feasible (exit route OK): {_fmt(metrics.get('feasible_graduations')):>8}")
    print(f"    Date range:               {metrics.get('date_range_start')} → {metrics.get('date_range_end')}")
    print(f"    Avg graduations/month:    {_fmt(metrics.get('avg_monthly_graduations'), '.1f'):>8}")

    print(f"\n  SPREAD (ARB OPPORTUNITY)")
    print(f"    Median spread @ graduation: {_fmt(metrics.get('median_spread_at_graduation_pct'), '.2f'):>7}%")
    print(f"    P25 / P75 spread:           {_fmt(metrics.get('p25_spread_pct'), '.2f')} / {_fmt(metrics.get('p75_spread_pct'), '.2f')} %")
    print(f"    Median blocks until closed: {_fmt(metrics.get('median_blocks_until_spread_closed'), '.0f'):>8}")
    print(f"    Spread closed by block+3:   {_fmt(metrics.get('pct_spread_closed_by_block_3'), '.1f'):>7}%")

    print(f"\n  COMPETITION")
    print(f"    Median bot count (first 3): {_fmt(metrics.get('median_bot_count_first_3_blocks'), '.1f'):>8}")
    print(f"    Events with 0 bots @ +1:    {_fmt(metrics.get('pct_events_zero_bots_block_1'), '.1f'):>7}%")
    trend = metrics.get("competition_trend_slope")
    trend_str = f"{trend:.4f} bots/sec" if trend is not None else "N/A"
    trend_dir = "↑ INCREASING" if (trend or 0) > 0 else ("↓ DECREASING" if (trend or 0) < 0 else "→ FLAT")
    print(f"    Competition trend:          {trend_str} {trend_dir}")

    print(f"\n  POOL & LIQUIDITY")
    print(f"    Median pool size:           ${_fmt(metrics.get('median_pool_size_usd'), ',.0f'):>12}")

    print(f"\n  P&L SIMULATION")
    print(f"    Win rate @ $5k:             {_fmt_pct(metrics.get('win_rate_small')):>8}")
    print(f"    Win rate @ $25k:            {_fmt_pct(metrics.get('win_rate_medium')):>8}")
    print(f"    Win rate @ $75k:            {_fmt_pct(metrics.get('win_rate_large')):>8}")
    print(f"    Median net profit ($25k):   ${_fmt(metrics.get('median_net_profit_medium_usd'), ',.2f'):>12}")
    print(f"    Est. monthly P&L ($25k):    ${_fmt(metrics.get('estimated_monthly_profit_medium'), ',.0f'):>12}")

    print(f"\n{'─' * 65}")
    print("  GO / NO-GO GATE EVALUATION")
    print(f"{'─' * 65}")
    g1_val = metrics.get("median_spread_at_graduation_pct")
    g2_val = metrics.get("win_rate_medium")
    g3_val = metrics.get("median_bot_count_first_3_blocks")
    g1 = gate_results["gate_1_median_spread_gte_2pct"]
    g2 = gate_results["gate_2_win_rate_medium_gte_55pct"]
    g3 = gate_results["gate_3_median_bot_count_lte_5"]
    print(f"  Gate 1 — Median spread ≥ 2.0%:      "
          f"{_fmt(g1_val, '.2f')}%   [{_gate_icon(g1)} {g1}]")
    print(f"  Gate 2 — Win rate @ $25k ≥ 55%:     "
          f"{_fmt_pct(g2_val)}    [{_gate_icon(g2)} {g2}]")
    print(f"  Gate 3 — Median bot count ≤ 5:      "
          f"{_fmt(g3_val, '.1f')}       [{_gate_icon(g3)} {g3}]")
    print(f"\n  VERDICT: {gate_results['overall_verdict']}")
    print(f"{sep}\n")


def _fmt(v, fmt="") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    try:
        return format(v, fmt) if fmt else str(v)
    except Exception:
        return str(v)


def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v * 100:.1f}%"


def _gate_icon(result: str) -> str:
    return "✓" if result == "PASS" else ("✗" if result == "FAIL" else "?")
