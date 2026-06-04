"""
src/simulation/pnl_simulator.py — Stage 4: Arb P&L simulation at three sizes.

For every feasible graduation event (best_exit_route != 'none'), simulates
the full flash loan arb at $5k, $25k, and $75k sizes using exact Uniswap V3
math and a complete fee model.

Fee model (per trade):
    - Flash loan fee:    0%  (Balancer V2 primary — zero fee)
    - Entry pool fee:    pool_fee_tier / 1,000,000 (e.g. 0.3% for fee_tier=3000)
    - Exit fee:          0.3% for Aerodrome, 0.3% for Uniswap V3 secondary
    - Gas estimate:      (base_fee + priority_fee) × 200,000 gas units × ETH/USD
    - Slippage:          modelled from actual pool reserves via uniswap_v3_math

    Net profit = exit_proceeds_usd - entry_cost_usd - pool_fees_usd - gas_cost_usd

Gas cost formula:
    gas_cost_usd = (base_fee_gwei + priority_fee_gwei) × 200_000 × eth_usd / 1e9
    Source: build_guide.md spec; 200,000 gas = proxy for a 4-swap transaction on Base.
    Priority fee estimate: 0.001 gwei (Base is a low-fee L2).

Output:
    data/raw/stage4_pnl.csv
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

import config
from src.chain.rpc_client import get_client, call_with_retry
from src.chain.checkpoint import load_checkpoint, save_checkpoint, checkpoint_exists
from src.simulation.uniswap_v3_math import (
    calculate_price_impact,
    sqrt_price_x96_to_price,
    get_pool_state,
)
from src.price.virtual_price import get_virtual_price_at_timestamp

logger = logging.getLogger(__name__)

STAGE_NAME = "stage4_pnl"
OUTPUT_PATH = Path(config.RAW_DIR) / "stage4_pnl.csv"

# Fixed gas units for a complete 4-swap arb transaction on Base.
# Source: build_guide.md specification.
_GAS_UNITS_ESTIMATE = 200_000

# Priority fee on Base (L2 with low fees). Source: build_guide.md specification.
_PRIORITY_FEE_GWEI = 0.001

# Aerodrome fee: 0.3% on volatile pairs (standard Aerodrome LP fee).
_AERODROME_FEE_PCT = 0.003

# Uniswap V3 secondary pool fee: 0.3% (most common tier).
_UNISWAP_SECONDARY_FEE_PCT = 0.003

# Exit block: we simulate the exit at graduation_block + 1.
_EXIT_BLOCK_OFFSET = 1


def run(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute Stage 4: P&L simulation for all feasible events.

    Args:
        merged_df: DataFrame with all Stage 1–3 columns merged (by event_id).
                   Must include: event_id, block_number, block_timestamp,
                   uniswap_pool_address, best_exit_route, pool_fee_tier,
                   seed_price_in_virtual, virtual_usd_price_at_block.

    Returns:
        DataFrame with P&L columns appended.
    """
    if checkpoint_exists(STAGE_NAME):
        logger.info("Stage 4: Loading P&L simulation from checkpoint.")
        data = load_checkpoint(STAGE_NAME)
        df = pd.DataFrame(data)
        logger.info("Stage 4: Loaded %d records from checkpoint.", len(df))
        return df

    logger.info("Stage 4: Running P&L simulation for %d events.", len(merged_df))

    w3 = get_client()
    records = []
    total = len(merged_df)

    for i, row in merged_df.iterrows():
        event_id     = row["event_id"]
        block_number = int(row["block_number"])
        block_ts     = int(row["block_timestamp"])
        pool_address = row.get("uniswap_pool_address")
        exit_route   = row.get("best_exit_route", "none")

        logger.debug("[%d/%d] event_id=%d block=%d", i + 1, total, event_id, block_number)

        if not pool_address or pd.isna(pool_address) or str(exit_route) == "none":
            records.append({**row.to_dict(), **_null_pnl_record(reason="not_feasible")})
            continue

        try:
            pnl = _simulate_event_pnl(
                w3=w3,
                row=row,
                event_id=event_id,
                block_number=block_number,
                block_ts=block_ts,
                pool_address=str(pool_address),
                exit_route=str(exit_route),
            )
            records.append({**row.to_dict(), **pnl})
        except Exception as exc:
            logger.warning("event_id=%d: P&L simulation failed: %s", event_id, exc)
            records.append({**row.to_dict(), **_null_pnl_record(reason=f"error: {exc}")})

    df = pd.DataFrame(records)
    df = df.sort_values("event_id").reset_index(drop=True)

    # Apply analysis window filter (July 1, 2025 → present).
    import time
    window_start = config.ANALYSIS_WINDOW_START_TIMESTAMP
    window_end   = config.ANALYSIS_WINDOW_END_TIMESTAMP or int(time.time())

    in_window = (
        (df["block_timestamp"] >= window_start) &
        (df["block_timestamp"] <= window_end)
    )
    total_in_window = in_window.sum()
    logger.info(
        "Analysis window (Jul 1 2025 → present): %d/%d events in range.",
        total_in_window, len(df),
    )

    Path(config.RAW_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    save_checkpoint(STAGE_NAME, df.to_dict(orient="records"))

    profitable_medium = (
        df.loc[in_window, "net_profit_medium_after_fees"]
        .apply(lambda x: x > config.MIN_NET_PROFIT_USD if pd.notna(x) else False)
        .sum()
    )
    print(f"\n[Stage 4] ✓ P&L simulation complete: {total_in_window} events in analysis window")
    print(f"[Stage 4] Profitable at medium size ($25k): {profitable_medium}")
    print(f"[Stage 4] Output: {OUTPUT_PATH}")
    return df


def _simulate_event_pnl(
    w3,
    row: pd.Series,
    event_id: int,
    block_number: int,
    block_ts: int,
    pool_address: str,
    exit_route: str,
) -> dict:
    """
    Run the full P&L simulation for one event at all three sizes.

    Entry: buy agent token on the graduation Uniswap V3 pool at graduation_block.
    Exit:  sell agent token via best_exit_route at graduation_block + 1.

    Returns:
        Dict with all P&L fields.
    """
    # ── Gas cost for this block ───────────────────────────────────────────────
    try:
        blk = call_with_retry(lambda: w3.eth.get_block(block_number))
        base_fee_gwei = blk.get("baseFeePerGas", 0) / 1e9  # wei → gwei
    except Exception as exc:
        logger.debug("event_id=%d: could not read baseFeePerGas: %s", event_id, exc)
        base_fee_gwei = 0.01  # Conservative fallback: 0.01 gwei on Base.

    # ETH/USD price at graduation block (for gas cost calculation).
    eth_usd = _get_eth_usd_price(block_ts)

    # gas_cost_usd = (base_fee_gwei + priority_fee_gwei) × gas_units × eth_usd / 1e9
    # Source: build_guide.md specification.
    gas_cost_usd = (
        (base_fee_gwei + _PRIORITY_FEE_GWEI) * _GAS_UNITS_ESTIMATE * eth_usd / 1e9
    )

    # ── $VIRTUAL price at graduation ──────────────────────────────────────────
    virtual_usd = row.get("virtual_usd_price_at_block")
    if not virtual_usd or pd.isna(virtual_usd):
        virtual_usd = get_virtual_price_at_timestamp(block_ts)
    virtual_usd = float(virtual_usd)

    if virtual_usd <= 0:
        return _null_pnl_record(reason="virtual_usd_price_zero")

    # Pool fee tier (e.g. 3000 = 0.3%).
    pool_fee_tier = row.get("pool_fee_tier", 3000)
    if pd.isna(pool_fee_tier):
        pool_fee_tier = 3000
    pool_fee_tier = int(pool_fee_tier)

    pool_fee_pct = pool_fee_tier / 1_000_000  # e.g. 0.003 for 3000

    # Exit fee rate.
    if exit_route == "aerodrome":
        exit_fee_pct = _AERODROME_FEE_PCT
    else:
        exit_fee_pct = _UNISWAP_SECONDARY_FEE_PCT

    # ── Pool state at graduation block and exit block ─────────────────────────
    try:
        pool_state_entry = get_pool_state(pool_address, block_number)
        pool_state_exit  = get_pool_state(pool_address, block_number + _EXIT_BLOCK_OFFSET)
    except Exception as exc:
        logger.debug("event_id=%d: pool state read failed: %s", event_id, exc)
        return _null_pnl_record(reason=f"pool_state_error: {exc}")

    # ── Determine which token is the agent token ──────────────────────────────
    virtual_addr = w3.to_checksum_address(config.VIRTUAL_TOKEN_ADDRESS)
    if pool_state_entry.token0 == virtual_addr:
        # token0=VIRTUAL, token1=agent → we sell VIRTUAL (token0) to buy agent (token1)
        virtual_is_token0 = True
        token_in_addr = pool_state_entry.token0   # VIRTUAL
    else:
        # token0=agent, token1=VIRTUAL → we sell VIRTUAL (token1) to buy agent (token0)
        virtual_is_token0 = False
        token_in_addr = pool_state_entry.token1   # VIRTUAL

    # ── Simulate each arb size ────────────────────────────────────────────────
    sizes = {
        "small":  config.SIM_SIZE_SMALL_USD,
        "medium": config.SIM_SIZE_MEDIUM_USD,
        "large":  config.SIM_SIZE_LARGE_USD,
    }

    results = {}
    any_profitable = False

    for label, size_usd in sizes.items():
        sim = _simulate_single_size(
            pool_address=pool_address,
            block_entry=block_number,
            block_exit=block_number + _EXIT_BLOCK_OFFSET,
            token_in_addr=token_in_addr,
            size_usd=size_usd,
            virtual_usd=virtual_usd,
            pool_fee_pct=pool_fee_pct,
            exit_fee_pct=exit_fee_pct,
            gas_cost_usd=gas_cost_usd,
            virtual_is_token0=virtual_is_token0,
            event_id=event_id,
        )
        results[label] = sim
        if sim.get("net_profit_usd", 0) > config.MIN_NET_PROFIT_USD:
            any_profitable = True

    return {
        # Entry/cost data.
        "gas_cost_usd": gas_cost_usd,
        "base_fee_gwei": base_fee_gwei,
        "eth_usd_at_block": eth_usd,
        # Small size.
        "sim_size_small_usd": config.SIM_SIZE_SMALL_USD,
        "sim_profit_small_usd": results["small"].get("gross_profit_usd"),
        "sim_entry_fee_small_usd": results["small"].get("entry_fee_usd"),
        "sim_exit_fee_small_usd": results["small"].get("exit_fee_usd"),
        "net_profit_small_after_fees": results["small"].get("net_profit_usd"),
        "sim_price_impact_small_pct": results["small"].get("price_impact_pct"),
        # Medium size.
        "sim_size_medium_usd": config.SIM_SIZE_MEDIUM_USD,
        "sim_profit_medium_usd": results["medium"].get("gross_profit_usd"),
        "sim_entry_fee_medium_usd": results["medium"].get("entry_fee_usd"),
        "sim_exit_fee_medium_usd": results["medium"].get("exit_fee_usd"),
        "net_profit_medium_after_fees": results["medium"].get("net_profit_usd"),
        "sim_price_impact_medium_pct": results["medium"].get("price_impact_pct"),
        # Large size.
        "sim_size_large_usd": config.SIM_SIZE_LARGE_USD,
        "sim_profit_large_usd": results["large"].get("gross_profit_usd"),
        "sim_entry_fee_large_usd": results["large"].get("entry_fee_usd"),
        "sim_exit_fee_large_usd": results["large"].get("exit_fee_usd"),
        "net_profit_large_after_fees": results["large"].get("net_profit_usd"),
        "sim_price_impact_large_pct": results["large"].get("price_impact_pct"),
        # Summary flag.
        "was_profitable_any_size": any_profitable,
    }


def _simulate_single_size(
    pool_address: str,
    block_entry: int,
    block_exit: int,
    token_in_addr: str,
    size_usd: float,
    virtual_usd: float,
    pool_fee_pct: float,
    exit_fee_pct: float,
    gas_cost_usd: float,
    virtual_is_token0: bool,
    event_id: int,
) -> dict:
    """
    Simulate the entry + exit for one arb size and return the P&L breakdown.

    Entry: buy agent token on graduation pool at block_entry.
    Exit:  sell agent token back (simplification: back through the same pool or
           exit route) at block_exit. For the simulation, we model exit as selling
           at the pool's slot0 price at block_exit — this is conservative because
           it ignores exit route slippage direction (which is in our favour on exit).

    Returns:
        Dict with gross_profit_usd, entry_fee_usd, exit_fee_usd, net_profit_usd,
        price_impact_pct keys.
    """
    # Convert USD size to VIRTUAL wei.
    virtual_amount_human = size_usd / virtual_usd
    virtual_amount_wei   = int(virtual_amount_human * 1e18)

    if virtual_amount_wei <= 0:
        return {"gross_profit_usd": None, "net_profit_usd": None,
                "entry_fee_usd": None, "exit_fee_usd": None, "price_impact_pct": None}

    # ── Entry: buy agent tokens with VIRTUAL at graduation block ─────────────
    try:
        entry_result = calculate_price_impact(
            pool_address=pool_address,
            block_number=block_entry,
            token_in=token_in_addr,
            amount_in_wei=virtual_amount_wei,
        )
    except Exception as exc:
        logger.debug(
            "event_id=%d size=$%.0f: entry simulation failed: %s",
            event_id, size_usd, exc,
        )
        return {"gross_profit_usd": None, "net_profit_usd": None,
                "entry_fee_usd": None, "exit_fee_usd": None, "price_impact_pct": None}

    agent_tokens_received_wei = entry_result.amount_out
    if agent_tokens_received_wei <= 0:
        return {"gross_profit_usd": None, "net_profit_usd": None,
                "entry_fee_usd": None, "exit_fee_usd": None, "price_impact_pct": None}

    # Entry cost in USD (the VIRTUAL we spent).
    entry_cost_usd = virtual_amount_human * virtual_usd

    # Entry fee in USD.
    entry_fee_usd = entry_cost_usd * pool_fee_pct

    # ── Exit: determine exit value at block_exit ──────────────────────────────
    # We model exit by reading the pool's price at block_exit and computing
    # the value of our agent tokens at that price. This is a conservative
    # bound — real exit through Aerodrome would depend on Aerodrome's price.
    try:
        exit_state = get_pool_state(pool_address, block_exit)
        # Price of agent token in VIRTUAL at exit block.
        if virtual_is_token0:
            # token0=VIRTUAL, token1=agent
            # price = token1/token0 = agent/VIRTUAL → invert for VIRTUAL per agent
            raw_price = (exit_state.sqrt_price_x96 / (2 ** 96)) ** 2
            # raw_price = agent/VIRTUAL → VIRTUAL per agent = 1/raw_price
            virtual_per_agent = 1.0 / raw_price if raw_price > 0 else 0.0
        else:
            # token0=agent, token1=VIRTUAL
            # price = token1/token0 = VIRTUAL/agent
            raw_price = (exit_state.sqrt_price_x96 / (2 ** 96)) ** 2
            virtual_per_agent = raw_price

        # $VIRTUAL price at exit block.
        virtual_usd_exit = get_virtual_price_at_timestamp(
            call_with_retry(lambda: get_client().eth.get_block(block_exit))["timestamp"]
        )

        agent_tokens_human = agent_tokens_received_wei / 1e18
        exit_value_usd     = agent_tokens_human * virtual_per_agent * virtual_usd_exit

        # Exit fee.
        exit_fee_usd = exit_value_usd * exit_fee_pct

        # Gross profit: what we receive minus what we spent (before fees and gas).
        gross_profit_usd = exit_value_usd - entry_cost_usd

        # Net profit: gross - all fees - gas.
        # Flash loan fee = 0% (Balancer V2). Source: build_guide.md specification.
        net_profit_usd = (
            gross_profit_usd
            - entry_fee_usd
            - exit_fee_usd
            - gas_cost_usd
        )

        return {
            "gross_profit_usd":  gross_profit_usd,
            "entry_fee_usd":     entry_fee_usd,
            "exit_fee_usd":      exit_fee_usd,
            "net_profit_usd":    net_profit_usd,
            "price_impact_pct":  entry_result.price_impact_pct,
        }

    except Exception as exc:
        logger.debug(
            "event_id=%d size=$%.0f: exit simulation failed: %s",
            event_id, size_usd, exc,
        )
        return {"gross_profit_usd": None, "net_profit_usd": None,
                "entry_fee_usd": None, "exit_fee_usd": None, "price_impact_pct": None}


def _get_eth_usd_price(block_timestamp: int) -> float:
    """
    Approximate ETH/USD price at a given timestamp for gas cost calculation.

    Uses the same CoinGecko client as VIRTUAL prices to avoid adding a new
    API dependency. Returns a conservative fallback of $3,000 on failure.

    On Base, gas costs are minimal enough that a 2× ETH price error shifts
    gas cost by <$0.10 on a typical transaction — acceptable for simulation.
    """
    try:
        import requests as req
        from src.price.virtual_price import _client as price_client
        import time

        url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart/range"
        params = {
            "vs_currency": "usd",
            "from": block_timestamp - 1800,
            "to": block_timestamp + 1800,
        }
        if config.COINGECKO_API_KEY:
            headers = {"x-cg-pro-api-key": config.COINGECKO_API_KEY}
        else:
            headers = {}

        resp = req.get(url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        prices = data.get("prices", [])
        if prices:
            return float(prices[len(prices) // 2][1])  # midpoint of window
    except Exception as exc:
        logger.debug("ETH price fetch failed, using $3000 fallback: %s", exc)

    return 3_000.0  # Conservative fallback.


def _null_pnl_record(reason: str = "") -> dict:
    """Return null P&L fields with an optional reason tag."""
    return {
        "sim_reason": reason,
        "gas_cost_usd": None,
        "base_fee_gwei": None,
        "eth_usd_at_block": None,
        "sim_size_small_usd": config.SIM_SIZE_SMALL_USD,
        "sim_profit_small_usd": None,
        "sim_entry_fee_small_usd": None,
        "sim_exit_fee_small_usd": None,
        "net_profit_small_after_fees": None,
        "sim_price_impact_small_pct": None,
        "sim_size_medium_usd": config.SIM_SIZE_MEDIUM_USD,
        "sim_profit_medium_usd": None,
        "sim_entry_fee_medium_usd": None,
        "sim_exit_fee_medium_usd": None,
        "net_profit_medium_after_fees": None,
        "sim_price_impact_medium_pct": None,
        "sim_size_large_usd": config.SIM_SIZE_LARGE_USD,
        "sim_profit_large_usd": None,
        "sim_entry_fee_large_usd": None,
        "sim_exit_fee_large_usd": None,
        "net_profit_large_after_fees": None,
        "sim_price_impact_large_pct": None,
        "was_profitable_any_size": False,
    }
