"""
src/price/dislocation.py — Stage 3a: Price dislocation (arb spread) measurement.

For each graduation event, this module:
    1. Reads the actual pool price (via slot0) at blocks +1, +3, +10, +30.
    2. Calculates fair value = seed_price_in_virtual × virtual_usd_price.
    3. Computes spread = (fair_value - pool_price) / fair_value × 100.
    4. Finds blocks_until_spread_closed (first block where spread < 0.5%).

Why pool slot0 at historical blocks (not just initialization price):
    slot0 gives the actual market price state at that block, including all trades
    that landed between the graduation block and the measurement block. This is
    the real arb opportunity — not just the init price.

    We use eth_call with block_identifier=block_number (archive node required).
    If the node is not an archive node, these calls will fail with "missing trie node".

Output:
    data/raw/stage3_prices.csv
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from web3 import Web3

import config
from src.chain.rpc_client import get_client, call_with_retry
from src.chain.checkpoint import load_checkpoint, save_checkpoint, checkpoint_exists
from src.price.virtual_price import get_virtual_price_at_timestamp

logger = logging.getLogger(__name__)

STAGE_NAME = "stage3_prices"
OUTPUT_PATH = Path(config.RAW_DIR) / "stage3_prices.csv"

# Block offsets at which we measure the spread.
_CHECKPOINT_BLOCKS = [0, 1, 3, 10, 30]

# Spread below this threshold is considered "closed".
_SPREAD_CLOSED_THRESHOLD_PCT = config.MIN_SPREAD_PCT_THRESHOLD


def run(stage2_df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute Stage 3a: compute price dislocation at each checkpoint block.

    Args:
        stage2_df: Output DataFrame from Stage 2 (includes pool seed data).

    Returns:
        DataFrame with price dislocation columns appended.
    """
    if checkpoint_exists(STAGE_NAME):
        logger.info("Stage 3a: Loading price dislocation from checkpoint.")
        data = load_checkpoint(STAGE_NAME)
        df = pd.DataFrame(data)
        logger.info("Stage 3a: Loaded %d records from checkpoint.", len(df))
        return df

    w3 = get_client()

    # Filter to events with valid pool data.
    valid = stage2_df[stage2_df["uniswap_pool_address"].notna()].copy()
    logger.info(
        "Stage 3a: Computing dislocation for %d events with pool data (%d total).",
        len(valid), len(stage2_df),
    )

    records = []
    total = len(valid)

    for i, row in valid.iterrows():
        event_id = row["event_id"]
        block_number = int(row["block_number"])
        block_timestamp = int(row["block_timestamp"])
        pool_address = row["uniswap_pool_address"]
        seed_price_in_virtual = row.get("seed_price_in_virtual")
        pool_token0 = row.get("pool_token0")

        if not pool_address or not seed_price_in_virtual:
            records.append({**row.to_dict(), **_null_dislocation_record()})
            continue

        logger.debug(
            "[%d/%d] event_id=%d block=%d pool=%s",
            i + 1, total, event_id, block_number, pool_address,
        )

        try:
            dis_record = _compute_dislocation(
                w3=w3,
                event_id=event_id,
                block_number=block_number,
                block_timestamp=block_timestamp,
                pool_address=pool_address,
                seed_price_in_virtual=float(seed_price_in_virtual),
                token0_address=pool_token0,
            )
            records.append({**row.to_dict(), **dis_record})
        except Exception as exc:
            logger.warning(
                "event_id=%d: dislocation computation failed: %s", event_id, exc
            )
            records.append({**row.to_dict(), **_null_dislocation_record()})

    # Append null rows for events with no pool data.
    no_pool = stage2_df[stage2_df["uniswap_pool_address"].isna()]
    for _, row in no_pool.iterrows():
        records.append({**row.to_dict(), **_null_dislocation_record()})

    df = pd.DataFrame(records)
    df = df.sort_values("event_id").reset_index(drop=True)

    Path(config.RAW_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    save_checkpoint(STAGE_NAME, df.to_dict(orient="records"))

    non_null = df["spread_at_graduation_pct"].notna().sum()
    median_spread = df["spread_at_graduation_pct"].median()
    logger.info(
        "Stage 3a complete: %d/%d events have spread data. Median spread: %.2f%%",
        non_null, len(df), median_spread if not pd.isna(median_spread) else 0,
    )
    print(f"\n[Stage 3a] ✓ Price dislocation computed: {non_null}/{len(df)} events")
    if not pd.isna(median_spread):
        print(f"[Stage 3a] Median spread at graduation: {median_spread:.2f}%")
    print(f"[Stage 3a] Output: {OUTPUT_PATH}")
    return df


def _compute_dislocation(
    w3: Web3,
    event_id: int,
    block_number: int,
    block_timestamp: int,
    pool_address: str,
    seed_price_in_virtual: float,
    token0_address: Optional[str],
) -> dict:
    """
    Compute price dislocation metrics for a single graduation event.

    Reads pool slot0 at each checkpoint block to get the actual market price.
    Computes spread vs. seed price (fair value) at each checkpoint.

    Args:
        w3:                     Web3 client.
        event_id:               Event identifier for logging.
        block_number:           Graduation block number.
        block_timestamp:        Graduation block Unix timestamp.
        pool_address:           Uniswap V3 pool address.
        seed_price_in_virtual:  Pool initialization price (VIRTUAL per agent token).
        token0_address:         The token0 address of the pool.

    Returns:
        Dict with dislocation fields.
    """
    pool_contract = w3.eth.contract(
        address=w3.to_checksum_address(pool_address),
        abi=config.UNISWAP_V3_POOL_ABI,
    )

    # Get $VIRTUAL/USDC prices at and around graduation.
    virtual_usd_at_block = get_virtual_price_at_timestamp(block_timestamp)

    # Seed price in USD.
    seed_price_usd = seed_price_in_virtual * virtual_usd_at_block

    # Gather block timestamps for the checkpoint blocks.
    # We need timestamps to fetch VIRTUAL price at each checkpoint.
    checkpoint_data: dict[int, dict] = {}

    for offset in _CHECKPOINT_BLOCKS:
        target_block = block_number + offset
        try:
            # Read slot0 at the target block using historical eth_call.
            # Requires archive node — will raise with "missing trie node" if not archive.
            slot0 = call_with_retry(
                lambda b=target_block: pool_contract.functions.slot0().call(
                    block_identifier=b
                )
            )
            sqrt_price_x96 = slot0[0]
            tick = slot0[1]

            # Convert sqrtPriceX96 to price.
            # Formula: price = (sqrtPriceX96 / 2^96)^2
            # Source: Uniswap V3 Core whitepaper, Section 3.
            raw_price = (sqrt_price_x96 / (2 ** 96)) ** 2

            # Determine direction: price is token1/token0.
            # We want agent token price in VIRTUAL terms.
            virtual_addr = w3.to_checksum_address(config.VIRTUAL_TOKEN_ADDRESS)
            if token0_address and w3.to_checksum_address(token0_address) == virtual_addr:
                # token0=VIRTUAL, token1=agent → raw_price = agent/VIRTUAL → invert
                if raw_price > 0:
                    pool_price_in_virtual = 1.0 / raw_price
                else:
                    pool_price_in_virtual = 0.0
            else:
                # token0=agent, token1=VIRTUAL → raw_price = VIRTUAL/agent (what we want)
                pool_price_in_virtual = raw_price

            # Fetch block timestamp for this checkpoint block.
            blk = call_with_retry(
                lambda b=target_block: w3.eth.get_block(b)
            )
            checkpoint_ts = blk["timestamp"]

            # Get $VIRTUAL price at this checkpoint timestamp.
            virtual_usd_at_checkpoint = get_virtual_price_at_timestamp(checkpoint_ts)

            # Agent token price in USD at this checkpoint.
            pool_price_usd = pool_price_in_virtual * virtual_usd_at_checkpoint

            # Spread: how much the pool price has moved from seed (fair value).
            # Positive spread = pool price below seed → buy opportunity.
            # Negative spread = pool price already above seed → no arb.
            if seed_price_usd > 0:
                spread_pct = (seed_price_usd - pool_price_usd) / seed_price_usd * 100.0
            else:
                spread_pct = 0.0

            checkpoint_data[offset] = {
                "sqrt_price_x96": sqrt_price_x96,
                "pool_price_in_virtual": pool_price_in_virtual,
                "pool_price_usd": pool_price_usd,
                "spread_pct": spread_pct,
                "virtual_usd": virtual_usd_at_checkpoint,
                "block_timestamp": checkpoint_ts,
            }

        except Exception as exc:
            error_str = str(exc).lower()
            if "missing trie node" in error_str or "state" in error_str:
                logger.error(
                    "Archive node required for historical slot0 queries. "
                    "Your RPC provider does not have archive access. "
                    "Use Alchemy Growth tier or QuickNode archive plan.\n"
                    "Error: %s",
                    exc,
                )
                raise RuntimeError(
                    "Archive node required. See log for details."
                ) from exc
            logger.debug(
                "event_id=%d offset=+%d: slot0 read failed: %s",
                event_id, offset, exc,
            )
            checkpoint_data[offset] = None

    # Determine blocks_until_spread_closed.
    blocks_until_closed = _find_spread_close_block(
        block_number=block_number,
        pool_contract=pool_contract,
        seed_price_usd=seed_price_usd,
        token0_address=token0_address,
        w3=w3,
    )

    return {
        "virtual_usd_price_at_block": virtual_usd_at_block,
        "seed_price_usd": seed_price_usd,
        # Block +0
        "agent_price_block_0_usd": checkpoint_data[0]["pool_price_usd"] if checkpoint_data.get(0) else None,
        "virtual_usd_price_t0": checkpoint_data[0]["virtual_usd"] if checkpoint_data.get(0) else None,
        "spread_at_graduation_pct": checkpoint_data[0]["spread_pct"] if checkpoint_data.get(0) else None,
        # Block +1
        "agent_price_block_plus_1": checkpoint_data[1]["pool_price_usd"] if checkpoint_data.get(1) else None,
        "virtual_usd_price_t1": checkpoint_data[1]["virtual_usd"] if checkpoint_data.get(1) else None,
        "spread_at_block_plus_1_pct": checkpoint_data[1]["spread_pct"] if checkpoint_data.get(1) else None,
        # Block +3
        "agent_price_block_plus_3": checkpoint_data[3]["pool_price_usd"] if checkpoint_data.get(3) else None,
        "spread_at_block_plus_3_pct": checkpoint_data[3]["spread_pct"] if checkpoint_data.get(3) else None,
        # Block +10
        "agent_price_block_plus_10": checkpoint_data[10]["pool_price_usd"] if checkpoint_data.get(10) else None,
        "spread_at_block_plus_10_pct": checkpoint_data[10]["spread_pct"] if checkpoint_data.get(10) else None,
        # Block +30
        "agent_price_block_plus_30": checkpoint_data[30]["pool_price_usd"] if checkpoint_data.get(30) else None,
        "spread_at_block_plus_30_pct": checkpoint_data[30]["spread_pct"] if checkpoint_data.get(30) else None,
        # Closure block
        "blocks_until_spread_closed": blocks_until_closed,
    }


def _find_spread_close_block(
    block_number: int,
    pool_contract,
    seed_price_usd: float,
    token0_address: Optional[str],
    w3: Web3,
    max_search_blocks: int = 50,
) -> Optional[int]:
    """
    Find the first block after graduation where spread drops below 0.5%.

    Scans blocks +1 through +max_search_blocks sequentially.
    Returns None if spread is still open at max_search_blocks.
    """
    virtual_addr = w3.to_checksum_address(config.VIRTUAL_TOKEN_ADDRESS)
    is_virtual_token0 = (
        token0_address and
        w3.to_checksum_address(token0_address) == virtual_addr
    )

    for offset in range(1, max_search_blocks + 1):
        target_block = block_number + offset
        try:
            slot0 = call_with_retry(
                lambda b=target_block: pool_contract.functions.slot0().call(
                    block_identifier=b
                )
            )
            sqrt_price_x96 = slot0[0]
            raw_price = (sqrt_price_x96 / (2 ** 96)) ** 2

            if is_virtual_token0:
                pool_price_in_virtual = 1.0 / raw_price if raw_price > 0 else 0.0
            else:
                pool_price_in_virtual = raw_price

            blk = call_with_retry(lambda b=target_block: w3.eth.get_block(b))
            virtual_usd = get_virtual_price_at_timestamp(blk["timestamp"])
            pool_price_usd = pool_price_in_virtual * virtual_usd

            if seed_price_usd > 0:
                spread_pct = abs((seed_price_usd - pool_price_usd) / seed_price_usd * 100.0)
            else:
                spread_pct = 0.0

            if spread_pct < _SPREAD_CLOSED_THRESHOLD_PCT:
                return offset

        except Exception:
            # Non-fatal — just means we can't determine closure at this block.
            break

    return None  # Spread did not close within the search window.


def _null_dislocation_record() -> dict:
    """Return null values for all dislocation fields."""
    return {
        "virtual_usd_price_at_block": None,
        "seed_price_usd": None,
        "agent_price_block_0_usd": None,
        "virtual_usd_price_t0": None,
        "spread_at_graduation_pct": None,
        "agent_price_block_plus_1": None,
        "virtual_usd_price_t1": None,
        "spread_at_block_plus_1_pct": None,
        "agent_price_block_plus_3": None,
        "spread_at_block_plus_3_pct": None,
        "agent_price_block_plus_10": None,
        "spread_at_block_plus_10_pct": None,
        "agent_price_block_plus_30": None,
        "spread_at_block_plus_30_pct": None,
        "blocks_until_spread_closed": None,
    }
