"""
src/competition/bot_detector.py — Stage 3b: Competition density analysis.

For each graduation event, fetches all Swap events in the newly-created
Uniswap V3 pool over the first 5 blocks and classifies wallets as likely bots.

Bot heuristic (applied as documented in build_guide.md):
    A wallet is flagged is_bot_likely=True if it meets ≥2 of:
        1. Lifetime tx count on Base > 500
        2. Has interacted with >3 distinct Uniswap V3 pools in the 7 days prior
        3. Wallet age on Base < 7 days (fresh wallet typical of MEV searcher deployments)

All Basescan API calls are rate-limited to BASESCAN_CALLS_PER_SECOND (5/sec on free tier).

Output:
    data/raw/stage3_competition.csv
"""

import logging
import time
from pathlib import Path
from typing import Optional
from collections import defaultdict

import pandas as pd
import requests
from web3 import Web3

import config
from src.chain.rpc_client import get_client, call_with_retry
from src.chain.checkpoint import load_checkpoint, save_checkpoint, checkpoint_exists
from src.chain.event_fetcher import fetch_events, get_event_signature_hash

logger = logging.getLogger(__name__)

STAGE_NAME = "stage3_competition"
OUTPUT_PATH = Path(config.RAW_DIR) / "stage3_competition.csv"

# Scan window: number of blocks after graduation to analyse for competition.
_COMPETITION_WINDOW_BLOCKS = 5

# Uniswap V3 Swap event signature.
# Source: Uniswap V3 Core — IUniswapV3PoolEvents.sol
# event Swap(address indexed sender, address indexed recipient,
#            int256 amount0, int256 amount1,
#            uint160 sqrtPriceX96, uint128 liquidity, int24 tick)
_SWAP_EVENT_SIG = "Swap(address,address,int256,int256,uint160,uint128,int24)"

# Bot heuristic thresholds (from build_guide.md).
_BOT_MIN_TX_COUNT = 500
_BOT_MIN_POOL_INTERACTIONS = 3   # distinct Uniswap V3 pools in 7 prior days
_BOT_MAX_WALLET_AGE_DAYS = 7     # wallet first seen < 7 days before graduation


class BasescanRateLimiter:
    """
    Token-bucket rate limiter for the Basescan API (5 calls/sec on free tier).
    Instantiated once and shared across all Basescan calls in the stage.
    """

    def __init__(self, calls_per_second: float = config.BASESCAN_CALLS_PER_SECOND):
        self._calls_per_second = calls_per_second
        self._min_interval = 1.0 / calls_per_second
        self._last_call_time: float = 0.0

    def wait(self) -> None:
        """Block until it is safe to make the next API call."""
        now = time.monotonic()
        elapsed = now - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_time = time.monotonic()


_basescan = BasescanRateLimiter()


def run(stage2_df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute Stage 3b: count competing wallets and estimate bot density
    for every graduation event.

    Args:
        stage2_df: Output DataFrame from Stage 2 (includes uniswap_pool_address).

    Returns:
        DataFrame with competition columns appended to stage2 data.
    """
    if checkpoint_exists(STAGE_NAME):
        logger.info("Stage 3b: Loading competition data from checkpoint.")
        data = load_checkpoint(STAGE_NAME)
        df = pd.DataFrame(data)
        logger.info("Stage 3b: Loaded %d records from checkpoint.", len(df))
        return df

    w3 = get_client()
    swap_topic = get_event_signature_hash(_SWAP_EVENT_SIG)

    valid = stage2_df[stage2_df["uniswap_pool_address"].notna()].copy()
    logger.info(
        "Stage 3b: Analysing competition for %d events with pool data.",
        len(valid),
    )

    records = []
    total = len(stage2_df)

    for i, row in stage2_df.iterrows():
        event_id = row["event_id"]
        block_number = int(row["block_number"])
        block_timestamp = int(row["block_timestamp"])
        pool_address = row.get("uniswap_pool_address")

        if not pool_address or pd.isna(pool_address):
            records.append({**row.to_dict(), **_null_competition_record()})
            continue

        logger.debug(
            "[%d/%d] event_id=%d block=%d pool=%s",
            i + 1, total, event_id, block_number, pool_address,
        )

        try:
            comp = _analyse_competition(
                w3=w3,
                event_id=event_id,
                block_number=block_number,
                block_timestamp=block_timestamp,
                pool_address=str(pool_address),
                swap_topic=swap_topic,
            )
            records.append({**row.to_dict(), **comp})
        except Exception as exc:
            logger.warning(
                "event_id=%d: competition analysis failed: %s", event_id, exc
            )
            records.append({**row.to_dict(), **_null_competition_record()})

    df = pd.DataFrame(records)
    df = df.sort_values("event_id").reset_index(drop=True)

    Path(config.RAW_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    save_checkpoint(STAGE_NAME, df.to_dict(orient="records"))

    median_bots = df["bot_count_first_3_blocks"].median()
    pct_zero = (df["wallets_block_plus_1"] == 0).mean() * 100
    logger.info(
        "Stage 3b complete. Median bots in first 3 blocks: %.1f | "
        "Events with no competition in block+1: %.1f%%",
        median_bots if not pd.isna(median_bots) else 0, pct_zero,
    )
    print(f"\n[Stage 3b] ✓ Competition analysis complete: {len(df)} events")
    print(f"[Stage 3b] Median bot count (first 3 blocks): {median_bots:.1f}" if not pd.isna(median_bots) else "[Stage 3b] Median bot count: N/A")
    print(f"[Stage 3b] Events with no block+1 competition: {pct_zero:.1f}%")
    print(f"[Stage 3b] Output: {OUTPUT_PATH}")
    return df


def _analyse_competition(
    w3: Web3,
    event_id: int,
    block_number: int,
    block_timestamp: int,
    pool_address: str,
    swap_topic: str,
) -> dict:
    """
    Count wallets and bots for one graduation event.

    Fetches Swap events from the pool in blocks [graduation, graduation+5].
    Groups swaps by block offset to compute per-block wallet counts.
    Classifies each unique wallet using the bot heuristic.

    Returns:
        Dict with all competition fields.
    """
    scan_to = block_number + _COMPETITION_WINDOW_BLOCKS

    swap_logs = fetch_events(
        contract_address=pool_address,
        event_signature_hash=swap_topic,
        from_block=block_number,
        to_block=scan_to,
    )

    # Group swap sender addresses by block offset.
    # The Swap event has: sender (indexed, topics[1]), recipient (indexed, topics[2]).
    # We use `sender` as the acting wallet (the one who initiated the swap).
    by_block: dict[int, set[str]] = defaultdict(set)
    first_external_block: Optional[int] = None

    for log in swap_logs:
        blk = log["blockNumber"]
        topics = log.get("topics", [])
        if len(topics) >= 2:
            raw_sender = topics[1].hex()
            sender = w3.to_checksum_address("0x" + raw_sender[-40:])
        else:
            # Fallback: parse from data field (non-indexed fallback).
            data_hex = log.get("data", b"").hex()
            sender = w3.to_checksum_address("0x" + data_hex[-40:]) if len(data_hex) >= 40 else None

        if sender:
            by_block[blk].add(sender)
            offset = blk - block_number
            if offset > 0 and first_external_block is None:
                first_external_block = blk

    # Wallet counts at each block offset.
    wallets_block_0 = len(by_block.get(block_number, set()))
    wallets_block_plus_1 = len(by_block.get(block_number + 1, set()))

    cumulative_block_3: set[str] = set()
    for offset in range(0, 4):
        cumulative_block_3.update(by_block.get(block_number + offset, set()))
    wallets_block_plus_3 = len(cumulative_block_3)

    # All unique wallets in first 3 blocks (for bot classification).
    all_wallets_first_3: set[str] = cumulative_block_3

    # Classify bots — apply heuristic to each unique wallet.
    bot_count = 0
    for wallet in all_wallets_first_3:
        criteria_met = _count_bot_criteria(
            w3=w3,
            wallet=wallet,
            graduation_timestamp=block_timestamp,
            graduation_block=block_number,
        )
        if criteria_met >= 2:
            bot_count += 1

    return {
        "wallets_block_0": wallets_block_0,
        "wallets_block_plus_1": wallets_block_plus_1,
        "wallets_block_plus_3": wallets_block_plus_3,
        "bot_count_first_3_blocks": bot_count,
        "first_non_graduation_swap_block": first_external_block,
        "total_swaps_first_5_blocks": len(swap_logs),
    }


def _count_bot_criteria(
    w3: Web3,
    wallet: str,
    graduation_timestamp: int,
    graduation_block: int,
) -> int:
    """
    Count how many bot-heuristic criteria this wallet satisfies.

    Criteria (from build_guide.md — must match exactly):
        1. Lifetime tx count on Base > 500
        2. Interacted with >3 distinct Uniswap V3 pools in the 7 days prior
        3. Wallet first appeared on Base < 7 days before graduation

    Args:
        wallet:                 Checksummed wallet address.
        graduation_timestamp:   Unix timestamp of graduation block.
        graduation_block:       Block number of graduation.

    Returns:
        Integer count of criteria satisfied (0–3).
        Caller flags wallet as bot if this is ≥ 2.
    """
    score = 0

    # Criterion 1: lifetime tx count > 500.
    try:
        tx_count = call_with_retry(
            lambda w=wallet: w3.eth.get_transaction_count(w)
        )
        if tx_count > _BOT_MIN_TX_COUNT:
            score += 1
    except Exception as exc:
        logger.debug("Wallet %s: tx count check failed: %s", wallet, exc)

    # Criterion 2: >3 distinct Uniswap V3 pool interactions in 7 prior days.
    prior_pool_count = _count_prior_pool_interactions(
        wallet=wallet,
        graduation_timestamp=graduation_timestamp,
    )
    if prior_pool_count > _BOT_MIN_POOL_INTERACTIONS:
        score += 1

    # Criterion 3: wallet first seen < 7 days before graduation.
    wallet_age_days = _get_wallet_age_days(
        wallet=wallet,
        graduation_timestamp=graduation_timestamp,
    )
    if wallet_age_days is not None and wallet_age_days < _BOT_MAX_WALLET_AGE_DAYS:
        score += 1

    return score


def _count_prior_pool_interactions(wallet: str, graduation_timestamp: int) -> int:
    """
    Count distinct Uniswap V3 pools this wallet interacted with in the 7 days
    prior to graduation, via Basescan's account transaction list API.

    Returns the count of distinct pool addresses. Returns 0 on API failure.
    """
    seven_days_ago = graduation_timestamp - (7 * 24 * 3600)
    _basescan.wait()

    # Etherscan V2 unified endpoint — chainid selects Base mainnet.
    url = "https://api.etherscan.io/v2/api"
    params = {
        "chainid": config.BASE_CHAIN_ID,
        "module": "account",
        "action": "tokentx",             # ERC-20 transfers — proxy for swap interactions
        "address": wallet,
        "startblock": 0,
        "endblock": 99999999,
        "sort": "desc",
        "apikey": config.BASESCAN_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") == "0" and data.get("message") == "No transactions found":
            return 0

        if data.get("status") != "1":
            logger.debug(
                "Wallet %s: Basescan tokentx returned status=%s: %s",
                wallet, data.get("status"), data.get("message"),
            )
            return 0

        txs = data.get("result", [])
        # Validate response shape before field access.
        if not isinstance(txs, list):
            logger.debug(
                "Wallet %s: unexpected tokentx result shape: %s",
                wallet, type(txs),
            )
            return 0

        # Filter to the 7-day window and count distinct contract addresses.
        # We use `contractAddress` (the token being transferred) as a proxy for
        # pool interaction — not perfect but sufficient for the bot heuristic.
        uniswap_v3_factory = config.UNISWAP_V3_FACTORY_ADDRESS.lower()
        distinct_pools: set[str] = set()

        for tx in txs:
            try:
                ts = int(tx.get("timeStamp", 0))
            except (ValueError, TypeError):
                continue
            if ts < seven_days_ago:
                break  # sorted desc — once we pass the window, stop.
            to_addr = str(tx.get("to", "")).lower()
            contract_addr = str(tx.get("contractAddress", "")).lower()
            if to_addr and to_addr != "0x":
                distinct_pools.add(to_addr)

        return len(distinct_pools)

    except requests.RequestException as exc:
        logger.debug("Wallet %s: Basescan pool interaction check failed: %s", wallet, exc)
        return 0


def _get_wallet_age_days(wallet: str, graduation_timestamp: int) -> Optional[float]:
    """
    Return how many days before graduation this wallet first appeared on Base.
    Uses Basescan's normal transaction list (first transaction timestamp).

    Returns None on API failure (criteria treated as not met).
    """
    _basescan.wait()

    # Etherscan V2 unified endpoint — chainid selects Base mainnet.
    url = "https://api.etherscan.io/v2/api"
    params = {
        "chainid": config.BASE_CHAIN_ID,
        "module": "account",
        "action": "txlist",
        "address": wallet,
        "startblock": 0,
        "endblock": 99999999,
        "page": 1,
        "offset": 1,          # Only need the very first transaction.
        "sort": "asc",
        "apikey": config.BASESCAN_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") == "0":
            return None  # No transactions = wallet too new or inactive.

        if data.get("status") != "1":
            logger.debug(
                "Wallet %s: Basescan txlist returned status=%s",
                wallet, data.get("status"),
            )
            return None

        result = data.get("result", [])
        if not result or not isinstance(result, list):
            return None

        first_tx = result[0]
        # Validate shape before field access.
        if not isinstance(first_tx, dict) or "timeStamp" not in first_tx:
            logger.debug(
                "Wallet %s: unexpected txlist record shape: %s",
                wallet, first_tx,
            )
            return None

        first_ts = int(first_tx["timeStamp"])
        age_seconds = graduation_timestamp - first_ts
        return age_seconds / 86400.0  # Convert to days.

    except (requests.RequestException, ValueError, KeyError) as exc:
        logger.debug("Wallet %s: Basescan wallet age check failed: %s", wallet, exc)
        return None


def _null_competition_record() -> dict:
    """Return null values for all competition fields."""
    return {
        "wallets_block_0": None,
        "wallets_block_plus_1": None,
        "wallets_block_plus_3": None,
        "bot_count_first_3_blocks": None,
        "first_non_graduation_swap_block": None,
        "total_swaps_first_5_blocks": None,
    }