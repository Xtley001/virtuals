"""
src/exit_routes/liquidity_checker.py — Stage 3c: Exit route liquidity check.

After buying the agent token at the graduation seed price, we need to sell it.
This module checks what exit routes existed at graduation time:
    1. Aerodrome pool for agent/VIRTUAL (most likely on Base)
    2. Pre-existing Uniswap V3 pool for agent token (rare but possible)

All queries use historical eth_call (archive node required) to get the exact
reserve state at the graduation block — not current state.

Minimum viable exit: Aerodrome depth > $10,000 in VIRTUAL equivalent.

Output:
    data/raw/stage3_exit_routes.csv
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from web3 import Web3

import config
from src.chain.rpc_client import get_client, call_with_retry
from src.chain.checkpoint import load_checkpoint, save_checkpoint, checkpoint_exists
from src.chain.event_fetcher import fetch_events, get_event_signature_hash
from src.price.virtual_price import get_virtual_price_at_timestamp

logger = logging.getLogger(__name__)

STAGE_NAME = "stage3_exit_routes"
OUTPUT_PATH = Path(config.RAW_DIR) / "stage3_exit_routes.csv"

# Minimum VIRTUAL liquidity (USD equivalent) to qualify as a usable exit route.
_MIN_EXIT_LIQUIDITY_USD = 10_000.0

# Uniswap V3 PoolCreated event — for detecting pre-existing secondary pools.
_POOL_CREATED_SIG = "PoolCreated(address,address,uint24,int24,address)"


def run(stage2_df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute Stage 3c: check exit route liquidity for every graduation event.

    Args:
        stage2_df: Output DataFrame from Stage 2.

    Returns:
        DataFrame with exit route columns appended.
    """
    if checkpoint_exists(STAGE_NAME):
        logger.info("Stage 3c: Loading exit route data from checkpoint.")
        data = load_checkpoint(STAGE_NAME)
        df = pd.DataFrame(data)
        logger.info("Stage 3c: Loaded %d records from checkpoint.", len(df))
        return df

    w3 = get_client()
    pool_created_topic = get_event_signature_hash(_POOL_CREATED_SIG)

    logger.info("Stage 3c: Checking exit routes for %d events.", len(stage2_df))

    records = []
    total = len(stage2_df)

    for i, row in stage2_df.iterrows():
        event_id = row["event_id"]
        block_number = int(row["block_number"])
        block_timestamp = int(row["block_timestamp"])
        agent_token_address = row["agent_token_address"]
        pool_address = row.get("uniswap_pool_address")

        if not pool_address or pd.isna(pool_address):
            records.append({**row.to_dict(), **_null_exit_record()})
            continue

        logger.debug(
            "[%d/%d] event_id=%d block=%d token=%s",
            i + 1, total, event_id, block_number, agent_token_address,
        )

        try:
            exit_rec = _check_exit_routes(
                w3=w3,
                event_id=event_id,
                block_number=block_number,
                block_timestamp=block_timestamp,
                agent_token_address=agent_token_address,
                graduation_pool_address=str(pool_address),
                pool_created_topic=pool_created_topic,
            )
            records.append({**row.to_dict(), **exit_rec})
        except Exception as exc:
            logger.warning(
                "event_id=%d: exit route check failed: %s", event_id, exc
            )
            records.append({**row.to_dict(), **_null_exit_record()})

    df = pd.DataFrame(records)
    df = df.sort_values("event_id").reset_index(drop=True)

    feasible = (df["best_exit_route"] != "none").sum()
    logger.info(
        "Stage 3c complete: %d/%d events have a viable exit route.",
        feasible, len(df),
    )
    Path(config.RAW_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    save_checkpoint(STAGE_NAME, df.to_dict(orient="records"))

    print(f"\n[Stage 3c] ✓ Exit routes checked: {feasible}/{len(df)} events have viable exit")
    print(f"[Stage 3c] Output: {OUTPUT_PATH}")
    return df


def _check_exit_routes(
    w3: Web3,
    event_id: int,
    block_number: int,
    block_timestamp: int,
    agent_token_address: str,
    graduation_pool_address: str,
    pool_created_topic: str,
) -> dict:
    """
    Check all available exit routes for one graduation event.

    Priority order:
        1. Aerodrome (agent/VIRTUAL stable or volatile pair)
        2. Pre-existing Uniswap V3 pool (any fee tier)
        3. None (arb not feasible without exit liquidity)

    Returns:
        Dict with exit route fields.
    """
    virtual_addr = w3.to_checksum_address(config.VIRTUAL_TOKEN_ADDRESS)
    agent_addr = w3.to_checksum_address(agent_token_address)

    # Fetch $VIRTUAL price at graduation block for USD conversion.
    try:
        virtual_usd = get_virtual_price_at_timestamp(block_timestamp)
    except Exception as exc:
        logger.debug("event_id=%d: could not get VIRTUAL price: %s", event_id, exc)
        virtual_usd = None

    # ── Check 1: Aerodrome pool for agent/VIRTUAL ─────────────────────────────
    aerodrome_liquidity_usd: Optional[float] = None
    aerodrome_pool_address: Optional[str] = None

    try:
        aerodrome_pool_address, aerodrome_liquidity_usd = _check_aerodrome_liquidity(
            w3=w3,
            agent_addr=agent_addr,
            virtual_addr=virtual_addr,
            block_number=block_number,
            virtual_usd=virtual_usd,
        )
    except Exception as exc:
        logger.debug(
            "event_id=%d: Aerodrome check failed: %s", event_id, exc
        )

    # ── Check 2: Pre-existing Uniswap V3 pool (created before graduation) ─────
    secondary_pool_address: Optional[str] = None
    secondary_pool_exists = False

    try:
        secondary_pool_address = _find_pre_existing_uniswap_pool(
            w3=w3,
            agent_addr=agent_addr,
            virtual_addr=virtual_addr,
            before_block=block_number,
            graduation_pool_address=graduation_pool_address,
            pool_created_topic=pool_created_topic,
        )
        secondary_pool_exists = secondary_pool_address is not None
    except Exception as exc:
        logger.debug(
            "event_id=%d: Secondary pool check failed: %s", event_id, exc
        )

    # ── Determine best exit route ─────────────────────────────────────────────
    best_exit = _select_best_exit(
        aerodrome_liquidity_usd=aerodrome_liquidity_usd,
        secondary_pool_exists=secondary_pool_exists,
    )

    # Estimate slippage on a $10k exit via the best route.
    exit_slippage_pct = _estimate_exit_slippage(
        w3=w3,
        best_exit=best_exit,
        aerodrome_pool_address=aerodrome_pool_address,
        secondary_pool_address=secondary_pool_address,
        graduation_pool_address=graduation_pool_address,
        agent_addr=agent_addr,
        virtual_addr=virtual_addr,
        block_number=block_number,
        virtual_usd=virtual_usd,
        exit_size_usd=10_000.0,
    )

    return {
        "aerodrome_pool_address": aerodrome_pool_address,
        "aerodrome_virtual_liquidity_usd": aerodrome_liquidity_usd,
        "secondary_pool_exists": secondary_pool_exists,
        "secondary_pool_address": secondary_pool_address,
        "best_exit_route": best_exit,
        "estimated_exit_slippage_10k_pct": exit_slippage_pct,
    }


def _check_aerodrome_liquidity(
    w3: Web3,
    agent_addr: str,
    virtual_addr: str,
    block_number: int,
    virtual_usd: Optional[float],
) -> tuple[Optional[str], Optional[float]]:
    """
    Look up the Aerodrome pool for agent/VIRTUAL and read its reserve depth
    at the graduation block via historical eth_call.

    Checks both stable=False (volatile) and stable=True pairs.

    Returns:
        (pool_address, virtual_liquidity_usd) or (None, None) if no pool.
    """
    aerodrome_factory = w3.eth.contract(
        address=w3.to_checksum_address(config.AERODROME_FACTORY_ADDRESS),
        abi=config.AERODROME_FACTORY_ABI,
    )

    for stable in [False, True]:
        try:
            pool_addr = call_with_retry(
                lambda s=stable: aerodrome_factory.functions.getPair(
                    agent_addr, virtual_addr, s
                ).call(block_identifier=block_number)
            )
        except Exception as exc:
            logger.debug("Aerodrome getPair(stable=%s) failed: %s", stable, exc)
            continue

        # Zero address means no pool exists.
        if not pool_addr or pool_addr == "0x" + "0" * 40:
            continue

        # Pool found — read reserves at graduation block.
        try:
            pool_contract = w3.eth.contract(
                address=w3.to_checksum_address(pool_addr),
                abi=config.AERODROME_POOL_ABI,
            )
            t0 = call_with_retry(
                lambda p=pool_contract: p.functions.token0().call(
                    block_identifier=block_number
                )
            )
            reserves = call_with_retry(
                lambda p=pool_contract: p.functions.getReserves().call(
                    block_identifier=block_number
                )
            )
            reserve0, reserve1, _ = reserves[0], reserves[1], reserves[2]

            # Identify which reserve is VIRTUAL.
            if w3.to_checksum_address(t0) == virtual_addr:
                virtual_reserve_raw = reserve0
            else:
                virtual_reserve_raw = reserve1

            virtual_reserve_human = virtual_reserve_raw / 1e18

            # Convert to USD.
            if virtual_usd and virtual_usd > 0:
                virtual_liquidity_usd = virtual_reserve_human * virtual_usd
            else:
                virtual_liquidity_usd = None

            logger.debug(
                "Aerodrome pool %s (stable=%s): %.2f VIRTUAL (≈$%.0f)",
                pool_addr, stable, virtual_reserve_human,
                virtual_liquidity_usd or 0,
            )
            return pool_addr, virtual_liquidity_usd

        except Exception as exc:
            logger.debug(
                "Aerodrome reserve read failed for pool %s: %s", pool_addr, exc
            )
            continue

    return None, None


def _find_pre_existing_uniswap_pool(
    w3: Web3,
    agent_addr: str,
    virtual_addr: str,
    before_block: int,
    graduation_pool_address: str,
    pool_created_topic: str,
) -> Optional[str]:
    """
    Search for a Uniswap V3 pool containing the agent token that was created
    BEFORE the graduation block (i.e., pre-existing secondary pool).

    Excludes the graduation pool itself (which was created AT the graduation block).

    Returns:
        Pool address string, or None if no pre-existing pool found.
    """
    # Scan from a reasonable window before graduation.
    # 7 days worth of blocks: 7 * 86400 / 2 = 302,400 blocks.
    search_from = max(0, before_block - 302_400)
    search_to = before_block - 1

    if search_from >= search_to:
        return None

    try:
        pool_logs = fetch_events(
            contract_address=config.UNISWAP_V3_FACTORY_ADDRESS,
            event_signature_hash=pool_created_topic,
            from_block=search_from,
            to_block=search_to,
        )
    except Exception as exc:
        logger.debug("Pre-existing pool scan failed: %s", exc)
        return None

    graduation_pool_lower = graduation_pool_address.lower()

    for log in pool_logs:
        # Extract token0 and token1 from indexed topics.
        topics = log.get("topics", [])
        if len(topics) < 4:
            continue

        t0 = w3.to_checksum_address("0x" + topics[1].hex()[-40:])
        t1 = w3.to_checksum_address("0x" + topics[2].hex()[-40:])

        # Check if this pool involves the agent token and VIRTUAL.
        involves_agent = (t0 == agent_addr or t1 == agent_addr)
        involves_virtual = (t0 == virtual_addr or t1 == virtual_addr)

        if not (involves_agent and involves_virtual):
            continue

        # Extract pool address from data field.
        data_hex = log.get("data", b"").hex()
        if len(data_hex) < 40:
            continue
        pool_addr = w3.to_checksum_address("0x" + data_hex[-40:])

        # Exclude the graduation pool itself.
        if pool_addr.lower() == graduation_pool_lower:
            continue

        logger.debug(
            "Found pre-existing secondary pool: %s (token0=%s, token1=%s)",
            pool_addr, t0, t1,
        )
        return pool_addr

    return None


def _select_best_exit(
    aerodrome_liquidity_usd: Optional[float],
    secondary_pool_exists: bool,
) -> str:
    """
    Select the best exit route based on available liquidity.

    Priority:
        aerodrome  → if Aerodrome depth > $10k
        uniswap_secondary → if a pre-existing Uniswap V3 pool exists
        none       → no viable exit (arb not feasible)
    """
    if aerodrome_liquidity_usd and aerodrome_liquidity_usd >= _MIN_EXIT_LIQUIDITY_USD:
        return "aerodrome"
    if secondary_pool_exists:
        return "uniswap_secondary"
    return "none"


def _estimate_exit_slippage(
    w3: Web3,
    best_exit: str,
    aerodrome_pool_address: Optional[str],
    secondary_pool_address: Optional[str],
    graduation_pool_address: str,
    agent_addr: str,
    virtual_addr: str,
    block_number: int,
    virtual_usd: Optional[float],
    exit_size_usd: float,
) -> Optional[float]:
    """
    Estimate percentage slippage on exiting `exit_size_usd` through the best exit.

    Uses the constant-product AMM formula for Aerodrome (x*y=k pools).
    For Uniswap V3, returns a rough estimate based on liquidity depth.

    Returns:
        Slippage percentage (e.g. 0.5 = 0.5%), or None if cannot be estimated.
    """
    if best_exit == "none" or not virtual_usd or virtual_usd <= 0:
        return None

    try:
        if best_exit == "aerodrome" and aerodrome_pool_address:
            return _aerodrome_slippage_estimate(
                w3=w3,
                pool_address=aerodrome_pool_address,
                virtual_addr=virtual_addr,
                block_number=block_number,
                virtual_usd=virtual_usd,
                sell_size_usd=exit_size_usd,
            )
    except Exception as exc:
        logger.debug("Slippage estimate failed for best_exit=%s: %s", best_exit, exc)

    return None


def _aerodrome_slippage_estimate(
    w3: Web3,
    pool_address: str,
    virtual_addr: str,
    block_number: int,
    virtual_usd: float,
    sell_size_usd: float,
) -> Optional[float]:
    """
    Constant-product AMM slippage estimate for Aerodrome (x*y=k).

    Formula (from constant-product AMM):
        k = x * y
        after selling delta_x: new_y = k / (x + delta_x)
        received = y - new_y
        slippage = (delta_x/x - received/y) / (delta_x/x) * 100
        Simplified: slippage ≈ delta_x / (x + delta_x) * 100  (for x*y=k)

    Source: Uniswap V2 whitepaper, Section 3 — constant product formula.
    """
    pool_contract = w3.eth.contract(
        address=w3.to_checksum_address(pool_address),
        abi=config.AERODROME_POOL_ABI,
    )
    t0 = call_with_retry(
        lambda: pool_contract.functions.token0().call(block_identifier=block_number)
    )
    reserves = call_with_retry(
        lambda: pool_contract.functions.getReserves().call(block_identifier=block_number)
    )

    r0, r1 = reserves[0] / 1e18, reserves[1] / 1e18

    # Identify VIRTUAL reserve.
    if w3.to_checksum_address(t0) == virtual_addr:
        r_virtual = r0
    else:
        r_virtual = r1

    if r_virtual <= 0 or virtual_usd <= 0:
        return None

    # Size of our sell in VIRTUAL terms.
    delta_virtual = sell_size_usd / virtual_usd

    # Slippage = delta / (reserve + delta) * 100
    # (price impact of buying delta_virtual worth of the reserve side)
    slippage_pct = delta_virtual / (r_virtual + delta_virtual) * 100.0
    return slippage_pct


def _null_exit_record() -> dict:
    """Return null values for all exit route fields."""
    return {
        "aerodrome_pool_address": None,
        "aerodrome_virtual_liquidity_usd": None,
        "secondary_pool_exists": None,
        "secondary_pool_address": None,
        "best_exit_route": "none",
        "estimated_exit_slippage_10k_pct": None,
    }
