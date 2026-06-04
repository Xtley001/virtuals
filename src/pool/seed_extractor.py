"""
src/pool/seed_extractor.py — Stage 2: Extract Uniswap V3 pool seed parameters.

For each graduation event, finds the Uniswap V3 pool created at that block and
decodes the initialization parameters: sqrtPriceX96, tick range, fee tier, and
seeded token amounts. The seed price (price at pool init) is the arb entry point.

sqrtPriceX96 → price formula (Uniswap V3 whitepaper, Section 3):
    price_token1_per_token0 = (sqrtPriceX96 / 2^96)^2

This gives the price of token0 in terms of token1.
Decimal adjustment:
    adjusted_price = raw_price * (10^decimals_token0) / (10^decimals_token1)

Output:
    data/raw/stage2_seeds.csv

Verification:
    For 5 events: look up the pool on Basescan, read the Initialize event,
    confirm sqrtPriceX96 matches.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from web3 import Web3

import config
from src.chain.rpc_client import get_client, call_with_retry
from src.chain.checkpoint import load_checkpoint, save_checkpoint, checkpoint_exists
from src.chain.event_fetcher import fetch_events, get_event_signature_hash

logger = logging.getLogger(__name__)

STAGE_NAME = "stage2_seeds"
OUTPUT_PATH = Path(config.RAW_DIR) / "stage2_seeds.csv"

# Uniswap V3 Initialize event signature.
# Source: Uniswap V3 Core — IUniswapV3PoolEvents.sol
# event Initialize(uint160 sqrtPriceX96, int24 tick)
_INIT_EVENT_SIG = "Initialize(uint160,int24)"

# Uniswap V3 PoolCreated event signature.
# Source: Uniswap V3 Core — IUniswapV3Factory.sol
# event PoolCreated(address indexed token0, address indexed token1,
#                   uint24 indexed fee, int24 tickSpacing, address pool)
_POOL_CREATED_SIG = "PoolCreated(address,address,uint24,int24,address)"

# Uniswap V3 Mint event — used to read seeded amounts.
# event Mint(address sender, address indexed owner, int24 indexed tickLower,
#            int24 indexed tickUpper, uint128 amount, uint256 amount0, uint256 amount1)
_MINT_EVENT_SIG = "Mint(address,address,int24,int24,uint128,uint256,uint256)"

# Number of blocks after graduation to look for pool creation.
_POOL_SEARCH_WINDOW = 3


def run(stage1_df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute Stage 2: extract pool seed parameters for every graduation event.

    Args:
        stage1_df: Output DataFrame from Stage 1.

    Returns:
        DataFrame with all Stage 1 columns plus pool seed fields.
    """
    if checkpoint_exists(STAGE_NAME):
        logger.info("Stage 2: Loading from checkpoint.")
        data = load_checkpoint(STAGE_NAME)
        df = pd.DataFrame(data)
        logger.info("Stage 2: Loaded %d pool seed records from checkpoint.", len(df))
        return df

    logger.info("Stage 2: Starting pool seed extraction for %d events.", len(stage1_df))

    w3 = get_client()

    # Pre-compute event topic hashes.
    init_topic = get_event_signature_hash(_INIT_EVENT_SIG)
    pool_created_topic = get_event_signature_hash(_POOL_CREATED_SIG)
    mint_topic = get_event_signature_hash(_MINT_EVENT_SIG)

    # Build the Uniswap V3 factory contract.
    uni_factory = w3.eth.contract(
        address=w3.to_checksum_address(config.UNISWAP_V3_FACTORY_ADDRESS),
        abi=config.UNISWAP_V3_FACTORY_ABI,
    )

    records = []
    total = len(stage1_df)

    for i, row in stage1_df.iterrows():
        event_id = row["event_id"]
        block_number = int(row["block_number"])
        agent_token_address = row["agent_token_address"]

        logger.debug(
            "[%d/%d] event_id=%d block=%d token=%s",
            i + 1, total, event_id, block_number, agent_token_address,
        )

        try:
            record = _extract_pool_seed(
                w3=w3,
                uni_factory=uni_factory,
                event_id=event_id,
                block_number=block_number,
                agent_token_address=agent_token_address,
                pool_created_topic=pool_created_topic,
                init_topic=init_topic,
                mint_topic=mint_topic,
            )
            if record is not None:
                records.append({**row.to_dict(), **record})
            else:
                # No pool found — record the event but mark fields as null.
                records.append({**row.to_dict(), **_null_seed_record()})
        except Exception as exc:
            logger.warning(
                "event_id=%d block=%d: pool seed extraction failed: %s",
                event_id, block_number, exc,
            )
            records.append({**row.to_dict(), **_null_seed_record()})

    df = pd.DataFrame(records)

    found = df["uniswap_pool_address"].notna().sum()
    logger.info(
        "Stage 2 complete: %d/%d events have pool seed data.",
        found, len(df),
    )

    Path(config.RAW_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    save_checkpoint(STAGE_NAME, df.to_dict(orient="records"))

    print(f"\n[Stage 2] ✓ Pool seeds extracted: {found}/{len(df)} events have pool data")
    print(f"[Stage 2] Output: {OUTPUT_PATH}")
    return df


def _extract_pool_seed(
    w3: Web3,
    uni_factory,
    event_id: int,
    block_number: int,
    agent_token_address: str,
    pool_created_topic: str,
    init_topic: str,
    mint_topic: str,
) -> Optional[dict]:
    """
    For one graduation event, find the Uniswap V3 pool and decode init params.

    Strategy:
        1. Scan blocks [graduation_block, graduation_block + _POOL_SEARCH_WINDOW]
           for PoolCreated events from the Uniswap V3 factory that involve
           the agent token or VIRTUAL.
        2. For the matching pool: read the Initialize event to get sqrtPriceX96.
        3. Read the first Mint event to get seeded amounts (amount0, amount1).
        4. Convert sqrtPriceX96 to a human-readable price.

    Returns:
        Dict with pool seed fields, or None if no matching pool found.
    """
    virtual_addr = w3.to_checksum_address(config.VIRTUAL_TOKEN_ADDRESS)
    agent_addr = w3.to_checksum_address(agent_token_address)

    # ── Step 1: Find PoolCreated for this agent token ─────────────────────────
    scan_from = block_number
    scan_to = block_number + _POOL_SEARCH_WINDOW

    pool_created_logs = fetch_events(
        contract_address=config.UNISWAP_V3_FACTORY_ADDRESS,
        event_signature_hash=pool_created_topic,
        from_block=scan_from,
        to_block=scan_to,
    )

    pool_address = None
    fee_tier = None
    token0_addr = None
    token1_addr = None

    for log in pool_created_logs:
        # Decode the PoolCreated log using the minimal factory ABI.
        factory_contract = w3.eth.contract(
            address=w3.to_checksum_address(config.UNISWAP_V3_FACTORY_ADDRESS),
            abi=config.UNISWAP_V3_FACTORY_ABI,
        )
        try:
            decoded = factory_contract.events.PoolCreated().process_log(log)
            args = decoded["args"]
        except Exception:
            # Decode topics manually as fallback.
            topics = log.get("topics", [])
            if len(topics) < 4:
                continue
            t0 = "0x" + topics[1].hex()[-40:]
            t1 = "0x" + topics[2].hex()[-40:]
            args = {
                "token0": w3.to_checksum_address(t0),
                "token1": w3.to_checksum_address(t1),
                "fee": int(topics[3].hex(), 16) if len(topics) > 3 else 0,
                "pool": "0x" + log["data"].hex()[-40:] if log.get("data") else None,
            }

        t0 = w3.to_checksum_address(args["token0"])
        t1 = w3.to_checksum_address(args["token1"])

        # Check if this pool pairs agent token with VIRTUAL.
        if (t0 == agent_addr and t1 == virtual_addr) or \
           (t0 == virtual_addr and t1 == agent_addr):
            pool_address = w3.to_checksum_address(args["pool"]) if args.get("pool") else None
            if pool_address is None:
                # Pool address is in the non-indexed part of the log data.
                # PoolCreated data field encodes (int24 tickSpacing, address pool)
                data_hex = log["data"].hex()
                # Last 40 hex chars = last 20 bytes = pool address.
                pool_address = w3.to_checksum_address("0x" + data_hex[-40:])
            fee_tier = args.get("fee", 0)
            token0_addr = t0
            token1_addr = t1
            break

    if pool_address is None:
        logger.debug(
            "event_id=%d block=%d: No VIRTUAL/agent pool found in PoolCreated events.",
            event_id, block_number,
        )
        return None

    logger.debug(
        "event_id=%d: Pool %s found (fee=%d bps, token0=%s, token1=%s)",
        event_id, pool_address, fee_tier // 100 if fee_tier else 0, token0_addr, token1_addr,
    )

    # ── Step 2: Read pool Initialize event ───────────────────────────────────
    init_logs = fetch_events(
        contract_address=pool_address,
        event_signature_hash=init_topic,
        from_block=scan_from,
        to_block=scan_to,
    )

    if not init_logs:
        logger.warning(
            "event_id=%d block=%d: No Initialize event found for pool %s.",
            event_id, block_number, pool_address,
        )
        return None

    # Decode Initialize event.
    # event Initialize(uint160 sqrtPriceX96, int24 tick)
    # Data layout: sqrtPriceX96 (32 bytes) | tick (32 bytes, signed)
    init_log = init_logs[0]
    init_data = init_log["data"].hex()
    # sqrtPriceX96 is uint160, tick is int24 — both ABI-encoded as 32-byte slots.
    sqrt_price_x96 = int(init_data[:64], 16)
    tick_raw = int(init_data[64:128], 16)
    # int24 is stored as uint256 in ABI encoding; convert to signed.
    if tick_raw >= 2**255:
        tick_raw -= 2**256
    initial_tick = tick_raw

    # ── Step 3: Read first Mint event for seeded amounts ─────────────────────
    mint_logs = fetch_events(
        contract_address=pool_address,
        event_signature_hash=mint_topic,
        from_block=scan_from,
        to_block=scan_to,
    )

    seed_virtual_amount: Optional[int] = None
    seed_agent_amount: Optional[int] = None
    tick_lower: Optional[int] = None
    tick_upper: Optional[int] = None

    if mint_logs:
        # Decode first Mint event to get seeded amounts and tick range.
        # event Mint(address sender, address indexed owner, int24 indexed tickLower,
        #            int24 indexed tickUpper, uint128 amount, uint256 amount0, uint256 amount1)
        # Indexed params: owner (topic1), tickLower (topic2), tickUpper (topic3)
        # Data: sender (32b), amount (32b), amount0 (32b), amount1 (32b)
        mint_log = mint_logs[0]
        topics = mint_log.get("topics", [])
        data_hex = mint_log["data"].hex()

        # Decode tick_lower, tick_upper from indexed topics.
        if len(topics) >= 4:
            tick_lower_raw = int(topics[2].hex(), 16)
            tick_upper_raw = int(topics[3].hex(), 16)
            # Convert to signed int24.
            if tick_lower_raw >= 2**255:
                tick_lower_raw -= 2**256
            if tick_upper_raw >= 2**255:
                tick_upper_raw -= 2**256
            tick_lower = tick_lower_raw
            tick_upper = tick_upper_raw

        # Data: (address sender=32b)(uint128 amount=32b)(uint256 amount0=32b)(uint256 amount1=32b)
        if len(data_hex) >= 256:
            # sender: bytes 0–63 (32 bytes)
            # amount: bytes 64–127 (uint128, packed in 32 bytes)
            amount0 = int(data_hex[128:192], 16)  # uint256
            amount1 = int(data_hex[192:256], 16)  # uint256

            # Determine which amount corresponds to VIRTUAL vs agent token.
            if token0_addr == virtual_addr:
                seed_virtual_amount = amount0
                seed_agent_amount = amount1
            else:
                seed_virtual_amount = amount1
                seed_agent_amount = amount0

    # ── Step 4: Compute seed price ────────────────────────────────────────────
    # Formula: price_token1_per_token0 = (sqrtPriceX96 / 2^96)^2
    # Source: Uniswap V3 Core whitepaper, Section 3 "Price Representation"
    # sqrtPriceX96 encodes sqrt(price) as a Q64.96 fixed-point number.
    raw_price = (sqrt_price_x96 / (2 ** 96)) ** 2

    # Adjust for decimals: both VIRTUAL and agent tokens use 18 decimals.
    # Since decimal_token0 == decimal_token1 == 18, no adjustment needed.
    # seed_price = amount of token1 per 1 token0 (both in human units).
    # We report seed_price_in_virtual = VIRTUAL per 1 agent token.
    virtual_decimals = 18
    agent_decimals = 18
    decimal_adjustment = (10 ** virtual_decimals) / (10 ** agent_decimals)

    if token0_addr == agent_addr:
        # token0 = agent, token1 = VIRTUAL
        # raw_price = VIRTUAL/agent (already what we want)
        seed_price_in_virtual = raw_price * decimal_adjustment
    else:
        # token0 = VIRTUAL, token1 = agent
        # raw_price = agent/VIRTUAL → invert to get VIRTUAL/agent
        if raw_price > 0:
            seed_price_in_virtual = (1.0 / raw_price) * decimal_adjustment
        else:
            logger.warning("event_id=%d: sqrtPriceX96=0 in Initialize event.", event_id)
            seed_price_in_virtual = 0.0

    return {
        "uniswap_pool_address": pool_address,
        "pool_fee_tier": fee_tier,
        "pool_token0": token0_addr,
        "pool_token1": token1_addr,
        "sqrt_price_x96_init": sqrt_price_x96,
        "initial_tick": initial_tick,
        "pool_tick_lower": tick_lower,
        "pool_tick_upper": tick_upper,
        "seed_price_in_virtual": seed_price_in_virtual,
        "seed_virtual_amount_raw": seed_virtual_amount,  # in wei (18 decimals)
        "seed_agent_amount_raw": seed_agent_amount,      # in wei (18 decimals)
        "seed_virtual_amount": (seed_virtual_amount / 1e18) if seed_virtual_amount else None,
        "seed_agent_amount": (seed_agent_amount / 1e18) if seed_agent_amount else None,
    }


def _null_seed_record() -> dict:
    """Return a dict of null values for all pool seed fields."""
    return {
        "uniswap_pool_address": None,
        "pool_fee_tier": None,
        "pool_token0": None,
        "pool_token1": None,
        "sqrt_price_x96_init": None,
        "initial_tick": None,
        "pool_tick_lower": None,
        "pool_tick_upper": None,
        "seed_price_in_virtual": None,
        "seed_virtual_amount_raw": None,
        "seed_agent_amount_raw": None,
        "seed_virtual_amount": None,
        "seed_agent_amount": None,
    }
