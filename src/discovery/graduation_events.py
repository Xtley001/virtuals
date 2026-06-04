"""
src/discovery/graduation_events.py — Stage 1: Discover every historical graduation event.

What this does:
    Queries the Virtuals Protocol factory contract for all graduation (agent launch)
    events from the protocol's deployment block to the current block. Each event
    represents one agent token graduating from the bonding curve to a permanent
    Uniswap V3 liquidity pool.

Output:
    data/raw/stage1_graduations.csv with columns:
        event_id, block_number, block_timestamp, agent_token_address,
        agent_token_name, graduation_tx_hash

Verification:
    After running, 5 random graduation_tx_hash values should be confirmed
    on basescan.org before Stage 2 begins.
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

STAGE_NAME = "stage1_graduations"
OUTPUT_PATH = Path(config.RAW_DIR) / "stage1_graduations.csv"


def run() -> pd.DataFrame:
    """
    Execute Stage 1: discover all graduation events.

    If a valid checkpoint exists, loads it instead of re-scanning the chain.
    Always returns a DataFrame with the stage1 output schema.

    Returns:
        DataFrame with columns: event_id, block_number, block_timestamp,
        agent_token_address, agent_token_name, graduation_tx_hash, uniswap_pool_address

    Raises:
        EnvironmentError: If VIRTUALS_FACTORY_ADDRESS is not set.
        RuntimeError:     On unrecoverable RPC failures.
    """
    if checkpoint_exists(STAGE_NAME):
        logger.info("Stage 1: Loading from checkpoint.")
        data = load_checkpoint(STAGE_NAME)
        df = pd.DataFrame(data)
        logger.info("Stage 1: Loaded %d graduation events from checkpoint.", len(df))
        return df

    logger.info("Stage 1: Starting graduation event discovery.")
    logger.info("Factory address: %s", config.VIRTUALS_FACTORY_ADDRESS)
    logger.info("Scan start block: %d", config.GRADUATION_START_BLOCK)

    w3 = get_client()
    current_block = call_with_retry(lambda: w3.eth.block_number)
    logger.info("Current block: %d", current_block)

    # ── Fetch factory ABI from Basescan ──────────────────────────────────────
    factory_abi = _fetch_factory_abi(config.VIRTUALS_FACTORY_ADDRESS)

    # ── Identify graduation event from ABI ───────────────────────────────────
    event_name, event_abi = _find_graduation_event(factory_abi)
    logger.info("Using graduation event: %s", event_name)

    # Build event signature for topic filter.
    # Signature format: EventName(type1,type2,...) — no spaces, no parameter names.
    param_types = ",".join(inp["type"] for inp in event_abi.get("inputs", []))
    event_signature = f"{event_name}({param_types})"
    event_topic = get_event_signature_hash(event_signature)
    logger.info("Event signature: %s → topic: %s", event_signature, event_topic)

    # ── Scan for graduation events ────────────────────────────────────────────
    raw_logs = fetch_events(
        contract_address=config.VIRTUALS_FACTORY_ADDRESS,
        event_signature_hash=event_topic,
        from_block=config.GRADUATION_START_BLOCK,
        to_block=current_block,
    )

    if not raw_logs:
        logger.error(
            "No graduation events found between blocks %d and %d. "
            "Check VIRTUALS_FACTORY_ADDRESS and GRADUATION_START_BLOCK in config.py / .env.",
            config.GRADUATION_START_BLOCK,
            current_block,
        )
        raise RuntimeError(
            "Stage 1 failed: no graduation events found. "
            "Verify the factory address emits the expected event."
        )

    logger.info("Found %d raw graduation log entries. Processing...", len(raw_logs))

    # ── Decode and enrich each event ─────────────────────────────────────────
    factory_contract = w3.eth.contract(
        address=w3.to_checksum_address(config.VIRTUALS_FACTORY_ADDRESS),
        abi=factory_abi,
    )

    records = []
    for idx, log in enumerate(raw_logs):
        try:
            record = _process_log(w3, factory_contract, event_name, log, idx)
            if record is not None:
                records.append(record)
        except Exception as exc:
            logger.warning(
                "Failed to process log at block %s, tx %s: %s",
                log.get("blockNumber"),
                log.get("transactionHash", b"").hex() if log.get("transactionHash") else "unknown",
                exc,
            )
            continue

    if not records:
        raise RuntimeError(
            "Stage 1 failed: all log entries failed to decode. "
            "The event ABI may not match the deployed contract."
        )

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["graduation_tx_hash"])
    df = df.sort_values("block_number").reset_index(drop=True)
    df["event_id"] = range(1, len(df) + 1)

    # Reorder columns.
    df = df[[
        "event_id", "block_number", "block_timestamp",
        "agent_token_address", "agent_token_name", "graduation_tx_hash",
    ]]

    # ── Save outputs ──────────────────────────────────────────────────────────
    Path(config.RAW_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    save_checkpoint(STAGE_NAME, df.to_dict(orient="records"))

    # ── Summary ───────────────────────────────────────────────────────────────
    first_ts = pd.to_datetime(df["block_timestamp"].min(), unit="s")
    last_ts = pd.to_datetime(df["block_timestamp"].max(), unit="s")
    logger.info(
        "Stage 1 complete: %d graduation events | Date range: %s → %s",
        len(df),
        first_ts.strftime("%Y-%m-%d"),
        last_ts.strftime("%Y-%m-%d"),
    )
    print(f"\n[Stage 1] ✓ {len(df)} graduation events | "
          f"{first_ts.strftime('%Y-%m-%d')} → {last_ts.strftime('%Y-%m-%d')}")
    print(f"[Stage 1] Output: {OUTPUT_PATH}")

    return df


def _fetch_factory_abi(factory_address: str) -> list:
    """
    Fetch the verified ABI for the factory contract from Basescan API.

    The ABI determines which event names exist and their parameter types.
    We cannot hardcode the ABI because the factory is upgradeable.

    Args:
        factory_address: Checksummed contract address.

    Returns:
        List of ABI entries.

    Raises:
        RuntimeError: If Basescan returns an error or the contract is unverified.
    """
    import requests

    # Etherscan V2 unified endpoint — same key works, chainid selects the network.
    # Basescan V1 (api.basescan.org/api) is deprecated as of 2025.
    url = "https://api.etherscan.io/v2/api"
    params = {
        "chainid": config.BASE_CHAIN_ID,
        "module": "contract",
        "action": "getabi",
        "address": factory_address,
        "apikey": config.BASESCAN_API_KEY,
    }

    for attempt in range(1, config.RPC_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "1":
                message = data.get("message", "unknown")
                result = data.get("result", "")
                if "not verified" in str(result).lower() or "not found" in str(result).lower():
                    raise RuntimeError(
                        f"Factory contract {factory_address} is not verified on Basescan.\n"
                        "This is a hard requirement — we cannot trust an unverified ABI.\n"
                        "Confirm VIRTUALS_FACTORY_ADDRESS in .env points to the correct "
                        "verified contract."
                    )
                raise RuntimeError(
                    f"Basescan ABI fetch failed: status={data.get('status')!r}, "
                    f"message={message!r}, result={result!r}"
                )

            import json
            abi = json.loads(data["result"])
            logger.info(
                "Fetched ABI for %s: %d entries.", factory_address, len(abi)
            )
            return abi

        except RuntimeError:
            raise
        except requests.RequestException as exc:
            if attempt < config.RPC_MAX_RETRIES:
                logger.warning(
                    "Basescan ABI fetch attempt %d/%d failed: %s. Retrying...",
                    attempt, config.RPC_MAX_RETRIES, exc,
                )
                time.sleep(config.RPC_RETRY_DELAY_SECONDS * attempt)
            else:
                raise RuntimeError(
                    f"Failed to fetch ABI from Basescan after {config.RPC_MAX_RETRIES} "
                    f"attempts: {exc}"
                ) from exc

    raise RuntimeError("Unreachable: ABI fetch loop exhausted without raising.")


def _find_graduation_event(abi: list) -> tuple[str, dict]:
    """
    Identify the graduation event from the factory ABI.

    Searches for known event names in priority order. The factory may use
    different naming conventions across protocol versions.

    Args:
        abi: Full contract ABI.

    Returns:
        (event_name, event_abi_entry) for the first matching event.

    Raises:
        RuntimeError: If no graduation event is found in the ABI.
    """
    # Candidate event names in priority order, based on known Virtuals Protocol
    # contract versions. The operator can override via GRADUATION_EVENT_NAME env var.
    candidates = [
        config.GRADUATION_EVENT_NAME,  # Operator override (default: "Launched")
        "Launched",
        "TokenGraduated",
        "AgentCreated",
        "VirtualCreated",
        "Graduated",
    ]

    events_in_abi = {
        entry["name"]: entry
        for entry in abi
        if entry.get("type") == "event"
    }

    logger.debug("Events found in factory ABI: %s", list(events_in_abi.keys()))

    for candidate in candidates:
        if candidate in events_in_abi:
            return candidate, events_in_abi[candidate]

    raise RuntimeError(
        f"No graduation event found in factory ABI.\n"
        f"Checked candidates: {candidates}\n"
        f"Events present in ABI: {list(events_in_abi.keys())}\n"
        "Set GRADUATION_EVENT_NAME in .env to match the correct event name."
    )


def _process_log(
    w3: Web3,
    factory_contract,
    event_name: str,
    log: dict,
    idx: int,
) -> Optional[dict]:
    """
    Decode a single raw log entry into a graduation record.

    Fetches block timestamp via eth_getBlock. This is a per-log RPC call
    but is unavoidable — timestamps are not in the log itself.

    Returns:
        Dict with graduation event fields, or None if decoding fails irrecoverably.
    """
    tx_hash = log["transactionHash"].hex()
    block_number = log["blockNumber"]

    # Decode the event log using the contract's ABI.
    try:
        event_obj = getattr(factory_contract.events, event_name)
        decoded = event_obj().process_log(log)
        args = decoded.get("args", {})
    except Exception as exc:
        logger.warning(
            "Failed to decode %s event at block %d, tx %s: %s",
            event_name, block_number, tx_hash, exc,
        )
        return None

    # Extract agent token address from decoded args.
    # Field names vary across contract versions; try all known names.
    agent_token_address = (
        args.get("token")
        or args.get("tokenAddress")
        or args.get("agentToken")
        or args.get("virtualToken")
        or _extract_address_from_log_topics(log, w3)
    )
    if not agent_token_address:
        logger.warning(
            "Could not extract agent_token_address from event at block %d, tx %s. "
            "Args: %s",
            block_number, tx_hash, dict(args),
        )
        return None

    agent_token_address = w3.to_checksum_address(agent_token_address)

    # Fetch block timestamp.
    block = call_with_retry(lambda bn=block_number: w3.eth.get_block(bn))
    block_timestamp = block["timestamp"]

    # Fetch agent token name from ERC-20 contract.
    agent_token_name = _get_token_name(w3, agent_token_address)

    return {
        "block_number": block_number,
        "block_timestamp": block_timestamp,
        "agent_token_address": agent_token_address,
        "agent_token_name": agent_token_name,
        "graduation_tx_hash": tx_hash,
    }


def _extract_address_from_log_topics(log: dict, w3: Web3) -> Optional[str]:
    """
    Fallback: extract an address from log topics when ABI decoding fails.
    Topic[1] is often the primary indexed parameter (token address).
    """
    topics = log.get("topics", [])
    if len(topics) >= 2:
        raw = topics[1]
        if hasattr(raw, "hex"):
            raw = raw.hex()
        # Ethereum addresses are padded to 32 bytes in topics.
        # Last 20 bytes = address.
        addr_hex = "0x" + raw[-40:]
        try:
            return w3.to_checksum_address(addr_hex)
        except Exception:
            pass
    return None


def _get_token_name(w3: Web3, token_address: str) -> str:
    """
    Fetch the name() string from an ERC-20 contract.
    Returns "UNKNOWN" on failure (does not raise — name is informational only).
    """
    try:
        contract = w3.eth.contract(
            address=token_address,
            abi=config.ERC20_MINIMAL_ABI,
        )
        return call_with_retry(lambda: contract.functions.name().call())
    except Exception as exc:
        logger.debug(
            "Could not fetch name for token %s: %s", token_address, exc
        )
        return "UNKNOWN"