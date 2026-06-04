"""
src/chain/event_fetcher.py — Paginated eth_getLogs abstraction.

This module provides the foundation for all on-chain event discovery.
It handles:
    - Automatic pagination across the full block range in BLOCKS_PER_BATCH chunks.
    - Progress bars via tqdm so long runs aren't silent.
    - Per-batch retry with backoff before raising.
    - Returns raw log objects (decoding happens in the caller module).

Why not use web3.py contract.events.X.get_logs()?
    The high-level API does not support full pagination transparently across
    the entire chain history. We need explicit batch control to stay within
    RPC provider limits and to checkpoint progress.
"""

import logging
import time
from typing import Any, Optional

from tqdm import tqdm
from web3.types import LogReceipt

import config
from src.chain.rpc_client import get_client, call_with_retry

logger = logging.getLogger(__name__)


def fetch_events(
    contract_address: str,
    event_signature_hash: str,
    from_block: int,
    to_block: Optional[int] = None,
    additional_topics: Optional[list] = None,
) -> list[LogReceipt]:
    """
    Fetch all matching event logs from `from_block` to `to_block` in batches.

    Args:
        contract_address:      The contract address to filter logs on.
                               Set to None to scan all addresses (for factory events).
        event_signature_hash:  Keccak256 hash of the event signature, e.g.
                               web3.keccak(text="PoolCreated(address,address,uint24,int24,address)").hex()
        from_block:            Starting block number (inclusive).
        to_block:              Ending block number (inclusive). Defaults to current block.
        additional_topics:     Optional list of additional topic filters (topics[1], topics[2], ...).
                               Each entry is either a single hex string or a list of hex strings (OR filter).

    Returns:
        List of raw LogReceipt dicts with keys:
            blockNumber, transactionHash, logIndex, data, topics, address, etc.

    Raises:
        ValueError:   If from_block > to_block.
        RuntimeError: If a batch fails after all retries.
    """
    w3 = get_client()

    if to_block is None:
        to_block = call_with_retry(lambda: w3.eth.block_number)
        logger.debug("to_block defaulted to current block: %d", to_block)

    if from_block > to_block:
        raise ValueError(
            f"from_block ({from_block}) > to_block ({to_block}). "
            "Nothing to scan."
        )

    total_blocks = to_block - from_block + 1
    total_batches = (total_blocks + config.BLOCKS_PER_BATCH - 1) // config.BLOCKS_PER_BATCH

    logger.info(
        "Scanning blocks %d–%d (%d blocks, %d batches of %d).",
        from_block,
        to_block,
        total_blocks,
        total_batches,
        config.BLOCKS_PER_BATCH,
    )

    # Build the topics filter.
    # topics[0] is always the event signature hash.
    topics: list = [event_signature_hash]
    if additional_topics:
        topics.extend(additional_topics)

    # Build filter params — contract_address is optional.
    def _build_filter(batch_from: int, batch_to: int) -> dict:
        params: dict[str, Any] = {
            "fromBlock": batch_from,
            "toBlock": batch_to,
            "topics": topics,
        }
        if contract_address is not None:
            params["address"] = w3.to_checksum_address(contract_address)
        return params

    all_logs: list[LogReceipt] = []
    batch_start = from_block

    with tqdm(total=total_blocks, unit="blocks", desc="Scanning blocks", ncols=100) as pbar:
        while batch_start <= to_block:
            batch_end = min(batch_start + config.BLOCKS_PER_BATCH - 1, to_block)

            # Per-batch retry loop — logs the failing batch clearly.
            batch_logs = _fetch_batch_with_retry(w3, _build_filter(batch_start, batch_end))
            all_logs.extend(batch_logs)

            batch_size = batch_end - batch_start + 1
            pbar.update(batch_size)

            if batch_logs:
                logger.debug(
                    "Batch %d–%d: %d logs found (total so far: %d).",
                    batch_start,
                    batch_end,
                    len(batch_logs),
                    len(all_logs),
                )

            batch_start = batch_end + 1

    # Deduplicate by (transactionHash, logIndex) — defensive against RPC
    # returning duplicates at batch boundaries.
    before_dedup = len(all_logs)
    seen: set[tuple] = set()
    unique_logs = []
    for log in all_logs:
        key = (log["transactionHash"].hex(), log["logIndex"])
        if key not in seen:
            seen.add(key)
            unique_logs.append(log)

    if len(unique_logs) != before_dedup:
        logger.warning(
            "Removed %d duplicate log entries (before: %d, after: %d).",
            before_dedup - len(unique_logs),
            before_dedup,
            len(unique_logs),
        )

    logger.info(
        "Event scan complete. Total unique logs: %d across %d blocks.",
        len(unique_logs),
        total_blocks,
    )
    return unique_logs


def _fetch_batch_with_retry(w3, filter_params: dict) -> list[LogReceipt]:
    """
    Fetch a single batch of logs with retry on transient failures.

    Args:
        w3:            Web3 instance.
        filter_params: eth_getLogs filter dict.

    Returns:
        List of log entries (may be empty if no matching events in range).

    Raises:
        RuntimeError: If all retries are exhausted for this batch.
    """
    delay = config.RPC_RETRY_DELAY_SECONDS
    last_exc: Optional[Exception] = None

    for attempt in range(1, config.RPC_MAX_RETRIES + 1):
        try:
            logs = w3.eth.get_logs(filter_params)
            return list(logs)
        except Exception as exc:
            last_exc = exc
            error_str = str(exc).lower()

            # Detect "block range too large" errors — reduce batch and retry.
            if "block range" in error_str or "limit" in error_str or "too large" in error_str:
                logger.warning(
                    "RPC rejected block range %d–%d (too large?): %s. "
                    "Consider reducing BLOCKS_PER_BATCH in config.py.",
                    filter_params["fromBlock"],
                    filter_params["toBlock"],
                    exc,
                )
                raise RuntimeError(
                    f"Block range {filter_params['fromBlock']}–{filter_params['toBlock']} "
                    f"rejected by RPC. Reduce BLOCKS_PER_BATCH in config.py.\n"
                    f"Error: {exc}"
                ) from exc

            logger.warning(
                "Batch %d–%d fetch attempt %d/%d failed: %s. Retrying in %.1fs...",
                filter_params["fromBlock"],
                filter_params["toBlock"],
                attempt,
                config.RPC_MAX_RETRIES,
                exc,
                delay,
            )
            if attempt < config.RPC_MAX_RETRIES:
                time.sleep(delay)
                delay *= 2

    raise RuntimeError(
        f"Batch {filter_params['fromBlock']}–{filter_params['toBlock']} failed after "
        f"{config.RPC_MAX_RETRIES} attempts. Last error: {last_exc}"
    ) from last_exc


def get_event_signature_hash(event_signature: str) -> str:
    """
    Compute keccak256 of an event signature string.

    Args:
        event_signature: Human-readable event signature,
                         e.g. "PoolCreated(address,address,uint24,int24,address)"

    Returns:
        Hex string of the keccak256 hash (0x-prefixed).
    """
    w3 = get_client()
    return w3.keccak(text=event_signature).hex()
