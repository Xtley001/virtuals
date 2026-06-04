"""
src/chain/event_fetcher.py — Paginated eth_getLogs abstraction.

This module provides the foundation for all on-chain event discovery.
It handles:
    - Automatic pagination across the full block range in BLOCKS_PER_BATCH chunks.
    - Progress bars via tqdm so long runs aren't silent.
    - Per-batch retry with backoff before raising.
    - Adaptive batch-size halving on HTTP 400 / block-range-too-large errors.
    - Returns raw log objects (decoding happens in the caller module).

Why not use web3.py contract.events.X.get_logs()?
    The high-level API does not support full pagination transparently across
    the entire chain history. We need explicit batch control to stay within
    RPC provider limits and to checkpoint progress.

Fix (2026-06-04):
    Alchemy returns a plain HTTP 400 Bad Request when the requested block
    range is too old / too large — the response body does NOT contain
    "block range", "limit", or "too large", so the original string-match
    guard was silently missing it. The fix:

      1.  Detect HTTP 400 / 413 / 429 status codes directly via the
          requests.exceptions.HTTPError status_code attribute.
      2.  On a 400 that looks range-related, halve EFFECTIVE_BATCH_SIZE and
          retry the same window instead of crashing immediately — this lets
          the scan recover automatically from provider-specific caps.
      3.  Surface a clear, actionable error message when the minimum batch
          size (MIN_BATCH_SIZE) is reached so the operator knows to lower
          BLOCKS_PER_BATCH in config.py or .env.
"""

import logging
import time
from typing import Any, Optional

import requests.exceptions
from tqdm import tqdm
from web3.types import LogReceipt

import config
from src.chain.rpc_client import get_client, call_with_retry

logger = logging.getLogger(__name__)

# Never split below this — avoids infinite halving on genuine auth/infra 400s.
MIN_BATCH_SIZE: int = 50


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
    effective_batch = config.BLOCKS_PER_BATCH
    total_batches = (total_blocks + effective_batch - 1) // effective_batch

    logger.info(
        "Scanning blocks %d–%d (%d blocks, %d batches of %d).",
        from_block,
        to_block,
        total_blocks,
        total_batches,
        effective_batch,
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
            batch_end = min(batch_start + effective_batch - 1, to_block)

            try:
                batch_logs = _fetch_batch_with_retry(w3, _build_filter(batch_start, batch_end))
            except _BlockRangeTooLargeError as exc:
                # RPC rejected this range — halve effective batch and retry window.
                new_batch = max(effective_batch // 2, MIN_BATCH_SIZE)
                if new_batch == effective_batch:
                    raise RuntimeError(
                        f"RPC keeps rejecting block ranges even at minimum batch size "
                        f"({MIN_BATCH_SIZE}). Check your RPC key, plan tier, and "
                        f"GRADUATION_START_BLOCK in config.py.\n"
                        f"Original error: {exc}"
                    ) from exc
                logger.warning(
                    "Block range too large or HTTP 400 at batch size %d — "
                    "halving to %d and retrying from block %d.",
                    effective_batch,
                    new_batch,
                    batch_start,
                )
                effective_batch = new_batch
                # Do NOT advance batch_start — retry the same window narrower.
                continue

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


class _BlockRangeTooLargeError(Exception):
    """Internal sentinel: RPC rejected the range — caller should halve and retry."""


def _is_range_error(exc: Exception) -> bool:
    """
    Return True if the exception signals that the requested block range was
    rejected by the RPC provider.

    Alchemy returns a plain HTTP 400 Bad Request — the response body may say
    "block range is too wide" but the exception message that web3.py surfaces
    is just the HTTP status line.  We therefore check:
      1. HTTP status code 400 or 413 directly on HTTPError.
      2. Keyword strings in the message for providers that embed a description.
    """
    error_str = str(exc).lower()
    keyword_hit = any(
        kw in error_str
        for kw in ("block range", "too large", "too wide", "limit exceeded", "response size")
    )
    if keyword_hit:
        return True

    # Check HTTP status codes directly.
    if isinstance(exc, requests.exceptions.HTTPError):
        try:
            status = exc.response.status_code
            if status in (400, 413):
                # 400 from Alchemy almost always means "range too large" or
                # "block too old for free tier" — treat as range error so we
                # halve and retry rather than crashing.
                return True
        except AttributeError:
            pass

    return False


def _fetch_batch_with_retry(w3, filter_params: dict) -> list[LogReceipt]:
    """
    Fetch a single batch of logs with retry on transient failures.

    Args:
        w3:            Web3 instance.
        filter_params: eth_getLogs filter dict.

    Returns:
        List of log entries (may be empty if no matching events in range).

    Raises:
        _BlockRangeTooLargeError: If the RPC signals the range is too large.
        RuntimeError:             If all retries are exhausted for this batch.
    """
    delay = config.RPC_RETRY_DELAY_SECONDS
    last_exc: Optional[Exception] = None

    for attempt in range(1, config.RPC_MAX_RETRIES + 1):
        try:
            logs = w3.eth.get_logs(filter_params)
            return list(logs)
        except Exception as exc:
            last_exc = exc

            # Range-too-large: signal caller to halve batch — do NOT retry here.
            if _is_range_error(exc):
                logger.warning(
                    "RPC rejected block range %d–%d (range too large / HTTP 400): %s.",
                    filter_params["fromBlock"],
                    filter_params["toBlock"],
                    exc,
                )
                raise _BlockRangeTooLargeError(str(exc)) from exc

            # Rate-limited: back off longer before retry.
            if isinstance(exc, requests.exceptions.HTTPError):
                try:
                    if exc.response.status_code == 429:
                        backoff = delay * 4
                        logger.warning(
                            "Rate-limited (429) on batch %d–%d. Backing off %.1fs...",
                            filter_params["fromBlock"],
                            filter_params["toBlock"],
                            backoff,
                        )
                        time.sleep(backoff)
                        delay *= 2
                        continue
                except AttributeError:
                    pass

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

def _is_range_error(exc: Exception) -> bool:
    error_str = str(exc).lower()
    keyword_hit = any(
        kw in error_str
        for kw in ("block range", "too large", "too wide", "limit exceeded", 
                   "response size", "pruned", "pruned history")  # ← add these two
    )