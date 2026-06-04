"""
src/chain/rpc_client.py — Web3 connection manager for Base mainnet.

Responsibilities:
    - Single shared Web3 instance used by all pipeline modules.
    - Validates chain ID is 8453 (Base) on startup; raises if not.
    - Automatic retry with exponential backoff on transient RPC failures.
    - Falls back to secondary RPC if primary exhausts all retries.
    - Logs every connection event and every retry with full context.
"""

import logging
import time
from typing import Optional

from web3 import Web3
from web3.exceptions import BlockNotFound
# web3.py v6 renamed ExtraDataToPOAMiddleware → geth_poa_middleware.
from web3.middleware import geth_poa_middleware

import config

logger = logging.getLogger(__name__)

# Module-level singleton — one instance per process, shared by all callers.
_client: Optional[Web3] = None
_active_rpc_url: Optional[str] = None


def _build_web3(rpc_url: str) -> Web3:
    """
    Construct a Web3 instance from an HTTP(S) URL.
    Injects POA middleware required for Base (PoA consensus layer).
    """
    provider = Web3.HTTPProvider(
        rpc_url,
        request_kwargs={"timeout": 30},
    )
    w3 = Web3(provider)
    # Base uses a PoA-compatible consensus; inject middleware to handle
    # the extraData field that otherwise causes decode failures.
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    return w3


def _validate_chain(w3: Web3, rpc_url: str) -> None:
    """
    Assert the connected node is Base mainnet (chain ID 8453).
    Raises ValueError with context if the chain ID does not match.
    """
    try:
        chain_id = w3.eth.chain_id
    except Exception as exc:
        raise ConnectionError(
            f"Could not read chain ID from RPC {rpc_url!r}: {exc}"
        ) from exc

    if chain_id != config.BASE_CHAIN_ID:
        raise ValueError(
            f"Wrong chain. Expected {config.BASE_CHAIN_ID} (Base), "
            f"got {chain_id}. Check your RPC URL in .env."
        )
    logger.info("Chain ID validated: %d (Base mainnet)", chain_id)


def _connect_with_retry(rpc_url: str, label: str) -> Web3:
    """
    Attempt to connect to `rpc_url` with up to RPC_MAX_RETRIES attempts.
    Uses exponential backoff between retries.
    Returns a validated Web3 instance on success.
    Raises ConnectionError if all attempts fail.
    """
    delay = config.RPC_RETRY_DELAY_SECONDS
    last_exc: Optional[Exception] = None

    for attempt in range(1, config.RPC_MAX_RETRIES + 1):
        try:
            logger.info("Connecting to %s RPC (attempt %d/%d)...", label, attempt, config.RPC_MAX_RETRIES)
            w3 = _build_web3(rpc_url)
            _validate_chain(w3, rpc_url)
            # Smoke test: fetch a real block number to confirm the node is live.
            block_number = w3.eth.block_number
            logger.info(
                "Connected to %s RPC. Current block: %d", label, block_number
            )
            return w3
        except (ConnectionError, ValueError) as exc:
            # Chain ID mismatch or cannot reach node — don't retry on wrong chain.
            raise
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Connection attempt %d/%d to %s failed: %s. Retrying in %.1fs...",
                attempt,
                config.RPC_MAX_RETRIES,
                label,
                exc,
                delay,
            )
            if attempt < config.RPC_MAX_RETRIES:
                time.sleep(delay)
                delay *= 2  # Exponential backoff.

    raise ConnectionError(
        f"All {config.RPC_MAX_RETRIES} connection attempts to {label} RPC failed. "
        f"Last error: {last_exc}"
    )


def get_client() -> Web3:
    """
    Return the shared Web3 client, initialising it on first call.

    Connection strategy:
        1. Try primary RPC (BASE_RPC_URL from config).
        2. If primary fails all retries, fall back to secondary (BASE_RPC_URL_FALLBACK).
        3. If both fail, raise ConnectionError — pipeline cannot proceed.

    Returns:
        Validated Web3 instance connected to Base mainnet (chain 8453).

    Raises:
        ConnectionError: Both primary and fallback RPCs exhausted.
        ValueError: Connected node is not Base mainnet.
    """
    global _client, _active_rpc_url

    if _client is not None:
        return _client

    # Try primary.
    try:
        w3 = _connect_with_retry(config.BASE_RPC_URL, label="primary")
        _client = w3
        _active_rpc_url = config.BASE_RPC_URL
        return _client
    except ConnectionError as primary_exc:
        logger.error("Primary RPC failed: %s. Trying fallback...", primary_exc)

    # Try fallback.
    try:
        w3 = _connect_with_retry(config.BASE_RPC_URL_FALLBACK, label="fallback")
        _client = w3
        _active_rpc_url = config.BASE_RPC_URL_FALLBACK
        logger.warning("Operating on FALLBACK RPC. Primary is unavailable.")
        return _client
    except ConnectionError as fallback_exc:
        raise ConnectionError(
            f"Both primary and fallback RPCs failed.\n"
            f"Primary error: {primary_exc}\n"
            f"Fallback error: {fallback_exc}\n"
            "Check BASE_RPC_URL and BASE_RPC_URL_FALLBACK in .env."
        ) from fallback_exc


def reset_client() -> None:
    """
    Force re-initialisation of the shared client on the next get_client() call.
    Used after a detected connection failure mid-pipeline.
    """
    global _client, _active_rpc_url
    _client = None
    _active_rpc_url = None
    logger.info("RPC client reset. Will reconnect on next get_client() call.")


def call_with_retry(fn, *args, **kwargs):
    """
    Execute a web3 call with automatic retry on transient failures.

    Catches only transient errors (network timeouts, rate limits).
    Raises immediately on permanent errors (invalid address, ABI mismatch).

    Args:
        fn: Callable that performs a web3 operation.
        *args, **kwargs: Forwarded to fn.

    Returns:
        Result of fn(*args, **kwargs).

    Raises:
        Exception: If all retries are exhausted or a permanent error occurs.
    """
    delay = config.RPC_RETRY_DELAY_SECONDS
    last_exc: Optional[Exception] = None

    for attempt in range(1, config.RPC_MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except (BlockNotFound, ValueError) as exc:
            # Permanent errors — retrying won't help.
            raise
        except Exception as exc:
            last_exc = exc
            error_str = str(exc).lower()

            # Detect rate limit responses and back off longer.
            if "rate limit" in error_str or "429" in error_str or "too many requests" in error_str:
                backoff = delay * 3
                logger.warning(
                    "Rate limited on attempt %d/%d. Backing off %.1fs: %s",
                    attempt, config.RPC_MAX_RETRIES, backoff, exc,
                )
                time.sleep(backoff)
            else:
                logger.warning(
                    "RPC call attempt %d/%d failed: %s. Retrying in %.1fs...",
                    attempt, config.RPC_MAX_RETRIES, exc, delay,
                )
                if attempt < config.RPC_MAX_RETRIES:
                    time.sleep(delay)
                    delay *= 2

    raise RuntimeError(
        f"RPC call failed after {config.RPC_MAX_RETRIES} attempts. "
        f"Last error: {last_exc}"
    ) from last_exc
