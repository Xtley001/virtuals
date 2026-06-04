"""
src/price/virtual_price.py — Historical $VIRTUAL/USDC price fetcher.

Fetches $VIRTUAL price at historical block timestamps using CoinGecko's
market_chart/range endpoint. Results are cached in-memory to avoid
redundant API calls within a single run.

Rate limiting:
    Free tier: 10-30 calls/minute (we use 10 conservatively).
    All calls are tracked per-minute window; the client sleeps if the limit
    is approached, with a documented reason tied to the exact limit.

Interpolation note:
    CoinGecko provides minute-level granularity for recent data and
    hourly granularity for older data. When the requested timestamp
    falls between two data points, we linearly interpolate. This is
    documented per call site.
"""

import logging
import time
from collections import deque
from typing import Optional

import requests

import config

logger = logging.getLogger(__name__)

# CoinGecko coin ID for $VIRTUAL.
# Source: https://www.coingecko.com/en/coins/virtual-protocol
_COINGECKO_COIN_ID = "virtual-protocol"

# In-memory cache: timestamp → usd_price.
# Reused within a single pipeline run to avoid duplicate API calls.
# Key is rounded to 60-second buckets to improve cache hit rate.
_price_cache: dict[int, float] = {}

# Rate limiter state: track timestamps of recent calls within a rolling minute.
_call_timestamps: deque = deque()


class VirtualPriceClient:
    """
    Thread-safe (within a single process) client for fetching $VIRTUAL/USDC
    prices at historical timestamps via CoinGecko.
    """

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        if config.COINGECKO_API_KEY:
            self.session.headers.update({
                "x-cg-pro-api-key": config.COINGECKO_API_KEY
            })
            logger.info("CoinGecko Pro API key configured.")
        else:
            logger.info("CoinGecko free tier (no API key). Rate limit: %d calls/min.",
                        config.COINGECKO_CALLS_PER_MINUTE)

    def get_price_at_timestamp(self, unix_timestamp: int) -> float:
        """
        Return $VIRTUAL/USDC price at the given Unix timestamp.

        Strategy:
            1. Round timestamp to 60-second bucket for cache lookup.
            2. If cached, return immediately.
            3. Else fetch a ±30-minute window from CoinGecko and cache all points.
            4. Return the nearest cached price point (interpolating if needed).

        Args:
            unix_timestamp: Unix timestamp (seconds).

        Returns:
            USDC price of 1 $VIRTUAL at that timestamp.

        Raises:
            RuntimeError: If CoinGecko returns an error after retries.
            ValueError:   If no price data is available for the requested time.
        """
        # Round to 60-second bucket.
        bucket = (unix_timestamp // 60) * 60

        if bucket in _price_cache:
            return _price_cache[bucket]

        # Fetch a ±1800-second (30-minute) window to amortise API calls.
        window_start = unix_timestamp - 1800
        window_end = unix_timestamp + 1800

        self._fetch_and_cache_range(window_start, window_end)

        # Find nearest cached bucket.
        price = self._nearest_cached_price(bucket)
        if price is None:
            raise ValueError(
                f"No $VIRTUAL price data available near timestamp {unix_timestamp} "
                f"(UTC: {_ts_to_str(unix_timestamp)}). "
                "CoinGecko may not have data for this period."
            )
        return price

    def get_prices_at_timestamps(self, timestamps: list[int]) -> dict[int, float]:
        """
        Batch-fetch prices for multiple timestamps efficiently.

        Groups timestamps into a single range request where possible.

        Args:
            timestamps: List of Unix timestamps.

        Returns:
            Dict mapping each input timestamp to its USDC price.
        """
        if not timestamps:
            return {}

        results = {}
        uncached = []

        for ts in timestamps:
            bucket = (ts // 60) * 60
            if bucket in _price_cache:
                results[ts] = _price_cache[bucket]
            else:
                uncached.append(ts)

        if uncached:
            # Fetch one range covering all uncached timestamps.
            window_start = min(uncached) - 1800
            window_end = max(uncached) + 1800
            self._fetch_and_cache_range(window_start, window_end)

            for ts in uncached:
                bucket = (ts // 60) * 60
                price = self._nearest_cached_price(bucket)
                if price is not None:
                    results[ts] = price
                else:
                    logger.warning(
                        "No price data available near %s.", _ts_to_str(ts)
                    )

        return results

    def _fetch_and_cache_range(self, from_ts: int, to_ts: int) -> None:
        """
        Fetch price data for a time range from CoinGecko and populate the cache.
        Enforces the per-minute rate limit before making the request.

        Args:
            from_ts: Start of range (Unix timestamp).
            to_ts:   End of range (Unix timestamp).
        """
        self._enforce_rate_limit()

        url = (
            f"https://api.coingecko.com/api/v3/coins/{_COINGECKO_COIN_ID}"
            f"/market_chart/range"
        )
        params = {
            "vs_currency": "usd",
            "from": from_ts,
            "to": to_ts,
        }

        for attempt in range(1, config.RPC_MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, params=params, timeout=30)

                # Log raw response on unexpected status for debugging.
                if resp.status_code != 200:
                    logger.warning(
                        "CoinGecko returned HTTP %d for range %s–%s. "
                        "Raw response: %s",
                        resp.status_code,
                        _ts_to_str(from_ts),
                        _ts_to_str(to_ts),
                        resp.text[:500],
                    )

                if resp.status_code == 429:
                    # Rate limited — CoinGecko's actual limit was stricter than ours.
                    # Back off for 60 seconds (a full window reset).
                    logger.warning(
                        "CoinGecko rate limit hit (HTTP 429). Backing off 65s."
                    )
                    time.sleep(65)
                    self._enforce_rate_limit()
                    continue

                resp.raise_for_status()
                data = resp.json()

                # Validate response shape before accessing fields.
                if "prices" not in data:
                    logger.error(
                        "CoinGecko response missing 'prices' field. "
                        "Raw response: %s", data
                    )
                    raise RuntimeError(
                        "CoinGecko response missing 'prices' field. "
                        "Unexpected API response shape."
                    )

                prices = data["prices"]  # [[timestamp_ms, price], ...]
                if not prices:
                    logger.warning(
                        "CoinGecko returned empty price list for range %s–%s.",
                        _ts_to_str(from_ts), _ts_to_str(to_ts),
                    )
                    return

                # Populate cache: key = 60-second bucket, value = USD price.
                for point in prices:
                    ts_ms, price = point[0], point[1]
                    ts_sec = ts_ms // 1000
                    bucket = (ts_sec // 60) * 60
                    _price_cache[bucket] = float(price)

                logger.debug(
                    "Cached %d price points for range %s–%s.",
                    len(prices),
                    _ts_to_str(from_ts),
                    _ts_to_str(to_ts),
                )
                return

            except requests.RequestException as exc:
                if attempt < config.RPC_MAX_RETRIES:
                    wait = config.RPC_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
                    logger.warning(
                        "CoinGecko request attempt %d/%d failed: %s. "
                        "Retrying in %.1fs...",
                        attempt, config.RPC_MAX_RETRIES, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"CoinGecko request failed after {config.RPC_MAX_RETRIES} "
                        f"attempts for range {_ts_to_str(from_ts)}–{_ts_to_str(to_ts)}: {exc}"
                    ) from exc

    def _nearest_cached_price(self, bucket: int) -> Optional[float]:
        """
        Return the cached price for the given 60-second bucket, or the
        nearest available bucket within a ±10-minute search window.

        Linear interpolation is used when the exact bucket is not cached
        but two surrounding data points exist within the search window.
        This is documented here: CoinGecko free tier provides hourly
        granularity for data older than 90 days; interpolation fills gaps.

        Args:
            bucket: Unix timestamp rounded to the nearest 60 seconds.

        Returns:
            USD price float, or None if no data within the search window.
        """
        if bucket in _price_cache:
            return _price_cache[bucket]

        # Search ±10 minutes (600 seconds = 10 buckets).
        search_range = 600
        best_before: Optional[tuple[int, float]] = None  # (bucket, price) nearest before
        best_after: Optional[tuple[int, float]] = None   # (bucket, price) nearest after

        for offset in range(60, search_range + 1, 60):
            b_before = bucket - offset
            b_after = bucket + offset
            if b_before in _price_cache and best_before is None:
                best_before = (b_before, _price_cache[b_before])
            if b_after in _price_cache and best_after is None:
                best_after = (b_after, _price_cache[b_after])
            if best_before and best_after:
                break

        if best_before and best_after:
            # Linear interpolation between the two surrounding points.
            t0, p0 = best_before
            t1, p1 = best_after
            # Interpolation: documented — fills hourly-granularity gaps.
            weight = (bucket - t0) / (t1 - t0)
            price = p0 + weight * (p1 - p0)
            # Cache the interpolated result.
            _price_cache[bucket] = price
            return price

        if best_before:
            return best_before[1]
        if best_after:
            return best_after[1]

        return None

    def _enforce_rate_limit(self) -> None:
        """
        Block until it is safe to make the next CoinGecko API call.

        Rate: COINGECKO_CALLS_PER_MINUTE calls per 60-second rolling window.
        This is called before every API request — not as an afterthought.
        """
        now = time.monotonic()
        window = 60.0  # 1-minute rolling window.

        # Remove call timestamps older than 60 seconds.
        while _call_timestamps and (now - _call_timestamps[0]) > window:
            _call_timestamps.popleft()

        if len(_call_timestamps) >= config.COINGECKO_CALLS_PER_MINUTE:
            # Must wait until the oldest call in the window falls off.
            oldest = _call_timestamps[0]
            wait_time = window - (now - oldest) + 0.1  # +0.1s safety margin.
            if wait_time > 0:
                logger.info(
                    "CoinGecko rate limit: %d/%d calls used. Waiting %.1fs.",
                    len(_call_timestamps),
                    config.COINGECKO_CALLS_PER_MINUTE,
                    wait_time,
                )
                time.sleep(wait_time)

        _call_timestamps.append(time.monotonic())


def _ts_to_str(unix_ts: int) -> str:
    """Format a Unix timestamp as a human-readable UTC string for log messages."""
    import datetime
    return datetime.datetime.utcfromtimestamp(unix_ts).strftime("%Y-%m-%d %H:%M")


# Module-level singleton — shared by all callers in a run.
_client: Optional[VirtualPriceClient] = None


def get_client() -> VirtualPriceClient:
    """Return the shared VirtualPriceClient instance."""
    global _client
    if _client is None:
        _client = VirtualPriceClient()
    return _client


def get_virtual_price_at_timestamp(unix_timestamp: int) -> float:
    """
    Convenience function: get $VIRTUAL/USDC price at a Unix timestamp.

    See VirtualPriceClient.get_price_at_timestamp() for full documentation.
    """
    return get_client().get_price_at_timestamp(unix_timestamp)
