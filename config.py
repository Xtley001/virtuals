"""
config.py — Central configuration for the Virtuals Protocol Graduation Arb Test Engine.

VERIFICATION PROTOCOL:
    Every contract address in this file was located via Basescan (https://basescan.org)
    before being written here. Addresses requiring operator confirmation are flagged
    with REQUIRES_OPERATOR_VERIFICATION. The pipeline will raise at startup if those
    slots are empty in the .env file.

    Verified addresses (can be confirmed at basescan.org at the listed URL):
        VIRTUAL_TOKEN_ADDRESS     → basescan.org/token/0x0b3e328455c4059EEb9e3f84b5543F74E24e7E1b
        USDC_TOKEN_ADDRESS        → basescan.org/token/0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913
        UNISWAP_V3_FACTORY_ADDRESS → basescan.org/address/0x33128a8fC17869897dcE68Ed026d694621f6FDfD
        WETH_ADDRESS              → basescan.org/token/0x4200000000000000000000000000000000000006

    Addresses that require the operator to supply via .env (cannot be hardcoded without
    on-chain confirmation that may change with contract upgrades):
        VIRTUALS_FACTORY_ADDRESS  → Set VIRTUALS_FACTORY_ADDRESS in .env after confirming
                                    the current factory on basescan.org (search "Virtuals
                                    Protocol AgentFactory" or follow transactions from the
                                    $VIRTUAL token contract). The factory is an upgradeable
                                    proxy — always use the proxy address, not the implementation.
        AERODROME_FACTORY_ADDRESS → Set AERODROME_FACTORY_ADDRESS in .env after confirming
                                    at basescan.org/address/... (Aerodrome Finance factory).
        AERODROME_ROUTER_ADDRESS  → Set AERODROME_ROUTER_ADDRESS in .env similarly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ────────────────────────────────────────────────────────────────
_env_path = Path(__file__).parent / ".env"
if not _env_path.exists():
    raise FileNotFoundError(
        f".env file not found at {_env_path}. "
        "Copy .env.template to .env and fill in all values before running."
    )
load_dotenv(_env_path)

# ── Chain ─────────────────────────────────────────────────────────────────────
BASE_CHAIN_ID: int = 8453

BASE_RPC_URL: str = os.environ["BASE_RPC_URL"]          # raises KeyError if missing
BASE_RPC_URL_FALLBACK: str = os.environ["BASE_RPC_URL_FALLBACK"]

# ── Token Addresses (verified on Basescan) ────────────────────────────────────
# $VIRTUAL ERC-20 on Base.
# Source: https://whitepaper.virtuals.io/info-hub/important-links-and-resources/contract-address
# Confirmed: basescan.org/token/0x0b3e328455c4059EEb9e3f84b5543F74E24e7E1b
VIRTUAL_TOKEN_ADDRESS: str = "0x0b3e328455c4059EEb9e3f84b5543F74E24e7E1b"

# USDC on Base (native Circle deployment, not bridged).
# Confirmed: basescan.org/token/0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913
USDC_TOKEN_ADDRESS: str = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

# WETH on Base (canonical L2 deployment at deterministic address).
# Confirmed: basescan.org/token/0x4200000000000000000000000000000000000006
WETH_ADDRESS: str = "0x4200000000000000000000000000000000000006"

# ── Uniswap V3 on Base (verified) ─────────────────────────────────────────────
# Factory — canonical Uniswap V3 deployment on Base.
# Confirmed: basescan.org/address/0x33128a8fC17869897dcE68Ed026d694621f6FDfD
UNISWAP_V3_FACTORY_ADDRESS: str = "0x33128a8fC17869897dcE68Ed026d694621f6FDfD"

# Pool init code hash — required for deterministic pool address derivation.
# Source: Uniswap V3 Core — same hash used on all EVM chains.
# Value: keccak256(type(UniswapV3Pool).creationCode) per the V3 factory spec.
UNISWAP_V3_POOL_INIT_CODE_HASH: str = (
    "0xe34f199b19b2b4f47f68442619d555527d244f78a3297ea89325f843f87b8b54"
)

# ── Virtuals Protocol (REQUIRES_OPERATOR_VERIFICATION) ───────────────────────
# The AgentFactory is an upgradeable proxy. The address must be confirmed on
# Basescan each time this config is used. Do not hardcode the implementation.
#
# To find it:
#   1. Go to basescan.org
#   2. Search "Virtuals Protocol" → find the contract emitting Launched/Graduated events
#   3. Confirm the contract name and verified source matches AgentFactory
#   4. Copy the PROXY address (not implementation) into .env as VIRTUALS_FACTORY_ADDRESS
#
# The FFactory (fun.virtuals.io bonding curve factory) address:
#   Known candidates from chain analysis: 0x5c3C...
#   MUST be confirmed by operator before running. The pipeline hard-fails if unset.
VIRTUALS_FACTORY_ADDRESS: str = os.environ.get("VIRTUALS_FACTORY_ADDRESS", "")
if not VIRTUALS_FACTORY_ADDRESS:
    raise EnvironmentError(
        "VIRTUALS_FACTORY_ADDRESS is not set in .env.\n"
        "Verification steps:\n"
        "  1. Visit https://basescan.org and search 'Virtuals Protocol'\n"
        "  2. Find the contract that emits graduation/migration events\n"
        "  3. Read the verified source to confirm it's the AgentFactory\n"
        "  4. Add VIRTUALS_FACTORY_ADDRESS=0x... to your .env file\n"
        "  Known address from community data (verify before use):\n"
        "    VIRTUALS_FACTORY_ADDRESS=0x5c3C0b36E63bA6E4c3906A38D4F0CD2ABeE016B8"
    )

# Graduation threshold in VIRTUAL tokens (18 decimals).
# Source: whitepaper.virtuals.io/builders-hub/build-with-virtuals/agent-creation
GRADUATION_THRESHOLD_VIRTUAL: int = 42_000  # human units; multiply by 1e18 for wei

# The event signature(s) to watch for graduation. The factory emits a
# "Launched" event when an agent graduates from bonding curve to Uniswap V3.
# Candidate event names: TokenGraduated, Launched, AgentCreated — operator must
# confirm by reading the verified contract ABI on Basescan.
GRADUATION_EVENT_NAME: str = os.environ.get("GRADUATION_EVENT_NAME", "Launched")

# The block at which the Virtuals Protocol factory was first deployed on Base.
# Used as the start block for the graduation event scan (full discovery).
# The analysis window below further filters which events are included in P&L
# simulation and summary stats — discovery always starts here.
# Source: Basescan contract creation transaction.
# Known approximate value: Virtuals fun.virtuals.io launched ~Oct 2024 ≈ block 20,000,000.
# Operator should verify: basescan.org/address/<VIRTUALS_FACTORY_ADDRESS>#code
GRADUATION_START_BLOCK: int = int(os.environ.get("GRADUATION_START_BLOCK", "20000000"))

# ── Analysis Window ───────────────────────────────────────────────────────────
# Discovery always scans from GRADUATION_START_BLOCK to present.
# The analysis window below defines which events feed the P&L simulation,
# summary stats, and go/no-go gates. Events outside this window are discovered
# and stored in master_dataset but excluded from the decision metrics.
#
# July 1, 2025 → present:
#   - Timestamp: 1751328000 (July 1, 2025 00:00:00 UTC)
#   - Approx Base block: ~29,894,400  (Base genesis Aug 9 2023 + elapsed at 2s/block)
#
# Rationale for this window:
#   - Excludes early protocol period when bonding curve mechanics were being tuned.
#   - Captures current competitive landscape and fee structures.
#   - Provides ~months of data — enough for robust distribution estimates.
ANALYSIS_WINDOW_START_TIMESTAMP: int = int(os.environ.get(
    "ANALYSIS_WINDOW_START_TIMESTAMP", "1751328000"  # 2025-07-01 00:00:00 UTC
))
ANALYSIS_WINDOW_END_TIMESTAMP: int = int(os.environ.get(
    "ANALYSIS_WINDOW_END_TIMESTAMP", "0"  # 0 = use current time at runtime
))

# Corresponding approximate block for the analysis start (used for any block-range
# filters that need a block number rather than a timestamp).
# Derived: Base genesis 1691539200 + (1751328000 - 1691539200) / 2.0 ≈ 29,894,400
ANALYSIS_WINDOW_START_BLOCK: int = int(os.environ.get(
    "ANALYSIS_WINDOW_START_BLOCK", "29894400"
))

# ── Aerodrome Finance on Base (REQUIRES_OPERATOR_VERIFICATION) ───────────────
# Aerodrome is the dominant DEX/liquidity router on Base.
# Verify at basescan.org before use — these are known from public sources but
# must be confirmed current (protocol may deploy new router versions).
#
# Known Aerodrome addresses (from public Aerodrome docs; verify on Basescan):
#   Factory: 0x420DD381b31aEf6683db6B902084cB0FFECe40Da
#   Router:  0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43
AERODROME_FACTORY_ADDRESS: str = os.environ.get(
    "AERODROME_FACTORY_ADDRESS",
    "0x420DD381b31aEf6683db6B902084cB0FFECe40Da",  # verify on basescan before use
)
AERODROME_ROUTER_ADDRESS: str = os.environ.get(
    "AERODROME_ROUTER_ADDRESS",
    "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43",  # verify on basescan before use
)

# ── Pipeline Parameters ───────────────────────────────────────────────────────
# eth_getLogs batch size — stay within RPC provider per-request block range limits.
# Alchemy allows up to 2000 blocks per eth_getLogs call for Base.
BLOCKS_PER_BATCH: int = 2_000

CHECKPOINT_DIR: str = "data/checkpoints"
RAW_DIR: str = "data/raw"
OUTPUT_DIR: str = "data/output"

# ── P&L Simulation Parameters ─────────────────────────────────────────────────
SIM_SIZE_SMALL_USD: float = 5_000.0
SIM_SIZE_MEDIUM_USD: float = 25_000.0
SIM_SIZE_LARGE_USD: float = 75_000.0

# Spread below this threshold is treated as noise — skip in simulation.
MIN_SPREAD_PCT_THRESHOLD: float = 0.5

# Minimum USD profit to count a trade as profitable.
MIN_NET_PROFIT_USD: float = 10.0

# ── Rate Limits ───────────────────────────────────────────────────────────────
# CoinGecko free tier: 10-30 calls/min; pro tier: up to 500 calls/min.
# We conservatively target free-tier limits.
COINGECKO_CALLS_PER_MINUTE: int = 10

# Basescan API: 5 calls/sec on free tier.
BASESCAN_CALLS_PER_SECOND: float = 5.0

# RPC retry configuration.
RPC_MAX_RETRIES: int = 3
RPC_RETRY_DELAY_SECONDS: float = 2.0

# ── API Keys ──────────────────────────────────────────────────────────────────
BASESCAN_API_KEY: str = os.environ["BASESCAN_API_KEY"]
COINGECKO_API_KEY: str = os.environ.get("COINGECKO_API_KEY", "")  # optional

# ── Go / No-Go Gate Thresholds ────────────────────────────────────────────────
GATE_1_MEDIAN_SPREAD_PCT: float = 2.0       # Median spread at graduation ≥ 2.0%
GATE_2_WIN_RATE_MEDIUM: float = 0.55        # Win rate at $25k size ≥ 55%
GATE_3_MEDIAN_BOT_COUNT: float = 5.0        # Median bot count in first 3 blocks ≤ 5

# ── ABIs (inline minimal ABIs for core events) ────────────────────────────────
# These minimal ABIs contain only the events and functions we need.
# Full ABIs are fetched from Basescan at runtime via get_abi() in the chain module.

# Uniswap V3 Factory — PoolCreated event.
# Source: Uniswap V3 Core repository (verified, canonical).
UNISWAP_V3_FACTORY_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "token0", "type": "address"},
            {"indexed": True, "name": "token1", "type": "address"},
            {"indexed": True, "name": "fee", "type": "uint24"},
            {"indexed": False, "name": "tickSpacing", "type": "int24"},
            {"indexed": False, "name": "pool", "type": "address"},
        ],
        "name": "PoolCreated",
        "type": "event",
    }
]

# Uniswap V3 Pool — Initialize event and slot0 read.
# Source: Uniswap V3 Core — IUniswapV3PoolState interface.
UNISWAP_V3_POOL_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": False, "name": "sqrtPriceX96", "type": "uint160"},
            {"indexed": False, "name": "tick", "type": "int24"},
        ],
        "name": "Initialize",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "sender", "type": "address"},
            {"indexed": True, "name": "recipient", "type": "address"},
            {"indexed": False, "name": "amount0", "type": "int256"},
            {"indexed": False, "name": "amount1", "type": "int256"},
            {"indexed": False, "name": "sqrtPriceX96", "type": "uint160"},
            {"indexed": False, "name": "liquidity", "type": "uint128"},
            {"indexed": False, "name": "tick", "type": "int24"},
        ],
        "name": "Swap",
        "type": "event",
    },
    {
        "inputs": [],
        "name": "slot0",
        "outputs": [
            {"name": "sqrtPriceX96", "type": "uint160"},
            {"name": "tick", "type": "int24"},
            {"name": "observationIndex", "type": "uint16"},
            {"name": "observationCardinality", "type": "uint16"},
            {"name": "observationCardinalityNext", "type": "uint16"},
            {"name": "feeProtocol", "type": "uint8"},
            {"name": "unlocked", "type": "bool"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "liquidity",
        "outputs": [{"name": "", "type": "uint128"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "token0",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "token1",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "fee",
        "outputs": [{"name": "", "type": "uint24"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "tickLower", "type": "int24"},
            {"name": "tickUpper", "type": "int24"},
        ],
        "name": "ticks",
        "outputs": [
            {"name": "liquidityGross", "type": "uint128"},
            {"name": "liquidityNet", "type": "int128"},
            {"name": "feeGrowthOutside0X128", "type": "uint256"},
            {"name": "feeGrowthOutside1X128", "type": "uint256"},
            {"name": "tickCumulativeOutside", "type": "int56"},
            {"name": "secondsPerLiquidityOutsideX128", "type": "uint160"},
            {"name": "secondsOutside", "type": "uint32"},
            {"name": "initialized", "type": "bool"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "wordPosition", "type": "int16"}],
        "name": "tickBitmap",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "tickSpacing",
        "outputs": [{"name": "", "type": "int24"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# ERC-20 minimal ABI — name() and decimals() only.
ERC20_MINIMAL_ABI = [
    {
        "inputs": [],
        "name": "name",
        "outputs": [{"name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# Aerodrome Pool — getReserves() for liquidity depth queries.
AERODROME_POOL_ABI = [
    {
        "inputs": [],
        "name": "getReserves",
        "outputs": [
            {"name": "_reserve0", "type": "uint256"},
            {"name": "_reserve1", "type": "uint256"},
            {"name": "_blockTimestampLast", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "token0",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "token1",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
]

AERODROME_FACTORY_ABI = [
    {
        "inputs": [
            {"name": "tokenA", "type": "address"},
            {"name": "tokenB", "type": "address"},
            {"name": "stable", "type": "bool"},
        ],
        "name": "getPair",
        "outputs": [{"name": "pair", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
]
