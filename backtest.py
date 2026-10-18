"""
DSH backtest runner.

Runs a strategy backtest with IS/OOS split, cost sensitivity, baseline
comparison, and sequential gates. Writes the result to the Neon ledger.
"""

import os
import json
import numpy as np
import vectorbt as vbt
import psycopg2


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
STRATEGY = os.environ.get("STRATEGY", "sma_cross_btc")
ALPHA    = os.environ.get("ALPHA", "mom")

OOS_SPLIT           = 0.70
OOS_MIN_SHARPE      = 0.30
COST_LEVELS_BPS     = [0, 1, 5]
COST_KILL_BPS       = 1
COST_KILL_MIN       = 0.10
BASELINE_MIN_DELTA  = 0.20     # strategy Sharpe must beat B&H by this much
MAX_DD_LIMIT        = -0.30


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def safe_float(x):
    try:
        v = float(x)
        return v if not np.isnan(v) else 0.0
    except Exception:
        return 0.0


def run_backtest(price, entries, exits, fees_bps: float):
    pf = vbt.Portfolio.from_signals(
        price, entries, exits,
        init_cash=10_000,
        fees=fees_bps / 10_000.0,
        freq="1D",
    )
    return safe_float(pf.sharpe_ratio()), safe_float(pf.max_drawdown())


def compute_baseline_sharpe(price_oos):
    """Buy-and-hold Sharpe on the OOS period (daily returns, annualized)."""
    rets = price_oos.pct_change().dropna()
    if len(rets) == 0 or rets.std() == 0:
        return 0.0
    return safe_float(rets.mean() / rets.std() * np.sqrt(252))


def apply_gates(oos_sharpe, cost_results, baseline_delta, max_dd):
    """Apply gates in order. First failing gate wins."""
    if oos_sharpe < OOS_MIN_SHARPE:
        return "rejected", "weak_oos"
    if cost_results[COST_KILL_BPS] < COST_KILL_MIN:
        return "rejected", "cost_kill"
    if baseline_delta < BASELINE_MIN_DELTA:
        return "rejected", "no_edge"
    if max_dd < MAX_DD_LIMIT:
        return "rejected", "max_dd"
    return "passed", None


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def run():
    # 1. Data
    data  = vbt.YFData.download("BTC-USD")
    price = data.get("Close")

    # 2. Signal
    fast = vbt.MA.run(price, 10)
    slow = vbt.MA.run(price, 50)
    entries = fast.ma_crossed_above(slow)
    exits   = fast.ma_crossed_below(slow)

    # 3. IS / OOS split
    n = len(price)
    split_idx = int(n * OOS_SPLIT)

    price_is,   price_oos   = price.iloc[:split_idx],   price.iloc[split_idx:]
    entries_is, entries_oos = entries.iloc[:split_idx], entries.iloc[split_idx:]
    exits_is,   exits_oos   = exits.iloc[:split_idx],   exits.iloc[split_idx:]

    # 4. Cost sensitivity on OOS
    cost_results = {}
    for bps in COST_LEVELS_BPS:
        sharpe, _ = run_backtest(price_oos, entries_oos, exits_oos, bps)
        cost_results[bps] = sharpe

    oos_sharpe = cost_results[COST_KILL_BPS]

    # 5. IS Sharpe (reporting only)
    is_sharpe, _ = run_backtest(price_is, entries_is, exits_is, COST_KILL_BPS)

    # 6. Drawdown on full period
    _, max_dd = run_backtest(price, entries, exits, COST_KILL_BPS)

    # 7. Baseline comparison
    baseline_sharpe = compute_baseline_sharpe(price_oos)
    baseline_delta  = oos_sharpe - baseline_sharpe

    # 8. Gates
    gate_status, reject_reason = apply_gates(
        oos_sharpe, cost_results, baseline_delta, max_dd
    )

    # 9. Placeholder (4d)
    deflated_sharpe = oos_sharpe * 0.7

    # 10. Write to Neon
    db_url = os.environ["DATABASE_URL"]
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO candidates
                  (strategy_name, alpha_source, spec_json,
                   is_sharpe, oos_sharpe, deflated_sharpe,
                   max_drawdown, baseline_delta,
                   gate_status, reject_reason)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    STRATEGY, ALPHA,
                    json.dumps({
                        "fast": 10,
                        "slow": 50,
                        "oos_split": OOS_SPLIT,
                        "oos_min_sharpe": OOS_MIN_SHARPE,
                        "cost_levels_bps": COST_LEVELS_BPS,
                        "cost_sharpe": cost_results,
                        "cost_kill_bps": COST_KILL_BPS,
                        "cost_kill_min": COST_KILL_MIN,
                        "baseline_min_delta": BASELINE_MIN_DELTA,
                        "baseline_sharpe": baseline_sharpe,
                        "max_dd_limit": MAX_DD_LIMIT,
                    }),
                    is_sharpe, oos_sharpe, deflated_sharpe,
                    max_dd, baseline_delta,
                    gate_status, reject_reason,
                ),
            )

    # 11. Log
    print(f"OK: {STRATEGY}  (alpha={ALPHA})")
    print(f"  IS  Sharpe (1bp)   = {is_sharpe:+.3f}")
    print(f"  OOS Sharpe 0bp     = {cost_results[0]:+.3f}")
    print(f"  OOS Sharpe 1bp     = {cost_results[1]:+.3f}  (min {OOS_MIN_SHARPE})")
    print(f"  OOS Sharpe 5bp     = {cost_results[5]:+.3f}")
    print(f"  Baseline Sharpe    = {baseline_sharpe:+.3f}  (buy & hold, OOS)")
    print(f"  Baseline delta     = {baseline_delta:+.3f}  (min {BASELINE_MIN_DELTA})")
    print(f"  Max DD   (1bp)     = {max_dd:+.3f}  (limit {MAX_DD_LIMIT})")
    print(f"  Deflated Sharpe    = {deflated_sharpe:+.3f}  [placeholder]")
    print(f"  Status             = {gate_status}"
          + (f"  ({reject_reason})" if reject_reason else ""))


if __name__ == "__main__":
    run()
