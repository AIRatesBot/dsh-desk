"""
DSH backtest runner.

Runs a strategy backtest with IS/OOS split, cost sensitivity, baseline
comparison, deflated Sharpe ratio, and sequential gates.
Writes the result to the Neon ledger.

All network calls have explicit timeouts so a stalled request fails fast
instead of hanging the CI job indefinitely.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import yfinance as yf
import vectorbt as vbt
import psycopg2
from deflated_sharpe import deflated_sharpe_ratio


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
BASELINE_MIN_DELTA  = 0.20
MAX_DD_LIMIT        = -0.30
DSR_ALPHA           = 0.05

YF_TIMEOUT          = 30     # seconds for Yahoo Finance HTTP call
DB_CONNECT_TIMEOUT  = 15     # seconds for Neon connection


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def safe_float(x):
    try:
        v = float(x)
        return v if not np.isnan(v) else 0.0
    except Exception:
        return 0.0


def download_price(ticker: str) -> pd.Series:
    """Download daily close price with explicit timeout."""
    t = yf.Ticker(ticker)
    hist = t.history(period="max", timeout=YF_TIMEOUT)
    if hist is None or hist.empty or "Close" not in hist.columns:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    price = hist["Close"].dropna()
    if price.empty:
        raise RuntimeError(f"empty price series for {ticker}")
    return price


def run_backtest(price, entries, exits, fees_bps: float):
    pf = vbt.Portfolio.from_signals(
        price, entries, exits,
        init_cash=10_000,
        fees=fees_bps / 10_000.0,
        freq="1D",
    )
    return safe_float(pf.sharpe_ratio()), safe_float(pf.max_drawdown()), pf


def compute_baseline_sharpe(price_oos):
    rets = price_oos.pct_change().dropna()
    if len(rets) == 0 or rets.std() == 0:
        return 0.0
    return safe_float(rets.mean() / rets.std() * np.sqrt(252))


def count_trials(db_url, alpha):
    with psycopg2.connect(db_url, connect_timeout=DB_CONNECT_TIMEOUT) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM candidates
                WHERE alpha_source = %s
                  AND strategy_name NOT LIKE 'test_%%'
                """,
                (alpha,),
            )
            return int(cur.fetchone()[0])


def compute_dsr(pf_oos, num_trials):
    rets = pf_oos.returns().squeeze().dropna()
    if len(rets) < 30 or rets.std() == 0:
        return {"dsr": 0.0, "p": 1.0, "sr_daily": 0.0,
                "skew": 0.0, "kurt": 3.0, "n": int(len(rets))}
    sr_daily = float(rets.mean() / rets.std())
    skew     = float(rets.skew())
    kurt     = float(rets.kurtosis() + 3.0)
    dsr, p = deflated_sharpe_ratio(
        observed_sr=sr_daily,
        num_trials=num_trials,
        num_obs=len(rets),
        skewness=skew,
        kurtosis=kurt,
    )
    return {"dsr": float(dsr), "p": float(p), "sr_daily": sr_daily,
            "skew": skew, "kurt": kurt, "n": int(len(rets))}


def apply_gates(oos_sharpe, cost_results, baseline_delta, max_dd, dsr, dsr_p):
    if oos_sharpe < OOS_MIN_SHARPE:
        return "rejected", "weak_oos"
    if cost_results[COST_KILL_BPS] < COST_KILL_MIN:
        return "rejected", "cost_kill"
    if baseline_delta < BASELINE_MIN_DELTA:
        return "rejected", "no_edge"
    if max_dd < MAX_DD_LIMIT:
        return "rejected", "max_dd"
    if dsr <= 0 or dsr_p >= DSR_ALPHA:
        return "rejected", "dsr_fail"
    return "passed", None


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def run():
    db_url = os.environ["DATABASE_URL"]

    # 1. Data
    print("STEP 1: downloading data...", flush=True)
    try:
        price = download_price("BTC-USD")
    except Exception as e:
        print(f"ERROR: data download failed: {e}", flush=True)
        sys.exit(1)
    print(f"STEP 1 OK: {len(price)} bars", flush=True)

    # 2. Signal
    print("STEP 2: building signal...", flush=True)
    fast = vbt.MA.run(price, 10)
    slow = vbt.MA.run(price, 50)
    entries = fast.ma_crossed_above(slow)
    exits   = fast.ma_crossed_below(slow)
    print("STEP 2 OK", flush=True)

    # 3. IS / OOS split
    print("STEP 3: splitting IS/OOS...", flush=True)
    n = len(price)
    split_idx = int(n * OOS_SPLIT)
    price_is,   price_oos   = price.iloc[:split_idx],   price.iloc[split_idx:]
    entries_is, entries_oos = entries.iloc[:split_idx], entries.iloc[split_idx:]
    exits_is,   exits_oos   = exits.iloc[:split_idx],   exits.iloc[split_idx:]
    print(f"STEP 3 OK: IS={len(price_is)}  OOS={len(price_oos)}", flush=True)

    # 4. Cost sensitivity
    print("STEP 4: cost sensitivity...", flush=True)
    cost_results = {}
    for bps in COST_LEVELS_BPS:
        sharpe, _, _ = run_backtest(price_oos, entries_oos, exits_oos, bps)
        cost_results[bps] = sharpe
    oos_sharpe = cost_results[COST_KILL_BPS]
    print(f"STEP 4 OK: {cost_results}", flush=True)

    # 5. IS Sharpe
    print("STEP 5: IS Sharpe...", flush=True)
    is_sharpe, _, _ = run_backtest(price_is, entries_is, exits_is, COST_KILL_BPS)
    print(f"STEP 5 OK: is_sharpe={is_sharpe:.3f}", flush=True)

    # 6. Full-period drawdown
    print("STEP 6: full-period drawdown...", flush=True)
    _, max_dd, _ = run_backtest(price, entries, exits, COST_KILL_BPS)
    print(f"STEP 6 OK: max_dd={max_dd:.3f}", flush=True)

    # 7. Baseline
    print("STEP 7: baseline...", flush=True)
    baseline_sharpe = compute_baseline_sharpe(price_oos)
    baseline_delta  = oos_sharpe - baseline_sharpe
    print(f"STEP 7 OK: baseline={baseline_sharpe:.3f}  delta={baseline_delta:+.3f}", flush=True)

    # 8. DSR
    print("STEP 8: DSR...", flush=True)
    _, _, pf_oos = run_backtest(price_oos, entries_oos, exits_oos, COST_KILL_BPS)
    num_trials = count_trials(db_url, ALPHA) + 1
    dsr_info   = compute_dsr(pf_oos, num_trials)
    dsr        = dsr_info["dsr"]
    dsr_p      = dsr_info["p"]
    print(f"STEP 8 OK: dsr={dsr:.3f} p={dsr_p:.4f} trials={num_trials}", flush=True)

    # 9. Gates
    gate_status, reject_reason = apply_gates(
        oos_sharpe, cost_results, baseline_delta, max_dd, dsr, dsr_p
    )

    # 10. Write to Neon
    print("STEP 9: writing to Neon...", flush=True)
    try:
        with psycopg2.connect(db_url, connect_timeout=DB_CONNECT_TIMEOUT) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO candidates
                      (strategy_name, alpha_source, spec_json,
                       is_sharpe, oos_sharpe, deflated_sharpe,
                       max_drawdown, baseline_delta,
                       gate_status, reject_reason,
                       dsr_pvalue, num_trials)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                            "dsr_alpha": DSR_ALPHA,
                            "dsr_sr_daily": dsr_info["sr_daily"],
                            "dsr_skew": dsr_info["skew"],
                            "dsr_kurt": dsr_info["kurt"],
                            "dsr_n_obs": dsr_info["n"],
                        }),
                        is_sharpe, oos_sharpe, dsr,
                        max_dd, baseline_delta,
                        gate_status, reject_reason,
                        dsr_p, num_trials,
                    ),
                )
    except Exception as e:
        print(f"ERROR: Neon write failed: {e}", flush=True)
        sys.exit(1)
    print("STEP 9 OK: written to Neon", flush=True)

    # 11. Summary
    print("---", flush=True)
    print(f"OK: {STRATEGY}  (alpha={ALPHA})", flush=True)
    print(f"  IS  Sharpe (1bp)   = {is_sharpe:+.3f}")
    print(f"  OOS Sharpe 0bp     = {cost_results[0]:+.3f}")
    print(f"  OOS Sharpe 1bp     = {cost_results[1]:+.3f}  (min {OOS_MIN_SHARPE})")
    print(f"  OOS Sharpe 5bp     = {cost_results[5]:+.3f}")
    print(f"  Baseline Sharpe    = {baseline_sharpe:+.3f}  (buy & hold, OOS)")
    print(f"  Baseline delta     = {baseline_delta:+.3f}  (min {BASELINE_MIN_DELTA})")
    print(f"  Max DD   (1bp)     = {max_dd:+.3f}  (limit {MAX_DD_LIMIT})")
    print(f"  DSR: sr_daily={dsr_info['sr_daily']:+.4f}  n={dsr_info['n']}  "
          f"trials={num_trials}  skew={dsr_info['skew']:+.2f}  "
          f"kurt={dsr_info['kurt']:.2f}")
    print(f"  DSR                = {dsr:+.3f}  p={dsr_p:.4f}  (alpha {DSR_ALPHA})")
    print(f"  Status             = {gate_status}"
          + (f"  ({reject_reason})" if reject_reason else ""))


if __name__ == "__main__":
    run()
