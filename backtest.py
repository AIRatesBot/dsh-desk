"""
DSH backtest runner. Minimal, ASCII-only, paste-safe.
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

STRATEGY = os.environ.get("STRATEGY", "sma_cross_btc")
ALPHA = os.environ.get("ALPHA", "mom")

OOS_SPLIT = 0.70
OOS_MIN_SHARPE = 0.30
COST_LEVELS_BPS = [0, 1, 5]
COST_KILL_BPS = 1
COST_KILL_MIN = 0.10
BASELINE_MIN_DELTA = 0.20
MAX_DD_LIMIT = -0.30
DSR_ALPHA = 0.05
YF_TIMEOUT = 30
DB_TIMEOUT = 15


def safe_float(x):
    try:
        v = float(x)
        if np.isnan(v):
            return 0.0
        return v
    except Exception:
        return 0.0


def download_price(ticker):
    t = yf.Ticker(ticker)
    hist = t.history(period="max", timeout=YF_TIMEOUT)
    if hist is None or hist.empty or "Close" not in hist.columns:
        raise RuntimeError("no data for " + ticker)
    price = hist["Close"].dropna()
    if price.empty:
        raise RuntimeError("empty price for " + ticker)
    return price


def run_backtest(price, entries, exits, fees_bps):
    pf = vbt.Portfolio.from_signals(
        price, entries, exits,
        init_cash=10000,
        fees=fees_bps / 10000.0,
        freq="1D",
    )
    return safe_float(pf.sharpe_ratio()), safe_float(pf.max_drawdown()), pf


def compute_baseline_sharpe(price_oos):
    rets = price_oos.pct_change().dropna()
    if len(rets) == 0 or rets.std() == 0:
        return 0.0
    return safe_float(rets.mean() / rets.std() * np.sqrt(252))


def count_trials(db_url, alpha):
    sql = (
        "SELECT COUNT(*) FROM candidates "
        "WHERE alpha_source = %s AND strategy_name NOT LIKE 'test_%%'"
    )
    with psycopg2.connect(db_url, connect_timeout=DB_TIMEOUT) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (alpha,))
            return int(cur.fetchone()[0])


def compute_dsr(pf_oos, num_trials):
    rets = pf_oos.returns().squeeze().dropna()
    if len(rets) < 30 or rets.std() == 0:
        return {
            "dsr": 0.0,
            "p": 1.0,
            "sr_daily": 0.0,
            "skew": 0.0,
            "kurt": 3.0,
            "n": int(len(rets)),
        }
    sr_daily = float(rets.mean() / rets.std())
    skew = float(rets.skew())
    kurt = float(rets.kurtosis() + 3.0)
    dsr, p = deflated_sharpe_ratio(
        observed_sr=sr_daily,
        num_trials=num_trials,
        num_obs=len(rets),
        skewness=skew,
        kurtosis=kurt,
    )
    return {
        "dsr": float(dsr),
        "p": float(p),
        "sr_daily": sr_daily,
        "skew": skew,
        "kurt": kurt,
        "n": int(len(rets)),
    }


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


def run():
    db_url = os.environ["DATABASE_URL"]

    print("STEP 1: data", flush=True)
    try:
        price = download_price("BTC-USD")
    except Exception as e:
        print("ERROR data:", e, flush=True)
        sys.exit(1)
    print("STEP 1 OK:", len(price), "bars", flush=True)

    print("STEP 2: signal", flush=True)
    fast = vbt.MA.run(price, 10)
    slow = vbt.MA.run(price, 50)
    entries = fast.ma_crossed_above(slow)
    exits = fast.ma_crossed_below(slow)
    print("STEP 2 OK", flush=True)

    print("STEP 3: split", flush=True)
    n = len(price)
    split_idx = int(n * OOS_SPLIT)
    price_is = price.iloc[:split_idx]
    price_oos = price.iloc[split_idx:]
    entries_is = entries.iloc[:split_idx]
    entries_oos = entries.iloc[split_idx:]
    exits_is = exits.iloc[:split_idx]
    exits_oos = exits.iloc[split_idx:]
    print("STEP 3 OK: IS=", len(price_is), "OOS=", len(price_oos), flush=True)

    print("STEP 4: cost", flush=True)
    cost_results = {}
    for bps in COST_LEVELS_BPS:
        s, _, _ = run_backtest(price_oos, entries_oos, exits_oos, bps)
        cost_results[bps] = s
    oos_sharpe = cost_results[COST_KILL_BPS]
    print("STEP 4 OK:", cost_results, flush=True)

    print("STEP 5: IS Sharpe", flush=True)
    is_sharpe, _, _ = run_backtest(price_is, entries_is, exits_is, COST_KILL_BPS)
    print("STEP 5 OK:", round(is_sharpe, 3), flush=True)

    print("STEP 6: max DD", flush=True)
    _, max_dd, _ = run_backtest(price, entries, exits, COST_KILL_BPS)
    print("STEP 6 OK:", round(max_dd, 3), flush=True)

    print("STEP 7: baseline", flush=True)
    baseline_sharpe = compute_baseline_sharpe(price_oos)
    baseline_delta = oos_sharpe - baseline_sharpe
    print(
        "STEP 7 OK: baseline=",
        round(baseline_sharpe, 3),
        "delta=",
        round(baseline_delta, 3),
        flush=True,
    )

    print("STEP 8: DSR", flush=True)
    _, _, pf_oos = run_backtest(price_oos, entries_oos, exits_oos, COST_KILL_BPS)
    num_trials = count_trials(db_url, ALPHA) + 1
    dsr_info = compute_dsr(pf_oos, num_trials)
    dsr = dsr_info["dsr"]
    dsr_p = dsr_info["p"]
    print(
        "STEP 8 OK: dsr=",
        round(dsr, 3),
        "p=",
        round(dsr_p, 4),
        "trials=",
        num_trials,
        flush=True,
    )

    gate_status, reject_reason = apply_gates(
        oos_sharpe, cost_results, baseline_delta, max_dd, dsr, dsr_p
    )

    print("STEP 9: DB write", flush=True)
    spec = {
        "oos_split": OOS_SPLIT,
        "cost_sharpe": cost_results,
        "baseline_sharpe": baseline_sharpe,
        "dsr_n_obs": dsr_info["n"],
        "dsr_skew": dsr_info["skew"],
        "dsr_kurt": dsr_info["kurt"],
    }
    sql = (
        "INSERT INTO candidates "
        "(strategy_name, alpha_source, spec_json, "
        "is_sharpe, oos_sharpe, deflated_sharpe, "
        "max_drawdown, baseline_delta, gate_status, reject_reason, "
        "dsr_pvalue, num_trials) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    )
    params = (
        STRATEGY,
        ALPHA,
        json.dumps(spec),
        is_sharpe,
        oos_sharpe,
        dsr,
        max_dd,
        baseline_delta,
        gate_status,
        reject_reason,
        dsr_p,
        num_trials,
    )
    try:
        with psycopg2.connect(db_url, connect_timeout=DB_TIMEOUT) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
    except Exception as e:
        print("ERROR DB:", e, flush=True)
        sys.exit(1)
    print("STEP 9 OK", flush=True)

    print("---", flush=True)
    print("OK:", STRATEGY, "alpha=", ALPHA, flush=True)
    print("IS Sharpe:", round(is_sharpe, 3), flush=True)
    print("OOS 0bp:", round(cost_results[0], 3), flush=True)
    print("OOS 1bp:", round(cost_results[1], 3), flush=True)
    print("OOS 5bp:", round(cost_results[5], 3), flush=True)
    print("Baseline:", round(baseline_sharpe, 3), flush=True)
    print("Baseline delta:", round(baseline_delta, 3), flush=True)
    print("Max DD:", round(max_dd, 3), flush=True)
    print("DSR:", round(dsr, 3), "p=", round(dsr_p, 4), "trials=", num_trials, flush=True)
    print("Status:", gate_status, reject_reason, flush=True)

    print("", flush=True)
    print("=== Legend ===", flush=True)
    print("Sharpe           Risk-adjusted return: mean(ret)/std(ret)*sqrt(252).", flush=True)
    print("                 >1.0 excellent, >0.5 good, <0.3 weak.", flush=True)
    print("IS               In-Sample: first 70% of data used for fitting.", flush=True)
    print("OOS              Out-of-Sample: last 30% of data held back.", flush=True)
    print("bp / bps         Basis points (1 bp = 0.01%). Fee per trade.", flush=True)
    print("Baseline         Buy-and-hold Sharpe on the OOS period.", flush=True)
    print("Baseline delta   OOS Sharpe minus Baseline Sharpe.", flush=True)
    print("                 Must exceed BASELINE_MIN_DELTA to justify trading.", flush=True)
    print("Max DD           Worst peak-to-trough loss on full period.", flush=True)
    print("                 Negative; -0.30 means -30%.", flush=True)
    print("DSR              Deflated Sharpe Ratio: corrected for trials,", flush=True)
    print("                 skew and kurtosis. Higher = more credible.", flush=True)
    print("sr_daily         Per-day Sharpe inside DSR (not annualized).", flush=True)
    print("n                Number of OOS observations (trading days).", flush=True)
    print("trials           Cumulative backtest runs for this alpha source.", flush=True)
    print("                 DSR gets stricter as trials grow.", flush=True)
    print("skew             Skewness of OOS returns. Positive = good for DSR.", flush=True)
    print("kurt             Kurtosis. >3 = fat tails (penalised by DSR).", flush=True)
    print("p                DSR p-value. Must be < 0.05 to pass dsr_fail.", flush=True)
    print("", flush=True)
    print("=== Gates (reject reasons) ===", flush=True)
    print("weak_oos    OOS Sharpe < 0.30         -> mechanism fails out of sample.", flush=True)
    print("cost_kill   OOS Sharpe at 1bp < 0.10  -> dies from transaction costs.", flush=True)
    print("no_edge     Baseline delta < 0.20     -> does not beat buy-and-hold.", flush=True)
    print("max_dd      Max DD < -0.30            -> drawdown exceeds limit.", flush=True)
    print("dsr_fail    DSR <= 0 or p >= 0.05     -> not significant after correction.", flush=True)
    print("passed      All gates cleared         -> candidate for further research.", flush=True)


if __name__ == "__main__":
    run()
