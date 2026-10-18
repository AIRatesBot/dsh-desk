import os
import json
import numpy as np
import vectorbt as vbt
import psycopg2

STRATEGY = os.environ.get("STRATEGY", "sma_cross_btc")
ALPHA    = os.environ.get("ALPHA", "mom")
OOS_SPLIT = 0.70          # 70 % in-sample, 30 % out-of-sample
OOS_MIN_SHARPE = 0.30     # portti: alle tämän → hylkäys


def safe_float(x):
    """Palauttaa 0.0 jos arvo on NaN tai ei muunnu."""
    try:
        v = float(x)
        return v if not np.isnan(v) else 0.0
    except Exception:
        return 0.0


def run():
    # 1. Data
    data  = vbt.YFData.download("BTC-USD")
    price = data.get("Close")

    # 2. Signaali (kausaalinen – käyttää vain mennyttä dataa)
    fast = vbt.MA.run(price, 10)
    slow = vbt.MA.run(price, 50)
    entries = fast.ma_crossed_above(slow)
    exits   = fast.ma_crossed_below(slow)

    # 3. Jaa data IS / OOS
    n = len(price)
    split_idx = int(n * OOS_SPLIT)

    price_is,   price_oos   = price.iloc[:split_idx],  price.iloc[split_idx:]
    entries_is, entries_oos = entries.iloc[:split_idx], entries.iloc[split_idx:]
    exits_is,   exits_oos   = exits.iloc[:split_idx],   exits.iloc[split_idx:]

    # 4. Backtest IS
    pf_is = vbt.Portfolio.from_signals(
        price_is, entries_is, exits_is,
        init_cash=10_000, fees=0.001, freq="1D"
    )
    is_sharpe = safe_float(pf_is.sharpe_ratio())

    # 5. Backtest OOS
    pf_oos = vbt.Portfolio.from_signals(
        price_oos, entries_oos, exits_oos,
        init_cash=10_000, fees=0.001, freq="1D"
    )
    oos_sharpe = safe_float(pf_oos.sharpe_ratio())

    # 6. Koko periodin drawdown (raportointia varten)
    pf_full = vbt.Portfolio.from_signals(
        price, entries, exits,
        init_cash=10_000, fees=0.001, freq="1D"
    )
    max_dd = safe_float(pf_full.max_drawdown())

    # 7. Portti: OOS-Sharpe
    if oos_sharpe < OOS_MIN_SHARPE:
        gate_status  = "rejected"
        reject_reason = "weak_oos"
    else:
        gate_status  = "passed"
        reject_reason = None

    # 8. Placeholderit (täytetään 4b–4d)
    deflated_sharpe = oos_sharpe * 0.7
    baseline_delta  = 0.0

    # 9. Kirjoita Neoniin
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
                        "fast": 10, "slow": 50,
                        "fees_bps": 10,
                        "oos_split": OOS_SPLIT,
                        "oos_min_sharpe": OOS_MIN_SHARPE,
                    }),
                    is_sharpe, oos_sharpe, deflated_sharpe,
                    max_dd, baseline_delta,
                    gate_status, reject_reason,
                ),
            )

    print(f"OK: {STRATEGY}")
    print(f"  IS Sharpe  = {is_sharpe:.3f}")
    print(f"  OOS Sharpe = {oos_sharpe:.3f}  (threshold {OOS_MIN_SHARPE})")
    print(f"  Max DD     = {max_dd:.3f}")
    print(f"  Status     = {gate_status}"
          + (f" ({reject_reason})" if reject_reason else ""))


if __name__ == "__main__":
    run()
