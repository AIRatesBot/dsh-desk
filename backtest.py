import os
import json
import numpy as np
import vectorbt as vbt
import psycopg2

STRATEGY = os.environ.get("STRATEGY", "sma_cross_btc")
ALPHA    = os.environ.get("ALPHA", "mom")
OOS_SPLIT = 0.70
OOS_MIN_SHARPE = 0.30
COST_LEVELS_BPS = [0, 1, 5]      # testattavat kustannustasot
COST_KILL_BPS   = 1              # jos Sharpe < tämä 1 bp:llä → hylkäys


def safe_float(x):
    try:
        v = float(x)
        return v if not np.isnan(v) else 0.0
    except Exception:
        return 0.0


def run_backtest(price, entries, exits, fees_bps: float):
    """Ajaa yhden backtestin annetuilla kustannuksilla. Palauttaa (sharpe, max_dd)."""
    pf = vbt.Portfolio.from_signals(
        price, entries, exits,
        init_cash=10_000,
        fees=fees_bps / 10_000.0,   # bp → prosentti
        freq="1D",
    )
    return safe_float(pf.sharpe_ratio()), safe_float(pf.max_drawdown())


def run():
    # 1. Data
    data  = vbt.YFData.download("BTC-USD")
    price = data.get("Close")

    # 2. Signaali
    fast = vbt.MA.run(price, 10)
    slow = vbt.MA.run(price, 50)
    entries = fast.ma_crossed_above(slow)
    exits   = fast.ma_crossed_below(slow)

    # 3. IS / OOS-jako
    n = len(price)
    split_idx = int(n * OOS_SPLIT)

    price_is,   price_oos   = price.iloc[:split_idx],  price.iloc[split_idx:]
    entries_is, entries_oos = entries.iloc[:split_idx], entries.iloc[split_idx:]
    exits_is,   exits_oos   = exits.iloc[:split_idx],   exits.iloc[split_idx:]

    # 4. Kustannusherkkyys OOS-jaksolla (portti pätee OOS:ään)
    cost_results = {}
    for bps in COST_LEVELS_BPS:
        sharpe, _ = run_backtest(price_oos, entries_oos, exits_oos, bps)
        cost_results[bps] = sharpe

    oos_sharpe = cost_results[COST_KILL_BPS]   # 1 bp on "referenssi-OOS"

    # 5. IS-Sharpe 1 bp:llä (raportointia varten)
    is_sharpe, _ = run_backtest(price_is, entries_is, exits_is, COST_KILL_BPS)

    # 6. Koko periodi drawdown (1 bp)
    _, max_dd = run_backtest(price, entries, exits, COST_KILL_BPS)

    # 7. Portit järjestyksessä
    if oos_sharpe < OOS_MIN_SHARPE:
        gate_status   = "rejected"
        reject_reason = "weak_oos"
    elif cost_results[COST_KILL_BPS] < 0.10:
        # Strategia kuolee jo 1 bp:ssä → ei kestä kustannuksia
        gate_status   = "rejected"
        reject_reason = "cost_kill"
    else:
        gate_status   = "passed"
        reject_reason = None

    # 8. Placeholderit (4c–4d)
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
                        "oos_split": OOS_SPLIT,
                        "oos_min_sharpe": OOS_MIN_SHARPE,
                        "cost_levels_bps": COST_LEVELS_BPS,
                        "cost_sharpe": cost_results,
                    }),
                    is_sharpe, oos_sharpe, deflated_sharpe,
                    max_dd, baseline_delta,
                    gate_status, reject_reason,
                ),
            )

    # 10. Loki
    print(f"OK: {STRATEGY}")
    print(f"  IS Sharpe (1 bp)   = {is_sharpe:.3f}")
    print(f"  OOS Sharpe 0 bp    = {cost_results[0]:.3f}")
    print(f"  OOS Sharpe 1 bp    = {cost_results[1]:.3f}  (threshold {OOS_MIN_SHARPE})")
    print(f"  OOS Sharpe 5 bp    = {cost_results[5]:.3f}")
    print(f"  Max DD (1 bp)      = {max_dd:.3f}")
    print(f"  Status             = {gate_status}"
          + (f" ({reject_reason})" if reject_reason else ""))


if __name__ == "__main__":
    run()
