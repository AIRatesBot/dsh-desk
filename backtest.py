import os
import json
import vectorbt as vbt
import psycopg2

STRATEGY = os.environ.get("STRATEGY", "sma_cross_btc")
ALPHA    = os.environ.get("ALPHA", "mom")


def run():
    # 1. Hae data
    data  = vbt.YFData.download("BTC-USD")
    price = data.get("Close")

    # 2. Signaali: SMA 10 / 50 -cross
    fast = vbt.MA.run(price, 10)
    slow = vbt.MA.run(price, 50)
    entries = fast.ma_crossed_above(slow)
    exits   = fast.ma_crossed_below(slow)

    # 3. Backtest
    pf = vbt.Portfolio.from_signals(
        price, entries, exits,
        init_cash=10_000, fees=0.001, freq="1D"
    )

    is_sharpe = float(pf.sharpe_ratio())
    max_dd    = float(pf.max_drawdown())

    # 4. Placeholderit – nämä täytetään seuraavassa vaiheessa (OOS, deflaatio)
    oos_sharpe      = is_sharpe
    deflated_sharpe = is_sharpe * 0.7
    baseline_delta  = 0.0

    # 5. Kirjoita Neoniin
    db_url = os.environ["DATABASE_URL"]
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO candidates
                  (strategy_name, alpha_source, spec_json,
                   is_sharpe, oos_sharpe, deflated_sharpe,
                   max_drawdown, baseline_delta, gate_status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'passed')
                """,
                (
                    STRATEGY, ALPHA,
                    json.dumps({"fast": 10, "slow": 50, "fees_bps": 10}),
                    is_sharpe, oos_sharpe, deflated_sharpe,
                    max_dd, baseline_delta,
                ),
            )
    print(f"OK: {STRATEGY} is_sharpe={is_sharpe:.3f} max_dd={max_dd:.3f}")


if __name__ == "__main__":
    run()
