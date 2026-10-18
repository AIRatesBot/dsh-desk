# DSH Candidate Pipeline

A minimal, fully-free research pipeline for filtering trading strategy
ideas into ranked candidates for further research. Runs entirely on free
cloud services, controllable from an iPad.

**This is not a production trading system.** It produces *candidates* —
strategies that have passed a set of statistical gates — not validated
strategies. A human must decide what to do with a passing candidate.

---

## What it does

1. Loads a strategy spec (from environment variables / workflow inputs).
2. Downloads data (BTC via yfinance, or FRED yield curve + TLT).
3. Builds a signal and runs a backtest.
4. Applies five sequential gates. The first failing gate determines
   the reject reason.
5. Writes the result to a shared Neon Postgres ledger.
6. Streamlit dashboard reads the ledger and shows:
   - passing candidates,
   - rejected candidates with reasons,
   - gate breakdown,
   - alpha-source breakdown,
   - DSR trend over time.

The pipeline's job is to **reject 80–95% of ideas credibly**, so the
human can focus on the remaining few.

---

## Architecture

    +-----------+      +-----------------+      +-----------+
    |  iPad     |----->|  GitHub Actions |----->|  Neon DB  |
    |  browser  |      |  (Python 3.11)  |      |  (Postgres)|
    +-----------+      +-----------------+      +-----------+
         |                     |                      ^
         |                     v                      |
         |              +-------------+               |
         |              | vectorbt    |               |
         |              | deflated-   |               |
         |              | sharpe      |               |
         |              +-------------+               |
         |                                            |
         v                                            |
    +-----------------+                               |
    | Streamlit Cloud |-------------------------------+
    | (dashboard)     |
    +-----------------+

- **iPad**: control surface only. No local compute.
- **GitHub Actions**: orchestrates backtest runs, free tier.
- **Neon**: shared ledger, free tier (1 GB).
- **Streamlit Cloud**: read-only dashboard, free tier.
- **vectorbt**: backtesting engine (Numba-accelerated).
- **deflated-sharpe**: Deflated Sharpe Ratio (Bailey & Lopez de Prado).

All services are free. No credit card required.

---

## Repository layout

    dsh-desk/
    ├── app.py                    # Streamlit dashboard (read-only)
    ├── requirements.txt          # Dashboard dependencies
    ├── backtest.py               # MOM strategy (BTC SMA cross)
    ├── backtest_carry.py         # CARRY strategy (FRED 10y-2y -> TLT)
    └── .github/workflows/
        ├── run.yml               # Workflow for backtest.py
        └── run_carry.yml         # Workflow for backtest_carry.py

Each strategy has its own file and its own workflow. **Adding a new
strategy means adding a new file pair — never editing an existing one.**

---

## The five gates

Every backtest is evaluated against five gates in this order. The first
gate that fails determines `reject_reason`.

| # | Gate        | Reject reason | Threshold                            | What it catches |
|---|-------------|---------------|--------------------------------------|-----------------|
| 1 | OOS Sharpe  | `weak_oos`    | OOS Sharpe < 0.30                    | Mechanism does not hold out of sample |
| 2 | Cost        | `cost_kill`   | OOS Sharpe at 1 bp < 0.10            | Dies from transaction costs |
| 3 | Baseline    | `no_edge`     | Baseline delta < 0.20                | Does not beat buy-and-hold |
| 4 | Drawdown    | `max_dd`      | Max DD < -0.30                       | Drawdown exceeds risk limit |
| 5 | Multi-test  | `dsr_fail`    | DSR <= 0 or p >= 0.05                | Not significant after correcting for trials |

**Why this order:** earlier gates are cheaper and more fundamental. A
strategy with no out-of-sample edge does not need to be tested for
transaction costs. A strategy that loses to buy-and-hold does not need
a drawdown review — it has already failed to justify active trading.

---

## Quick start: run a backtest

1. Go to **Actions** tab in GitHub.
2. Pick a workflow:
   - **DSH run backtest** — for `mom` strategy (BTC SMA cross).
   - **DSH run carry backtest** — for `carry` strategy (FRED + TLT).
3. Click **Run workflow**.
4. Fill inputs (for `mom`: `strategy=sma_cross_btc`, `alpha=mom`).
5. Wait ~2 minutes.
6. Open Streamlit dashboard — the new run appears in one of the tables.

### Required GitHub secrets

| Secret           | Purpose                              | Free source |
|------------------|--------------------------------------|-------------|
| `DATABASE_URL`   | Neon Postgres connection (pooled)    | console.neon.tech |
| `FRED_API_KEY`   | FRED yield curve download (carry)    | fred.stlouisfed.org |

### Required Neon schema

Run once in Neon SQL Editor:

    CREATE TABLE IF NOT EXISTS candidates (
        id              BIGSERIAL PRIMARY KEY,
        strategy_name   TEXT NOT NULL,
        alpha_source    TEXT NOT NULL,
        spec_json       JSONB,
        is_sharpe       DOUBLE PRECISION,
        oos_sharpe      DOUBLE PRECISION,
        deflated_sharpe DOUBLE PRECISION,
        max_drawdown    DOUBLE PRECISION,
        baseline_delta  DOUBLE PRECISION,
        gate_status     TEXT DEFAULT 'pending',
        reject_reason   TEXT,
        created_at      TIMESTAMPTZ DEFAULT NOW()
    );

    ALTER TABLE candidates ADD COLUMN IF NOT EXISTS dsr_pvalue DOUBLE PRECISION;
    ALTER TABLE candidates ADD COLUMN IF NOT EXISTS num_trials INTEGER;

    CREATE INDEX idx_candidates_status ON candidates (gate_status);
    CREATE INDEX idx_candidates_alpha  ON candidates (alpha_source);

---

## Adding a new alpha source

Do **not** edit `backtest.py` or `backtest_carry.py`. Instead:

1. Copy `backtest_carry.py` to `backtest_<name>.py`.
2. Replace `load_carry_data()` with your data + signal logic.
3. Keep the shared structure: gates, DSR, DB write.
4. Copy `.github/workflows/run_carry.yml` to `run_<name>.yml`.
5. Point it at `backtest_<name>.py`.
6. Add `FRED_API_KEY`-like secret only if your data source needs it.
7. Run from Actions tab.

The new strategy shares the ledger and dashboard automatically —
`alpha_source` is set to whatever you pass to the workflow.

---

## Dashboard

`app.py` uses Streamlit, connects to Neon via `st.connection`, and is
**read-only** — no computations. It shows six sections:

1. **Top candidates** — passing strategies, sorted by DSR.
2. **Rejected** — rejected strategies with reasons.
3. **Gate breakdown** — how many runs failed at each gate (last 30 days).
4. **Alpha source breakdown** — runs and pass count per alpha source,
   with a warning when any source has `trials > 15`.
5. **DSR trend** — DSR vs OOS Sharpe over time, per run.
6. **Disclaimer** — data coverage and what "passed" means.

### Streamlit secrets

    [connections.neon]
    url = "postgresql://neondb_owner:...@ep-xxx-pooler...neon.tech/neondb?sslmode=require"

    [connections.neon.create_engine_kwargs]
    pool_pre_ping = true
    pool_recycle = 300

`pool_pre_ping` and `pool_recycle` prevent "SSL connection has been
closed unexpectedly" errors caused by Neon's serverless auto-suspend.

---

## Design principles

1. **Candidates, not strategies.** A passing strategy has cleared five
   gates. It has not been validated on new data, slippage, capacity, or
   regime stability. Label it correctly on the dashboard.
2. **Tuned to reject, not to accept.** Precision over recall. The human
   research budget is the bottleneck, so false negatives are cheaper
   than false positives.
3. **DSR is mandatory.** Without it, running N tests produces N chances
   of a false positive. DSR deflates Sharpe by the number of trials,
   skewness, and kurtosis. Ignoring it turns the pipeline into a false-
   confidence machine.
4. **Isolated strategies.** Each alpha source lives in its own file and
   workflow. Adding or removing one cannot break another.
5. **One shared ledger.** All strategies write to the same `candidates`
   table. The dashboard shows them together.

---

## Current strategies

| File                  | Alpha   | Signal                          | Instrument | Trials |
|-----------------------|---------|---------------------------------|------------|--------|
| `backtest.py`         | `mom`   | SMA 10 / 50 cross               | BTC-USD    | 14     |
| `backtest_carry.py`   | `carry` | DGS10-DGS2 z-score > 0.5        | TLT        | 1      |

**Status on latest run:**

- `sma_cross_btc` — rejected `no_edge` (baseline delta 0.04, also
  `max_dd` -0.72 and `dsr_fail` p=0.45). Beta strategy, not alpha.
- `yield_curve_carry_tlt` — rejected `weak_oos` (OOS Sharpe 0.13).
  Defensive overlay, not standalone alpha. Baseline delta 0.58 shows
  it avoided the 2022 TLT drawdown, but did not generate return.

**Both rejections are valid outcomes.** The pipeline exists to reject
ideas, not to find them.

---

## Limitations

1. **BTC and TLT only.** No swap rates, no breakevens, no vol surfaces.
   Any strategy that needs those cannot be tested here.
2. **TLT is an ETF proxy, not a swap.** Carry strategy is
   indicative, not production-equivalent.
3. **Trials grow permanently.** Once `mom` has 14 trials, all future
   `mom` strategies are deflated by 14. Eventually no `mom` strategy
   can pass the DSR gate. This is intentional — it prevents
   "test until something works" behavior. To get a clean slate, use a
   new `alpha_source`.
4. **No slippage model.** Costs are symmetric bps. Real fills have
   market impact and spreads.
5. **No regime analysis.** A strategy that passes gates may still only
   work in one regime (e.g. low-vol). The pipeline does not detect
   this.
6. **No independent replication.** The same code generates and
   evaluates. A Replicator using a different model and library would
   catch silent bugs that static analysis misses.
7. **iOS paste corruption risk.** When editing code on iPad, use file
   upload (Files app -> GitHub "Upload files") rather than pasting
   into the GitHub text editor. iOS Smart Punctuation and QuickType
   can corrupt long pastes (e.g. `0.0` -> `0A.0`, `"dsr"` -> `".Mdsr"`).

---

## What the pipeline does NOT do

- It does not write strategies from papers. That is a manual step.
- It does not optimize parameters. It tests one configuration per run.
- It does not validate a strategy on new data. A passing candidate
  still requires forward testing.
- It does not assess capacity. A strategy that works on 10,000 USD may
  not work on 10 million.
- It does not check correlation with existing book. A "new" candidate
  may be highly correlated with what the desk already trades.

---

## Next steps (optional)

In rough order of value per hour:

1. **Iterate carry.** Try `Z_ENTRY = 0.3` or `Z_WINDOW = 63`. Each
   iteration adds 1 trial; stop after 3-5.
2. **Carry as overlay, not standalone.** Keep TLT always long, scale
   position by z-score. This matches how carry is actually traded.
3. **Spec generation with Groq.** Add `spec_groq.py` that reads a
   paper abstract and produces `spec_json`. Human still approves.
4. **Replicator.** Add a second LLM (different provider) that
   independently re-implements the signal. Compare backtests. This is
   the highest-value addition but also the most work.
5. **Regime tags.** Add a `regime` column to `candidates` (e.g.
   "low_vol", "high_vol") and show regime breakdown in dashboard.

---

## Free-tier limits

| Service           | Free limit                              | Notes |
|-------------------|-----------------------------------------|-------|
| GitHub Actions    | 2000 min/mo (private) or unlimited (public) | Keep repo public for unlimited minutes |
| Neon              | 1 GB storage, auto-suspend after 5 min  | Fine for this workload |
| Streamlit Cloud   | 1 GB RAM per app, unlimited public apps | Use Chrome on iOS (Safari + Streamlit unreliable on iOS 26) |
| Groq (if used)    | ~14400 requests/day                     | Enough for spec generation |
| FRED              | Free, no daily limit published          | Standard API key |

---

## Contact / maintenance

This is a research tool. If the pipeline produces an unexpected result,
check the `spec_json` column in Neon — it records every parameter used
in the run, so any run is fully reproducible.

**Never delete rows from `candidates`.** The trial count is derived from
row count. Deleting rows silently resets the DSR deflation and
invalidates every prior run's statistics.
