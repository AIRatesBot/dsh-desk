import streamlit as st
import pandas as pd

st.set_page_config(page_title="DSH Desk", layout="wide")
st.title("DSH - Candidate Monitor")

conn = st.connection("neon", type="sql")


# ---------------------------------------------------------------
# Schema check
# ---------------------------------------------------------------
try:
    schema_df = conn.query("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'candidates'
        ORDER BY ordinal_position;
    """, ttl="300s")
    available = set(schema_df["column_name"].tolist())
except Exception as e:
    st.error("Cannot query Neon schema: " + str(e))
    st.stop()

required = {"dsr_pvalue", "num_trials", "max_drawdown", "deflated_sharpe"}
missing = required - available
if missing:
    st.error("Missing columns in 'candidates': " + ", ".join(sorted(missing)))
    st.info("Run in Neon SQL Editor: ALTER TABLE candidates ADD COLUMN IF NOT EXISTS dsr_pvalue DOUBLE PRECISION; ALTER TABLE candidates ADD COLUMN IF NOT EXISTS num_trials INTEGER;")
    st.stop()


# ---------------------------------------------------------------
# 1. Top candidates
# ---------------------------------------------------------------
st.subheader("Top candidates for further research")
df = conn.query("""
    SELECT strategy_name, alpha_source, is_sharpe, oos_sharpe,
           deflated_sharpe, dsr_pvalue, num_trials,
           baseline_delta, max_drawdown, created_at
    FROM candidates
    WHERE gate_status = 'passed'
      AND LEFT(strategy_name, 5) <> 'test_'
    ORDER BY deflated_sharpe DESC NULLS LAST
    LIMIT 20;
""", ttl="30s")

if df.empty:
    st.info("No passing candidates yet.")
else:
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------
# 2. Rejected
# ---------------------------------------------------------------
st.subheader("Rejected - reason")
rej = conn.query("""
    SELECT strategy_name, alpha_source, reject_reason,
           oos_sharpe, deflated_sharpe, dsr_pvalue,
           baseline_delta, max_drawdown, created_at
    FROM candidates
    WHERE gate_status = 'rejected'
      AND LEFT(strategy_name, 5) <> 'test_'
    ORDER BY created_at DESC
    LIMIT 50;
""", ttl="30s")

if rej.empty:
    st.caption("No rejected candidates.")
else:
    st.dataframe(rej, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------
# 3. Gate breakdown
# ---------------------------------------------------------------
st.subheader("Gate breakdown - last 30 days")
gb = conn.query("""
    SELECT
      COUNT(*) FILTER (WHERE gate_status = 'passed')      AS passed,
      COUNT(*) FILTER (WHERE reject_reason = 'weak_oos')  AS weak_oos,
      COUNT(*) FILTER (WHERE reject_reason = 'cost_kill') AS cost_kill,
      COUNT(*) FILTER (WHERE reject_reason = 'no_edge')   AS no_edge,
      COUNT(*) FILTER (WHERE reject_reason = 'max_dd')    AS max_dd,
      COUNT(*) FILTER (WHERE reject_reason = 'dsr_fail')  AS dsr_fail,
      COUNT(*)                                             AS total
    FROM candidates
    WHERE created_at > NOW() - INTERVAL '30 days'
      AND LEFT(strategy_name, 5) <> 'test_';
""", ttl="60s")

if gb.empty or int(gb.iloc[0]["total"]) == 0:
    st.caption("No runs in the last 30 days.")
else:
    row = gb.iloc[0]
    total = int(row["total"])
    passed = int(row["passed"])
    cols = st.columns(6)
    cols[0].metric("Passed", passed)
    cols[1].metric("weak_oos", int(row["weak_oos"]))
    cols[2].metric("cost_kill", int(row["cost_kill"]))
    cols[3].metric("no_edge", int(row["no_edge"]))
    cols[4].metric("max_dd", int(row["max_dd"]))
    cols[5].metric("dsr_fail", int(row["dsr_fail"]))
    pass_rate = (passed / total) * 100 if total > 0 else 0
    st.caption("Total: " + str(total) + "  |  Pass rate: " + str(round(pass_rate, 1)) + "%")
    if pass_rate > 50:
        st.warning("Pass rate above 50%. Gates may be too loose.")
    elif pass_rate == 0 and total > 10:
        st.info("No passes across " + str(total) + " runs. Gates may be too strict or strategies genuinely lack edge.")


# ---------------------------------------------------------------
# 4. Alpha source breakdown
# ---------------------------------------------------------------
st.subheader("Alpha source breakdown")
ab = conn.query("""
    SELECT alpha_source,
           COUNT(*) AS runs,
           COUNT(*) FILTER (WHERE gate_status = 'passed') AS passed,
           MAX(num_trials) AS max_trials,
           MAX(created_at) AS last_run
    FROM candidates
    WHERE LEFT(strategy_name, 5) <> 'test_'
    GROUP BY alpha_source
    ORDER BY runs DESC;
""", ttl="60s")

if ab.empty:
    st.caption("No runs yet.")
else:
    st.dataframe(ab, use_container_width=True, hide_index=True)
    for _, r in ab.iterrows():
        mt = r["max_trials"]
        if pd.notna(mt) and int(mt) > 15:
            st.warning("Alpha source '" + str(r["alpha_source"]) + "' has " + str(int(mt)) + " trials. DSR gate is now very strict.")


# ---------------------------------------------------------------
# 5. DSR trend
# ---------------------------------------------------------------
st.subheader("DSR trend - last 50 runs")
trend = conn.query("""
    SELECT created_at, oos_sharpe, deflated_sharpe
    FROM candidates
    WHERE LEFT(strategy_name, 5) <> 'test_'
      AND deflated_sharpe IS NOT NULL
    ORDER BY created_at ASC
    LIMIT 50;
""", ttl="60s")

if trend.empty:
    st.caption("No data yet.")
else:
    trend = trend.set_index("created_at")
    st.line_chart(trend)


# ---------------------------------------------------------------
# 6. Disclaimer
# ---------------------------------------------------------------
st.divider()
st.caption("Data coverage: BTC daily closes (yfinance). Swap rates, breakevens and vol surfaces NOT included. A 'passed' candidate is a candidate for further research - not a validated strategy.")
