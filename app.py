import streamlit as st

st.set_page_config(page_title="DSH Desk", layout="wide")
st.title("DSH — Candidate Monitor")

conn = st.connection("neon", type="sql")

# ---------------------------------------------------------------
# Top candidates for further research
# ---------------------------------------------------------------
st.subheader("Top candidates for further research")
df = conn.query("""
    SELECT strategy_name, alpha_source, is_sharpe, oos_sharpe,
           deflated_sharpe, baseline_delta, created_at
    FROM candidates
    WHERE gate_status = 'passed'
    ORDER BY deflated_sharpe DESC NULLS LAST
    LIMIT 20;
""", ttl="30s")

if df.empty:
    st.info("No passing candidates yet.")
else:
    st.dataframe(df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
# Rejected candidates with reason codes
# ---------------------------------------------------------------
st.subheader("Rejected — reason")
rej = conn.query("""
    SELECT strategy_name, alpha_source, reject_reason, created_at
    FROM candidates
    WHERE gate_status = 'rejected'
    ORDER BY created_at DESC
    LIMIT 50;
""", ttl="30s")

if rej.empty:
    st.caption("No rejected candidates.")
else:
    st.dataframe(rej, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
# Pipeline health (last 30 days)
# ---------------------------------------------------------------
st.subheader("Pipeline health (last 30 days)")
health = conn.query("""
    SELECT
      COUNT(*) FILTER (WHERE gate_status = 'passed')   AS passed,
      COUNT(*) FILTER (WHERE gate_status = 'rejected') AS rejected,
      COUNT(*)                                          AS total
    FROM candidates
    WHERE created_at > NOW() - INTERVAL '30 days';
""", ttl="60s")
st.dataframe(health, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
# Data coverage disclaimer
# ---------------------------------------------------------------
st.divider()
st.caption(
    "⚠️ Data coverage: US Treasury yield curve only. "
    "Swap rates, breakevens and vol surfaces are NOT included. "
    "A 'passed' candidate is a candidate for further research — not a validated strategy."
)
