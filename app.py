import streamlit as st

st.set_page_config(page_title="DSH Desk", layout="wide")
st.title("DSH — Candidate Monitor")

conn = st.connection("neon", type="sql")

# ---------------------------------------------------------------
# Top candidates for further research
# ---------------------------------------------------------------
st.subheader("Top candidates for further research")
df = conn.query("""
    SELECT strategy_name,
           alpha_source,
           is_sharpe,
           oos_sharpe,
           deflated_sharpe,
           dsr_pvalue,
           num_trials,
           baseline_delta,
           max_drawdown,
           created_at
    FROM candidates
    WHERE gate_status = 'passed'
    ORDER BY deflated_sharpe DESC NULLS LAST
    LIMIT 20;
""", ttl="30s")

if df.empty:
    st.info("No passing candidates yet.")
else:
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "strategy_name":  st.column_config.TextColumn("Strategy"),
            "alpha_source":   st.column_config.TextColumn("Alpha"),
            "is_sharpe":      st.column_config.NumberColumn("IS Sharpe",    format="%.3f"),
            "oos_sharpe":     st.column_config.NumberColumn("OOS Sharpe",   format="%.3f"),
            "deflated_sharpe": st.column_config.NumberColumn("DSR",         format="%.3f",
                                  help="Deflated Sharpe Ratio (multi-test corrected)"),
            "dsr_pvalue":     st.column_config.NumberColumn("DSR p",        format="%.4f",
                                  help="p-value; must be < 0.05 to pass"),
            "num_trials":     st.column_config.NumberColumn("Trials",       format="%d"),
            "baseline_delta": st.column_config.NumberColumn("vs B&H",       format="%+.3f"),
            "max_drawdown":   st.column_config.NumberColumn("Max DD",       format="%.3f"),
            "created_at":     st.column_config.DatetimeColumn("Created",    format="YYYY-MM-DD HH:mm"),
        },
    )

# ---------------------------------------------------------------
# Rejected candidates with reason codes
# ---------------------------------------------------------------
st.subheader("Rejected — reason")
rej = conn.query("""
    SELECT strategy_name,
           alpha_source,
           reject_reason,
           oos_sharpe,
           deflated_sharpe,
           dsr_pvalue,
           baseline_delta,
           max_drawdown,
           created_at
    FROM candidates
    WHERE gate_status = 'rejected'
    ORDER BY created_at DESC
    LIMIT 50;
""", ttl="30s")

if rej.empty:
    st.caption("No rejected candidates.")
else:
    st.dataframe(
        rej,
        use_container_width=True,
        hide_index=True,
        column_config={
            "strategy_name":  st.column_config.TextColumn("Strategy"),
            "alpha_source":   st.column_config.TextColumn("Alpha"),
            "reject_reason":  st.column_config.TextColumn("Reason"),
            "oos_sharpe":     st.column_config.NumberColumn("OOS Sharpe", format="%.3f"),
            "deflated_sharpe": st.column_config.NumberColumn("DSR",       format="%.3f"),
            "dsr_pvalue":     st.column_config.NumberColumn("DSR p",      format="%.4f"),
            "baseline_delta": st.column_config.NumberColumn("vs B&H",     format="%+.3f"),
            "max_drawdown":   st.column_config.NumberColumn("Max DD",     format="%.3f"),
            "created_at":     st.column_config.DatetimeColumn("Created",  format="YYYY-MM-DD HH:mm"),
        },
    )

# ---------------------------------------------------------------
# Gate pass-rate breakdown
# ---------------------------------------------------------------
st.subheader("Gate pass rate — last 30 days")

breakdown = conn.query("""
    SELECT
      COUNT(*) FILTER (WHERE gate_status = 'passed')                              AS passed,
      COUNT(*) FILTER (WHERE reject_reason = 'weak_oos')                          AS weak_oos,
      COUNT(*) FILTER (WHERE reject_reason = 'cost_kill')                         AS cost_kill,
      COUNT(*) FILTER (WHERE reject_reason = 'no_edge')                           AS no_edge,
      COUNT(*) FILTER (WHERE reject_reason = 'max_dd')                            AS max_dd,
      COUNT(*) FILTER (WHERE reject_reason = 'dsr_fail')                          AS dsr_fail,
      COUNT(*)                                                                    AS total
    FROM candidates
    WHERE created_at > NOW() - INTERVAL '30 days'
      AND strategy_name NOT LIKE 'test_%';
""", ttl="60s")

st.dataframe(breakdown, use_container_width=True, hide_index=True)

# Pipeline health warning
if not breakdown.empty and breakdown.iloc[0]["total"] > 0:
    total = int(breakdown.iloc[0]["total"])
    passed = int(breakdown.iloc[0]["passed"] or 0)
    pass_rate = passed / total
    if pass_rate > 0.50:
        st.warning(
            f"⚠️ Pass rate {pass_rate:.0%} is above 50%. Gates may be too loose — "
            f"review thresholds or check for data leakage."
        )
    elif pass_rate == 0 and total > 10:
        st.info(
            f"ℹ️ Pass rate is 0% across {total} runs. Gates may be too strict, "
            f"or the tested strategies genuinely lack edge."
        )

# ---------------------------------------------------------------
# Data coverage disclaimer
# ---------------------------------------------------------------
st.divider()
st.caption(
    "⚠️ Data coverage: US Treasury yield curve and BTC daily closes only. "
    "Swap rates, breakevens and vol surfaces are NOT included. "
    "A 'passed' candidate is a candidate for further research — "
    "not a validated strategy."
)
