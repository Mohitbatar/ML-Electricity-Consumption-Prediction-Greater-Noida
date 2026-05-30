import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import (
    section, COLORS, MONTH_SHORT, apply_layout, MODEL_META,
    active_models, build_forecasts, insight, no_models_warn, warn
)

def tab_future(hh, comm, ctrl):
    active       = active_models(ctrl)
    ds           = ctrl['dataset']
    future_years = ctrl['future_years']
    steps        = future_years * 12

    section(f"Long-range forecast — {future_years} year{'s' if future_years > 1 else ''} ahead")
    st.caption(
        f"Projecting {steps} months ({future_years} years) from Jan 2026 baseline "
        "using active models"
    )

    if ds == "Household":
        grouped = hh.groupby(['Year', 'month_num'])['Units Consumed'].mean().sort_index()
        unit   = "kWh / month (Household)"
    else:
        grouped = comm.groupby(['Year', 'month_num'])['Units Consumed (After Solar)'].mean().sort_index()
        unit   = "kWh / month (Commercial)"

    series = grouped.values
    if not active:
        no_models_warn()
        return

    last_year, last_month = grouped.index[-1]
    future_labels = []
    for i in range(1, steps + 1):
        m = last_month + i
        y = last_year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        future_labels.append(f"{MONTH_SHORT[m-1]} {y}")

    fc_all = build_forecasts(series, steps, ds, hh, comm, ctrl.get('growth_rate', 4.0))

    fig = go.Figure()
    for m in active:
        fc   = [float(v) for v in fc_all[m]]
        meta = MODEL_META[m]
        if m == 'SARIMA':
            std_val = float(np.std(series)) * 0.15
            upper   = [v + std_val * (1 + i * 0.04) for i, v in enumerate(fc)]
            lower   = [max(50.0, v - std_val * (1 + i * 0.04)) for i, v in enumerate(fc)]
            hex_c = meta['color'].lstrip('#')
            r, g, b = int(hex_c[0:2],16), int(hex_c[2:4],16), int(hex_c[4:6],16)
            fig.add_trace(go.Scatter(
                x=future_labels + future_labels[::-1],
                y=upper + lower[::-1],
                fill='toself', fillcolor=f"rgba({r},{g},{b},0.12)",
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False, name='SARIMA CI'))
        fig.add_trace(go.Scatter(
            x=future_labels, y=fc, name=m,
            line=dict(color=meta['color'], width=2.5, dash=meta['dash']),
            mode='lines+markers', marker=dict(size=3)))

    fig.update_xaxes(tickangle=45, tickfont_size=10, tickmode='array',
                     tickvals=future_labels[::6])
    apply_layout(fig, height=440, yaxis_title=unit,
                 title=f"{ds} — {future_years}-Year Future Energy Forecast")
    st.plotly_chart(fig, use_container_width=True)

    section("Annual aggregated forecast — active models")
    annual_rows = []
    for yr_idx in range(future_years):
        row = {'Year': str(2026 + yr_idx)}
        month_slice = slice(yr_idx * 12, (yr_idx + 1) * 12)
        for m in active:
            fc_yr = [float(v) for v in fc_all[m][month_slice]]
            if fc_yr:
                row[f"{m} — Annual kWh"]   = round(sum(fc_yr), 0)
                row[f"{m} — Monthly Avg"]  = round(float(np.mean(fc_yr)), 1)
            else:
                row[f"{m} — Annual kWh"]   = "—"
                row[f"{m} — Monthly Avg"]  = "—"
        annual_rows.append(row)
    annual_df = pd.DataFrame(annual_rows).set_index('Year')
    st.dataframe(annual_df, use_container_width=True)

    section("Month-by-month detail — Year 1 (2026) — active models")
    yr1_rows = []
    for i in range(min(12, steps)):
        row = {'Month': MONTH_SHORT[i]}
        for m in active:
            row[m] = round(float(fc_all[m][i]), 1)
        yr1_rows.append(row)
    yr1_df = pd.DataFrame(yr1_rows).set_index('Month')
    st.dataframe(yr1_df, use_container_width=True)

    section("Year-over-year growth trend — active models")
    if future_years >= 2:
        growth_rows = []
        for m in active:
            fc_vals  = [float(v) for v in fc_all[m]]
            yr_totals = []
            for y in range(future_years):
                chunk = fc_vals[y * 12: (y + 1) * 12]
                yr_totals.append(sum(chunk) if chunk else 0.0)

            growth_row = {'Model': m}
            for y in range(1, future_years):
                prev = yr_totals[y - 1]
                curr = yr_totals[y]
                if prev != 0:
                    pct = (curr - prev) / prev * 100
                    growth_row[f"{2026+y-1} → {2026+y}"] = f"{pct:+.1f}%"
                else:
                    growth_row[f"{2026+y-1} → {2026+y}"] = "N/A"
            growth_rows.append(growth_row)

        growth_df = pd.DataFrame(growth_rows).set_index('Model')
        st.dataframe(growth_df, use_container_width=True)
    else:
        warn("Set 'Years to project' to 2 or more in the sidebar to see growth trends.")

    section("Peak month prediction — active models")
    peak_rows = []
    for m in active:
        fc = [float(v) for v in fc_all[m]]
        if not fc:
            continue
        peak_idx   = int(np.argmax(fc))
        trough_idx = int(np.argmin(fc))
        peak_rows.append({
            'Model':        m,
            'Peak Month':   future_labels[peak_idx],
            'Peak kWh':     round(fc[peak_idx], 1),
            'Trough Month': future_labels[trough_idx],
            'Trough kWh':   round(fc[trough_idx], 1),
            'Range kWh':    round(fc[peak_idx] - fc[trough_idx], 1),
        })
    if peak_rows:
        peak_df = pd.DataFrame(peak_rows).set_index('Model')
        st.dataframe(peak_df, use_container_width=True)

    insight(
        f"<b>Interpretation:</b> The {future_years}-year projection shows "
        "energy demand continuing to rise, driven by population growth and increasing "
        "appliance penetration in Greater Noida. "
        "<b>SARIMA</b> preserves seasonal cycles with widening confidence bands over time. "
        "<b>XGBoost</b> repeats the learned seasonal pattern. "
        "<b>ARIMA</b> projects a gradual drift toward the long-run mean. "
        "Use the Annual Aggregated table above for grid planning and infrastructure sizing."
    )
