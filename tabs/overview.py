import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import (
    section, metric_card, COLORS, MONTH_SHORT, apply_layout,
    active_models, build_forecasts, insight, no_models_warn
)

def tab_overview(hh, comm, ctrl):
    active = active_models(ctrl)
    section("Dataset summary")

    c1, c2, c3, c4 = st.columns(4)
    avg_hh    = hh['Units Consumed'].mean()
    avg_comm  = comm['Units Consumed (After Solar)'].mean()
    solar_pct = (hh['Solar'] == 'Yes').mean() * 100
    metric_card(c1, f"{avg_hh:.0f}",    "Avg Household kWh",    "monthly avg")
    metric_card(c2, f"{avg_comm:,.0f}", "Avg Commercial kWh",   "monthly avg")
    metric_card(c3, f"{solar_pct:.0f}%","Solar Adoption",       "of households")
    metric_card(c4, f"{len(active)}",   "Active models",        "of 3 selected")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        section("Monthly average consumption")
        hh_m   = hh.groupby('month_num')['Units Consumed'].mean().sort_index()
        comm_m = comm.groupby('month_num')['Units Consumed (After Solar)'].mean().sort_index()
        hh_vals   = [hh_m.get(i, 0) for i in range(1, 13)]
        comm_vals = [comm_m.get(i, 0) / 5 for i in range(1, 13)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=MONTH_SHORT, y=hh_vals, name="Household (kWh)",
                                  line=dict(color=COLORS['teal'], width=2.5),
                                  fill='tozeroy', fillcolor='rgba(29,184,122,0.14)',
                                  mode='lines+markers', marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=MONTH_SHORT, y=comm_vals, name="Commercial / 5",
                                  line=dict(color=COLORS['blue'], width=2, dash='dash'),
                                  mode='lines+markers', marker=dict(size=5)))
        apply_layout(fig, height=280, yaxis_title="kWh / month")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        section("Consumption by season")
        hh_s   = hh.groupby('Season')['Units Consumed'].mean().round(1)
        comm_s = comm.groupby('Season')['Units Consumed (After Solar)'].mean().round(1)
        seasons = sorted(list(set(hh_s.index) | set(comm_s.index)))
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Household", x=seasons,
                             y=[hh_s.get(s, 0) for s in seasons],
                             marker_color=COLORS['teal'], opacity=0.9))
        fig.add_trace(go.Bar(name="Commercial / 5", x=seasons,
                             y=[comm_s.get(s, 0) / 5 for s in seasons],
                             marker_color=COLORS['blue'], opacity=0.85))
        apply_layout(fig, height=280, barmode='group', yaxis_title="kWh / month")
        st.plotly_chart(fig, use_container_width=True)

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        section("Commercial — by business type")
        biz = comm.groupby('Business Type')['Units Consumed (After Solar)'].mean().sort_values()
        biz_cols = [COLORS['teal'], COLORS['blue'], COLORS['amber'],
                    COLORS['coral'], COLORS['purple'], COLORS['pink'], COLORS['green']]
        fig = go.Figure(go.Bar(x=biz.values, y=biz.index, orientation='h',
                               marker_color=biz_cols[:len(biz)],
                               text=[f"{v:.0f}" for v in biz.values],
                               textposition='outside'))
        apply_layout(fig, height=280, xaxis_title="Avg kWh / month", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        section("Solar panel impact on consumption")
        sol_hh = hh.groupby('Solar')['Units Consumed'].mean()
        no_val_hh  = float(sol_hh.get('No',  0))
        yes_val_hh = float(sol_hh.get('Yes', 0))

        sol_comm = comm.groupby('Solar')['Units Consumed (After Solar)'].mean()
        no_val_comm  = float(sol_comm.get('No',  0))
        yes_val_comm = float(sol_comm.get('Yes', 0))

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Household", x=["No Solar", "With Solar"],
                             y=[no_val_hh, yes_val_hh],
                             marker_color=COLORS['teal'], opacity=0.9,
                             text=[f"{no_val_hh:.0f}", f"{yes_val_hh:.0f}"],
                             textposition='outside'))
        fig.add_trace(go.Bar(name="Commercial", x=["No Solar", "With Solar"],
                             y=[no_val_comm, yes_val_comm],
                             marker_color=COLORS['blue'], opacity=0.85,
                             text=[f"{no_val_comm:.0f}", f"{yes_val_comm:.0f}"],
                             textposition='outside'))
        apply_layout(fig, height=280, barmode='group', yaxis_title="Avg kWh / month")
        st.plotly_chart(fig, use_container_width=True)

    section("Quick model summary — active models only")
    if not active:
        no_models_warn()
    else:
        hh_series = hh.groupby('month_num')['Units Consumed'].mean().sort_index().values
        fc_all    = build_forecasts(hh_series, 12, "Household", hh, comm, ctrl.get('growth_rate', 4.0))
        rows = []
        for m in active:
            fc = fc_all[m]
            rows.append({
                'Model':             m,
                'Avg Forecast kWh':  round(float(np.mean(fc)), 1),
                'Peak Forecast kWh': round(float(max(fc)), 1),
                'Min Forecast kWh':  round(float(min(fc)), 1),
            })
        summary_df = pd.DataFrame(rows).set_index('Model')
        st.dataframe(summary_df, use_container_width=True)

    insight("<b>Key observations:</b> Summer (May-Jun) drives peak household demand "
            "(490+ kWh/month) from air-conditioning load. Warehouses and restaurants are the "
            "highest commercial consumers. Solar adoption reduces consumption by ~12% on average. "
            "Commercial demand is 4-5x higher than household per connection.")
