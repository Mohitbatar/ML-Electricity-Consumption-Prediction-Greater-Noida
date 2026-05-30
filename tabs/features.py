import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils import (
    section, COLORS, apply_layout, active_models, warn
)

def tab_features(hh, comm, models, ctrl):
    active = active_models(ctrl)
    hh_m   = models['hh']
    col_l, col_r = st.columns(2)

    with col_l:
        section("XGBoost feature importance (Household)")
        if 'XGBoost' not in active:
            warn("Enable XGBoost to see feature importance.")
        else:
            fi_vals  = hh_m['fi']
            fi_names = ['Month', 'Solar Flag', 'House Type', 'Rooms', 'Solar Capacity']
            fig = go.Figure(go.Bar(
                x=fi_vals * 100, y=fi_names, orientation='h',
                marker_color=[COLORS['teal'] if v == max(fi_vals) else COLORS['blue']
                              for v in fi_vals],
                text=[f"{v*100:.1f}%" for v in fi_vals], textposition='outside'))
            apply_layout(fig, height=280, xaxis_title="Importance (%)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        section("Consumption by number of rooms")
        rooms_avg = hh.groupby('Rooms')['Units Consumed'].mean().sort_index()
        shades    = [f"rgba(29,184,122,{0.3 + i*0.12})" for i in range(len(rooms_avg))]
        fig = go.Figure(go.Bar(
            x=[f"{r} room{'s' if r > 1 else ''}" for r in rooms_avg.index],
            y=rooms_avg.values,
            marker_color=shades,
            text=[f"{v:.0f}" for v in rooms_avg.values], textposition='outside'))
        apply_layout(fig, height=280, yaxis_title="Avg kWh / month", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        section("Seasonal patterns — radar chart")
        hh_s   = hh.groupby('Season')['Units Consumed'].mean()
        comm_s = comm.groupby('Season')['Units Consumed (After Solar)'].mean()
        seasons = sorted(list(set(hh_s.index) | set(comm_s.index)))
        hh_vals_r   = [float(hh_s.get(s, 0)) for s in seasons]
        comm_vals_r = [float(comm_s.get(s, 0)) / 5 for s in seasons]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=hh_vals_r + [hh_vals_r[0]],
            theta=seasons + [seasons[0]], fill='toself',
            name='Household', line_color=COLORS['teal'],
            fillcolor='rgba(29,184,122,0.18)'))
        fig.add_trace(go.Scatterpolar(
            r=comm_vals_r + [comm_vals_r[0]],
            theta=seasons + [seasons[0]], fill='toself',
            name='Commercial / 5', line_color=COLORS['blue'],
            fillcolor='rgba(77,159,224,0.18)'))
        fig.update_layout(
            polar=dict(
                bgcolor="#0c1e14",
                radialaxis=dict(visible=True, gridcolor="#1a3828",
                                linecolor="#264a36", tickfont=dict(color="#7aab90")),
                angularaxis=dict(gridcolor="#1a3828", linecolor="#264a36",
                                 tickfont=dict(color="#7aab90"))),
            paper_bgcolor="#0c1e14", font_family="DM Sans", font_color="#c8ddd0",
            height=320,
            legend=dict(orientation="h", y=-0.18, bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=30, b=65, l=40, r=40))
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        section("House type — consumption distribution")
        apt  = hh[hh['House Type'] == 'Apartment']['Units Consumed']
        indp = hh[hh['House Type'] == 'Independent']['Units Consumed']
        fig  = go.Figure()
        fig.add_trace(go.Box(y=apt.values,  name='Apartment',
                             marker_color=COLORS['teal'], boxmean=True))
        fig.add_trace(go.Box(y=indp.values, name='Independent',
                             marker_color=COLORS['blue'], boxmean=True))
        apply_layout(fig, height=320, yaxis_title="kWh / month")
        st.plotly_chart(fig, use_container_width=True)

    section("Correlation matrix — Household features")
    hh2 = hh.copy()
    hh2['house_type_num'] = (hh2['House Type'] == 'Independent').astype(int)
    corr_cols = ['month_num','Rooms','Solar Capacity (kW)','solar_flag',
                 'house_type_num','Units Consumed']
    corr  = hh2[corr_cols].corr()
    fig_h = px.imshow(corr, text_auto='.2f',
                      color_continuous_scale=[[0,'#E8614A'],[0.5,'#112b1f'],[1,'#1DB87A']],
                      zmin=-1, zmax=1)
    fig_h.update_layout(font_family="DM Sans", font_color="#c8ddd0",
                         paper_bgcolor="#0c1e14", plot_bgcolor="#0c1e14",
                         height=360, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_h, use_container_width=True)

    section("Feature summary table")
    feat_df = pd.DataFrame({
        'Feature':     ['Month / Season', 'Rooms', 'Solar Capacity (kW)',
                        'Solar Flag', 'House Type'],
        'Importance %': [f"{v*100:.1f}%" for v in hh_m['fi']],
        'Data Type':   ['Temporal','Numeric','Numeric','Binary','Categorical'],
        'Key insight': ['Strongest seasonal driver', 'Linear growth with rooms',
                        'Reduces actual bill', 'Moderate reduction effect',
                        'Minimal difference between types'],
    }).set_index('Feature')
    st.dataframe(feat_df, use_container_width=True)
    if 'XGBoost' not in active:
        warn("Feature importance percentages are from XGBoost. "
             "Enable XGBoost for the most accurate view.")
