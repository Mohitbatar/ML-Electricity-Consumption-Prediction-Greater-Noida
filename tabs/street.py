import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import (
    section, metric_card, COLORS, apply_layout, insight
)

def tab_street():
    section("GNIDA Street Light Energy Estimation")

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "33,534",     "Total lights estimated", "")
    metric_card(c2, "36,217 kWh", "Daily consumption",      "")
    metric_card(c3, "13.2 GWh",   "Annual estimate",        "")
    metric_card(c4, "90 W",       "LED fixture wattage",    "assumed")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        section("Road network breakdown")
        fig = go.Figure(go.Pie(
            labels=['Wide roads (206 km)', 'Internal roads (594 km)'],
            values=[206, 594], hole=0.42,
            marker_colors=[COLORS['amber'], COLORS['teal']],
            textinfo='label+percent', textfont=dict(color='#c8ddd0')))
        fig.update_layout(font_family="DM Sans", font_color="#c8ddd0",
                          paper_bgcolor="#0c1e14", height=300,
                          margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        section("Daily consumption by road type")
        fig2 = go.Figure(go.Bar(
            x=['Wide roads (both sides)', 'Internal roads (one side)', 'Total'],
            y=[14832.72, 21384.0, 36216.72],
            marker_color=[COLORS['amber'], COLORS['teal'], COLORS['coral']],
            text=['14,833 kWh', '21,384 kWh', '36,217 kWh'],
            textposition='outside', width=0.52))
        apply_layout(fig2, height=300, yaxis_title="kWh / day", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    section("Interactive calculation tool")
    col_a, col_b, col_c = st.columns(3)
    with col_a: wattage  = st.slider("Fixture wattage (W)",    50, 200, 90,  step=10)
    with col_b: spacing  = st.slider("Light spacing (m)",      20,  50, 30,  step=5)
    with col_c: op_hours = st.slider("Operating hours / day",   8,  14, 12)

    wide_lights = int(2 * (206_000 / spacing))
    int_lights  = int(594_000 / spacing)
    wide_kwh    = wide_lights * (wattage / 1000) * op_hours
    int_kwh     = int_lights  * (wattage / 1000) * op_hours
    total_kwh   = wide_kwh + int_kwh
    annual_gwh  = total_kwh * 365 / 1e6

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Wide road lights",     f"{wide_lights:,}")
    r2.metric("Internal road lights", f"{int_lights:,}")
    r3.metric("Daily consumption",    f"{total_kwh:,.0f} kWh")
    r4.metric("Annual consumption",   f"{annual_gwh:.2f} GWh")

    section("Calculation detail table")
    calc_df = pd.DataFrame({
        'Parameter':      ['Road length', 'Spacing', 'Sides lit', 'Light count',
                           'Wattage', 'Hours/day', 'kWh/day'],
        'Wide Roads':     ['206 km', f'{spacing} m', 'Both (x2)',
                           f'{wide_lights:,}', f'{wattage} W', f'{op_hours} h',
                           f'{wide_kwh:,.0f}'],
        'Internal Roads': ['594 km', f'{spacing} m', 'One side',
                           f'{int_lights:,}', f'{wattage} W', f'{op_hours} h',
                           f'{int_kwh:,.0f}'],
    }).set_index('Parameter')
    st.dataframe(calc_df, use_container_width=True)

    insight("<b>Methodology (GNIDA 2020 data):</b><br>"
            "Wide roads (45-132 m width, 206 km total): lights on both sides — "
            "2 x (206,000 / spacing) fixtures<br>"
            "Internal roads (less than 45 m width, 594 km total): lights one side — "
            "594,000 / spacing fixtures<br>"
            "Validated by GNIDA installation of 3,740 lights across 64 villages "
            "and Noida's 100,000+ LED street lights across urban/rural/industrial areas.")
