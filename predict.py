import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from utils import (
    section, metric_card, COLORS, MONTH_MAP, MONTH_SHORT, apply_layout,
    active_models, insight, warn, no_models_warn
)

def tab_predict(hh, comm, models, ctrl):
    active = active_models(ctrl)
    dataset = ctrl['dataset']
    section(f"Single-property prediction ({dataset})")

    solar      = ctrl['solar']
    sol_cap    = ctrl['sol_cap']
    pred_month = ctrl['pred_month']
    
    gb_m = models['hh' if dataset == 'Household' else 'comm']['gb']

    if dataset == 'Household':
        rooms   = ctrl['rooms']
        h_type  = ctrl['h_type']
        
        le = LabelEncoder().fit(hh['House Type'])
        known_types = list(le.classes_)
        safe_type = h_type if h_type in known_types else known_types[0]
        
        inp = [[MONTH_MAP[pred_month], 1 if solar == 'Yes' else 0,
                int(le.transform([safe_type])[0]), rooms, sol_cap]]
        
        seg = hh[(hh['Month'] == pred_month) & (hh['Rooms'] == rooms)]['Units Consumed']
        avg_act = float(seg.mean()) if len(seg) > 0 else None
        
        st.markdown(f"**Profile:** {h_type} · {rooms} rooms · "
                    f"Solar: {solar} ({sol_cap} kW) · Month: {pred_month}")
                    
        def make_row(m_num): return [[m_num, 1 if solar == 'Yes' else 0, int(le.transform([safe_type])[0]), rooms, sol_cap]]
    else:
        b_type    = ctrl['b_type']
        conn_load = ctrl['conn_load']
        
        le = LabelEncoder().fit(comm['Business Type'])
        known_types = list(le.classes_)
        safe_type = b_type if b_type in known_types else known_types[0]
        
        inp = [[MONTH_MAP[pred_month], int(le.transform([safe_type])[0]),
                1 if solar == 'Yes' else 0, conn_load, sol_cap]]
                
        seg = comm[(comm['Month'] == pred_month) & (comm['Business Type'] == b_type)]['Units Consumed (After Solar)']
        avg_act = float(seg.mean()) if len(seg) > 0 else None
        
        st.markdown(f"**Profile:** {b_type} · {conn_load} kW load · "
                    f"Solar: {solar} ({sol_cap} kW) · Month: {pred_month}")
                    
        def make_row(m_num): return [[m_num, int(le.transform([safe_type])[0]), 1 if solar == 'Yes' else 0, conn_load, sol_cap]]

    gb_val = max(0.0, float(gb_m.predict(inp)[0]))
    if avg_act is None:
        avg_act = gb_val

    st.markdown("<br>", unsafe_allow_html=True)

    sup_active = [m for m in active if m in ('XGBoost',)]
    n_cols     = max(len(sup_active) + 1, 2)
    cols_pred  = st.columns(n_cols)
    col_idx    = 0
    if 'XGBoost' in sup_active:
        metric_card(cols_pred[col_idx], f"{gb_val:,.0f}", "XGBoost", "kWh predicted")
        col_idx += 1
    metric_card(cols_pred[-1], f"{avg_act:,.0f}", "Dataset avg (same segment)", "kWh actual")

    if not sup_active:
        warn("Enable XGBoost in the sidebar to see custom predictions.")

    st.markdown("<br>", unsafe_allow_html=True)
    section(f"Predicted annual profile for this property — active supervised models")

    monthly_preds = []
    for m_num in range(1, 13):
        i2 = make_row(m_num)
        monthly_preds.append({
            'Month':             MONTH_SHORT[m_num - 1],
            'XGBoost':           round(max(0.0, float(gb_m.predict(i2)[0])), 1),
        })
    mp_df = pd.DataFrame(monthly_preds)

    fig = go.Figure()
    if 'XGBoost' in sup_active:
        fig.add_trace(go.Scatter(x=mp_df['Month'], y=mp_df['XGBoost'], name='XGBoost',
                                  line=dict(color=COLORS['teal'], width=2.5),
                                  mode='lines+markers', marker=dict(size=5)))
    apply_layout(fig, height=300, yaxis_title="Predicted kWh")
    st.plotly_chart(fig, use_container_width=True)

    section("Monthly prediction table — active models")
    if not sup_active:
        no_models_warn()
    else:
        disp_cols = ['Month'] + [c for c in ['XGBoost'] if c in sup_active]
        st.dataframe(mp_df[disp_cols].set_index('Month'), use_container_width=True)

    annual_kwh = mp_df['XGBoost'].sum() if 'XGBoost' in mp_df.columns else gb_val * 12
    rate = 6.5 if dataset == 'Household' else 8.5
    bill_est   = annual_kwh * rate
    solar_note = (f"Solar savings: ~Rs {round(sol_cap * 1400 * rate):,}/year"
                  if solar == 'Yes' and sol_cap > 0
                  else "Consider adding solar to reduce costs.")
    insight(f"<b>Annual estimate (XGBoost):</b> ~{annual_kwh:,.0f} kWh/year · "
            f"Estimated annual bill: Rs {bill_est:,.0f} (@ Rs {rate:.2f}/kWh average rate). "
            f"{solar_note}")
