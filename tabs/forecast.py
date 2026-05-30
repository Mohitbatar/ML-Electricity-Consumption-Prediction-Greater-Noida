import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import (
    section, COLORS, MONTH_SHORT, apply_layout, MODEL_META,
    active_models, build_forecasts, insight, no_models_warn
)

def tab_forecast(hh, comm, ctrl):
    ds      = ctrl['dataset']
    horizon = ctrl['horizon']
    active  = active_models(ctrl)

    if ds == "Household":
        grouped = hh.groupby(['Year', 'month_num'])['Units Consumed'].mean().sort_index()
        unit   = "kWh / month (Household)"
    else:
        grouped = comm.groupby(['Year', 'month_num'])['Units Consumed (After Solar)'].mean().sort_index()
        unit   = "kWh / month (Commercial)"

    series = grouped.values
    hist_labels = [f"{MONTH_SHORT[m-1]} {y} (hist)" for y, m in grouped.index]
    
    last_year, last_month = grouped.index[-1]
    fc_labels = []
    for i in range(1, horizon + 1):
        m = last_month + i
        y = last_year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        fc_labels.append(f"{MONTH_SHORT[m-1]} {y} (fc+{i})")
        
    section("Historical Data")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=hist_labels, y=list(series), name="Historical",
                              line=dict(color=COLORS['gray'], width=3),
                              mode='lines+markers', marker=dict(size=5)))
    fig_hist.update_xaxes(tickangle=45, tickfont_size=10,
                     tickmode='array', tickvals=hist_labels[::2])
    apply_layout(fig_hist, height=350, yaxis_title=unit,
                 title=f"{ds} Historical Energy Consumption")
    st.plotly_chart(fig_hist, use_container_width=True)

    fc_all = build_forecasts(series, horizon, ds, hh, comm, ctrl.get('growth_rate', 4.0))

    section("Forecast Projection")
    fig_fc = go.Figure()
    if not active:
        no_models_warn()
    else:
        for m in active:
            meta = MODEL_META[m]
            fig_fc.add_trace(go.Scatter(
                x=fc_labels,
                y=[float(v) for v in fc_all[m]],
                name=m,
                line=dict(color=meta['color'], width=2, dash=meta['dash']),
                mode='lines+markers', marker=dict(size=4)))
        
        fig_fc.update_xaxes(tickangle=45, tickfont_size=10,
                         tickmode='array', tickvals=fc_labels)
        apply_layout(fig_fc, height=350, yaxis_title=unit,
                     title=f"{ds} Energy Forecast — {horizon} months ahead")
        st.plotly_chart(fig_fc, use_container_width=True)

    section("Forecast values table — active models only")
    if not active:
        no_models_warn()
    else:
        fc_dict = {'Month': fc_labels}
        for m in active:
            fc_dict[m] = [round(float(v), 1) for v in fc_all[m]]
        fc_table = pd.DataFrame(fc_dict).set_index('Month')
        st.dataframe(fc_table, use_container_width=True)

        if len(active) > 1:
            section("Ensemble average of active models")
            ensemble = np.mean([[float(v) for v in fc_all[m]] for m in active], axis=0)
            ens_df   = pd.DataFrame({'Month': fc_labels,
                                     'Ensemble Average': [round(float(v), 1) for v in ensemble]
                                     }).set_index('Month')
            st.dataframe(ens_df, use_container_width=True)

    insight("<b>Model behaviour:</b> "
            "<b>SARIMA</b> best reproduces seasonal peaks (summer highs, winter lows). "
            "<b>XGBoost</b> closely tracks the historical seasonal shape. "
            "<b>ARIMA</b> mean-reverts gradually toward the unconditional mean. ")
