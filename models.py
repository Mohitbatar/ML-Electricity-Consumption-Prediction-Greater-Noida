import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import (
    section, metric_card, COLORS, apply_layout, MODEL_META,
    active_models, calc_metrics, arima_forecast, sarima_forecast,
    insight, warn, no_models_warn
)

def tab_models(hh, comm, models, ctrl):
    active = active_models(ctrl)
    section("Performance metrics — active models only")

    hh_m  = models['hh']
    com_m = models['comm']

    gbh = hh_m['gb_metrics']
    gbc = com_m['gb_metrics']

    supervised_map = {'XGBoost': (gbh, gbc)}
    active_sup     = [m for m in active if m in supervised_map]

    if active_sup:
        cols_top = st.columns(len(active_sup) * 2)
        idx = 0
        for m in active_sup:
            mh, mc = supervised_map[m]
            cls = "mc-red" if mh['R2'] < 0 else ("mc-amber" if mh['R2'] < 0.3 else "")
            metric_card(cols_top[idx],     f"{mh['R2']:.4f}", f"{m} R² (HH)",   "household", cls)
            metric_card(cols_top[idx + 1], f"{mc['R2']:.4f}", f"{m} R² (Comm)", "commercial")
            idx += 2
        st.markdown("<br>", unsafe_allow_html=True)

    if not active:
        no_models_warn()
        return

    hh_series   = hh.groupby(['Year', 'month_num'])['Units Consumed'].mean().sort_index().values.astype(float)
    comm_series = comm.groupby(['Year', 'month_num'])['Units Consumed (After Solar)'].mean().sort_index().values.astype(float)

    def ts_in_sample_metrics(series, model_name):
        n = len(series)
        if n < 13:
            return {"RMSE": "N/A", "MAE": "N/A", "MAPE": "N/A", "R2": "N/A", "MSE": "N/A"}
        # Hold-out last 12 points for testing; train on the rest
        split  = max(12, n - 12)
        train  = series[:split]
        actual = series[split:]
        steps  = len(actual)
        if model_name == 'ARIMA':
            pred = arima_forecast(train, steps=steps)
        elif model_name == 'SARIMA':
            pred = sarima_forecast(train, steps=steps)
        else:
            return {"RMSE": "N/A", "MAE": "N/A", "MAPE": "N/A", "R2": "N/A", "MSE": "N/A", "actual": [], "pred": []}
        
        res = calc_metrics(actual, pred)
        res["actual"] = list(actual)
        res["pred"] = list(pred)
        return res

    ts_hh_metrics   = {m: ts_in_sample_metrics(hh_series,   m) for m in ['ARIMA','SARIMA']}
    ts_comm_metrics = {m: ts_in_sample_metrics(comm_series, m) for m in ['ARIMA','SARIMA']}

    hh_all_rows = [

        {"Model": "XGBoost",
         "RMSE": gbh['RMSE'], "MAE": gbh['MAE'], "MAPE (%)": gbh['MAPE'],
         "R²": gbh['R2'], "MSE": gbh['MSE'],
         "Type": "Supervised", "Notes": "Best — captures seasonal nonlinearity"},

        {"Model": "ARIMA",
         "RMSE": ts_hh_metrics['ARIMA']['RMSE'], "MAE": ts_hh_metrics['ARIMA']['MAE'], "MAPE (%)": ts_hh_metrics['ARIMA']['MAPE'],
         "R²": "—", "MSE": "—",
         "Type": "Time Series", "Notes": "Mean-reverting; good short-term trend"},

        {"Model": "SARIMA",
         "RMSE": ts_hh_metrics['SARIMA']['RMSE'], "MAE": ts_hh_metrics['SARIMA']['MAE'], "MAPE (%)": ts_hh_metrics['SARIMA']['MAPE'],
         "R²": "—", "MSE": "—",
         "Type": "Time Series", "Notes": "Best seasonal pattern reproduction"},

    ]
    hh_df = pd.DataFrame([r for r in hh_all_rows if r['Model'] in active])
    st.markdown("**Household dataset — active models**")
    if hh_df.empty:
        no_models_warn()
    else:
        st.dataframe(hh_df.set_index("Model"), use_container_width=True)

    comm_all_rows = [

        {"Model": "XGBoost",
         "RMSE": gbc['RMSE'], "MAE": gbc['MAE'], "MAPE (%)": gbc['MAPE'],
         "R²": gbc['R2'], "MSE": gbc['MSE'],
         "Type": "Supervised", "Notes": "Best — business type and load are key drivers"},

        {"Model": "ARIMA",
         "RMSE": ts_comm_metrics['ARIMA']['RMSE'], "MAE": ts_comm_metrics['ARIMA']['MAE'], "MAPE (%)": ts_comm_metrics['ARIMA']['MAPE'],
         "R²": "—", "MSE": "—",
         "Type": "Time Series", "Notes": "Monthly aggregated trend"},

        {"Model": "SARIMA",
         "RMSE": ts_comm_metrics['SARIMA']['RMSE'], "MAE": ts_comm_metrics['SARIMA']['MAE'], "MAPE (%)": ts_comm_metrics['SARIMA']['MAPE'],
         "R²": "—", "MSE": "—",
         "Type": "Time Series", "Notes": "Seasonality weaker in commercial"},

    ]
    comm_df = pd.DataFrame([r for r in comm_all_rows if r['Model'] in active])
    st.markdown("**Commercial dataset — active models**")
    if comm_df.empty:
        no_models_warn()
    else:
        st.dataframe(comm_df.set_index("Model"), use_container_width=True)

    section("Performance comparison — all models")
    if not active:
        warn("Enable at least one model to see RMSE and MAPE bar charts.")
    else:
        rmse_hh_vals, mape_hh_vals = [], []
        rmse_comm_vals, mape_comm_vals = [], []
        for m in active:
            if m == 'XGBoost':
                rmse_hh_vals.append(gbh['RMSE'])
                mape_hh_vals.append(gbh['MAPE'])
                rmse_comm_vals.append(gbc['RMSE'])
                mape_comm_vals.append(gbc['MAPE'])
            else:
                rmse_hh_vals.append(ts_hh_metrics[m]['RMSE'])
                mape_hh_vals.append(ts_hh_metrics[m]['MAPE'])
                rmse_comm_vals.append(ts_comm_metrics[m]['RMSE'])
                mape_comm_vals.append(ts_comm_metrics[m]['MAPE'])

        def _to_num(v):
            """Return numeric value or None so Plotly renders a gap instead of crashing."""
            return v if v != "N/A" else None

        st.markdown("#### RMSE (Root Mean Squared Error) — lower is better")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            fig_rmse_h = go.Figure(go.Bar(
                x=active, y=[_to_num(v) for v in rmse_hh_vals],
                marker_color=[MODEL_META[m]['color'] for m in active],
                text=[f"{v:.2f}" if v != "N/A" else "N/A" for v in rmse_hh_vals], textposition='outside'))
            apply_layout(fig_rmse_h, height=280, yaxis_title="RMSE (kWh)",
                         showlegend=False, title="Household")
            st.plotly_chart(fig_rmse_h, use_container_width=True)
        with col_r2:
            fig_rmse_c = go.Figure(go.Bar(
                x=active, y=[_to_num(v) for v in rmse_comm_vals],
                marker_color=[MODEL_META[m]['color'] for m in active],
                text=[f"{v:.2f}" if v != "N/A" else "N/A" for v in rmse_comm_vals], textposition='outside'))
            apply_layout(fig_rmse_c, height=280, yaxis_title="RMSE (kWh)",
                         showlegend=False, title="Commercial")
            st.plotly_chart(fig_rmse_c, use_container_width=True)

        st.markdown("#### MAPE (Mean Absolute Percentage Error) — lower is better")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig_mape_h = go.Figure(go.Bar(
                x=active, y=[_to_num(v) for v in mape_hh_vals],
                marker_color=[MODEL_META[m]['color'] for m in active],
                text=[f"{v:.2f}%" if v != "N/A" else "N/A" for v in mape_hh_vals], textposition='outside'))
            apply_layout(fig_mape_h, height=260, yaxis_title="MAPE (%)",
                         showlegend=False, title="Household")
            st.plotly_chart(fig_mape_h, use_container_width=True)
        with col_m2:
            fig_mape_c = go.Figure(go.Bar(
                x=active, y=[_to_num(v) for v in mape_comm_vals],
                marker_color=[MODEL_META[m]['color'] for m in active],
                text=[f"{v:.2f}%" if v != "N/A" else "N/A" for v in mape_comm_vals], textposition='outside'))
            apply_layout(fig_mape_c, height=260, yaxis_title="MAPE (%)",
                         showlegend=False, title="Commercial")
            st.plotly_chart(fig_mape_c, use_container_width=True)

    section("Predicted vs Actual scatter — XGBoost")
    if 'XGBoost' not in active:
        warn("Enable XGBoost to see scatter plots.")
    else:
        cols = st.columns(2)
        
        # Household
        act_hh  = hh_m['y_te']
        pred_hh = hh_m['gb_pred']
        rng_hh  = [float(min(act_hh)), float(max(act_hh))]
        fig_hh = go.Figure()
        fig_hh.add_trace(go.Scatter(x=act_hh, y=pred_hh, mode='markers',
                                  marker=dict(color=MODEL_META['XGBoost']['color'], size=7, opacity=0.75),
                                  name='XGBoost (HH)'))
        fig_hh.add_trace(go.Scatter(x=rng_hh, y=rng_hh, mode='lines',
                                  line=dict(color=COLORS['gray'], dash='dash', width=1.5),
                                  name='Perfect fit'))
        apply_layout(fig_hh, height=320, xaxis_title="Actual kWh", yaxis_title="Predicted kWh", title="Household")
        cols[0].plotly_chart(fig_hh, use_container_width=True)

        # Commercial
        act_cc  = com_m['y_te']
        pred_cc = com_m['gb_pred']
        rng_cc  = [float(min(act_cc)), float(max(act_cc))]
        fig_cc = go.Figure()
        fig_cc.add_trace(go.Scatter(x=act_cc, y=pred_cc, mode='markers',
                                  marker=dict(color=MODEL_META['XGBoost']['color'], size=7, opacity=0.75),
                                  name='XGBoost (Comm)'))
        fig_cc.add_trace(go.Scatter(x=rng_cc, y=rng_cc, mode='lines',
                                  line=dict(color=COLORS['gray'], dash='dash', width=1.5),
                                  name='Perfect fit'))
        apply_layout(fig_cc, height=320, xaxis_title="Actual kWh", yaxis_title="Predicted kWh", title="Commercial")
        cols[1].plotly_chart(fig_cc, use_container_width=True)

    section("Residual distribution — XGBoost (Household)")
    if 'XGBoost' not in active:
        warn("Enable XGBoost to see the residual distribution chart.")
    else:
        resid = hh_m['y_te'] - hh_m['gb_pred']
        fig_r = go.Figure()
        fig_r.add_trace(go.Histogram(x=resid, nbinsx=20,
                                      marker_color=COLORS['teal'], opacity=0.85,
                                      name='Residuals'))
        fig_r.add_vline(x=0, line_dash="dash", line_color=COLORS['coral'])
        apply_layout(fig_r, height=250, xaxis_title="Residual (kWh)",
                     yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig_r, use_container_width=True)

    insight(
        "<b>Metric guide:</b><br>"
        "<b>RMSE</b> (Root Mean Squared Error) — penalises large errors more heavily; lower is better.<br>"
        "<b>MAE</b> (Mean Absolute Error) — average absolute deviation; more robust to outliers.<br>"
        "<b>MAPE</b> (Mean Absolute Percentage Error) — scale-independent; lower % = better fit. "
        "For time-series models MAPE is estimated via a hold-out last-12-months split.<br>"
        "<b>R²</b> (Coefficient of Determination) — proportion of variance explained; 1.0 = perfect. "
        "Only available for supervised (test-set) models."
    )
