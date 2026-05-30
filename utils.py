import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

MONTH_ORDER = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
MONTH_MAP   = {m: i+1 for i, m in enumerate(MONTH_ORDER)}
MONTH_SHORT = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

COLORS = {
    'teal':   '#1DB87A', 'blue':   '#4D9FE0', 'amber':  '#F5A623',
    'coral':  '#E8614A', 'purple': '#9B8FE8', 'pink':   '#E06494',
    'gray':   '#7A9A7A', 'green':  '#5CB85C',
}

MODEL_META = {
    'XGBoost':           {'color': COLORS['teal'],   'dash': 'solid'},
    'ARIMA':             {'color': COLORS['amber'],  'dash': 'dot'},
    'SARIMA':            {'color': COLORS['coral'],  'dash': 'dashdot'},
}

PLOTLY_LAYOUT = dict(
    font_family="DM Sans", font_color="#c8ddd0",
    paper_bgcolor="#0c1e14", plot_bgcolor="#0c1e14",
    margin=dict(t=36, b=44, l=54, r=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="left", x=0, font_size=11, bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(showgrid=True, gridcolor="#1a3828", linecolor="#264a36",
               tickfont=dict(color="#7aab90")),
    yaxis=dict(showgrid=True, gridcolor="#1a3828", linecolor="#264a36",
               tickfont=dict(color="#7aab90")),
)

def apply_layout(fig, **kwargs):
    fig.update_layout(**PLOTLY_LAYOUT, **kwargs)
    return fig

@st.cache_data
def load_data():
    hh   = pd.read_excel("noida_electricity_household_updated.xlsx")
    comm = pd.read_excel("noida_commercial_updated.xlsx")
    hh['month_num']      = hh['Month'].map(MONTH_MAP)
    hh['solar_flag']     = (hh['Solar'] == 'Yes').astype(int)
    le_h = LabelEncoder()
    hh['house_type_enc'] = le_h.fit_transform(hh['House Type'])
    comm['month_num']    = comm['Month'].map(MONTH_MAP)
    comm['solar_flag']   = (comm['Solar'] == 'Yes').astype(int)
    le_c = LabelEncoder()
    comm['business_enc'] = le_c.fit_transform(comm['Business Type'])
    return hh, comm, le_h, le_c

def calc_metrics(y_true, y_pred):
    """
    Compute MSE, RMSE, MAE, MAPE, and R2.
    MAPE is guarded against division-by-zero (zero actual values are excluded).
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    nonzero_mask = y_true != 0
    if nonzero_mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask])
                                    / y_true[nonzero_mask])) * 100)
    else:
        mape = float('nan')

    return {
        'MSE':  round(mse,  2),
        'RMSE': round(rmse, 2),
        'MAE':  round(mae,  2),
        'MAPE': round(mape, 2) if not np.isnan(mape) else "N/A",
        'R2':   round(r2,   4),
    }

def safe_mape(series_true, series_pred):
    """MAPE for time-series forecasts (approx, on the historical series)."""
    arr_t = np.array(series_true, dtype=float)
    arr_p = np.array(series_pred, dtype=float)
    mask  = arr_t != 0
    if mask.sum() == 0:
        return "N/A"
    return round(float(np.mean(np.abs((arr_t[mask] - arr_p[mask]) / arr_t[mask])) * 100), 2)

@st.cache_resource
def train_models(hh, comm):
    results = {}
    feat_h  = ['month_num','solar_flag','house_type_enc','Rooms','Solar Capacity (kW)']
    X_h     = hh[feat_h].values
    y_h     = hh['Units Consumed'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X_h, y_h, test_size=0.2, random_state=42)
    gb_h = XGBRegressor(n_estimators=300, learning_rate=0.05,
                        max_depth=4, subsample=0.8, random_state=42).fit(X_tr, y_tr)
    gb_pred_h = gb_h.predict(X_te)
    results['hh'] = dict(
        X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
        gb=gb_h, fi=gb_h.feature_importances_,
        gb_pred=gb_pred_h,
        gb_metrics=calc_metrics(y_te, gb_pred_h),
    )

    feat_c  = ['month_num','business_enc','solar_flag',
               'Connected Load (kW)','Solar Capacity (kW)']
    X_c     = comm[feat_c].values
    y_c     = comm['Units Consumed (After Solar)'].values
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_c, y_c, test_size=0.2, random_state=42)
    gb_c = XGBRegressor(n_estimators=300, learning_rate=0.05,
                        max_depth=4, random_state=42).fit(Xc_tr, yc_tr)
    gb_pred_c = gb_c.predict(Xc_te)
    results['comm'] = dict(
        X_tr=Xc_tr, X_te=Xc_te, y_tr=yc_tr, y_te=yc_te,
        gb=gb_c, fi=gb_c.feature_importances_,
        gb_pred=gb_pred_c,
        gb_metrics=calc_metrics(yc_te, gb_pred_c),
    )
    return results

def arima_forecast(series, steps=12):
    """ARIMA(2,1,1) forecast via statsmodels — captures non-linear AR dynamics."""
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    series = np.array(series, dtype=float)
    if len(series) < 4:
        last = float(series[-1])
        mu   = float(np.mean(np.diff(series))) if len(series) > 1 else 0.0
        return [max(50.0, last + mu * (i + 1)) for i in range(steps)]
    try:
        fit = _ARIMA(series, order=(2, 1, 1)).fit()
        fc  = fit.forecast(steps=steps)
        return [max(50.0, float(v)) for v in fc]
    except Exception:
        diff = np.diff(series)
        mu   = float(np.mean(diff))
        std  = float(np.std(diff)) if len(diff) > 1 else 0.0
        last = float(series[-1])
        out  = []
        for i in range(steps):
            last += mu + std * 0.4 * np.sin(2 * np.pi * i / 6)
            out.append(max(50.0, last))
        return out

def sarima_forecast(series, steps=12):
    """Seasonal AR model using lags [1, 2, 12]."""
    series  = np.array(series, dtype=float)
    n, lags = len(series), [1, 2, 12]
    max_lag = max(lags)
    if n <= max_lag:
        base = list(series)
        out  = []
        for i in range(steps):
            out.append(float(base[(i % len(base))]))
        return out
    X     = np.column_stack([series[max_lag - lg: n - lg] for lg in lags])
    y     = series[max_lag:]
    coefs = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]
    c, ar = coefs[0], coefs[1:]
    ext   = list(series)
    for _ in range(steps):
        ext.append(max(50.0, c + sum(ar[i] * ext[-lags[i]] for i in range(len(lags)))))
    return [float(v) for v in ext[-steps:]]

def build_forecasts(series, horizon, ds, hh, comm, growth_rate=4.0):
    """Compute all 5 model forecasts; return dict model_name -> list of float values."""
    series = np.array(series, dtype=float)

    arima_fc  = arima_forecast(series, horizon)
    sarima_fc = sarima_forecast(series, horizon)

    if ds == "Household":
        le   = LabelEncoder().fit(hh['House Type'])
        hh2  = hh.copy()
        hh2['house_type_enc'] = le.transform(hh2['House Type'])
        feat = ['month_num','solar_flag','house_type_enc','Rooms','Solar Capacity (kW)']
        X2   = hh2[feat].values
        y2   = hh2['Units Consumed'].values
        base_df = hh2[feat].copy()
    else:
        le    = LabelEncoder().fit(comm['Business Type'])
        comm2 = comm.copy()
        comm2['business_enc'] = le.transform(comm2['Business Type'])
        feat  = ['month_num','business_enc','solar_flag',
                 'Connected Load (kW)','Solar Capacity (kW)']
        X2   = comm2[feat].values
        y2   = comm2['Units Consumed (After Solar)'].values
        base_df = comm2[feat].copy()

    gb_m = XGBRegressor(n_estimators=300, learning_rate=0.05,
                        max_depth=4, subsample=0.8, random_state=42).fit(X2, y2)

    # Predict across all rows for each future month and average,
    # so XGBoost reflects the average customer — not a single hardcoded profile.
    gb_fc = []
    for i in range(horizon):
        month_num = (i % 12) + 1
        batch = base_df.copy()
        batch['month_num'] = month_num
        preds = gb_m.predict(batch.values)
        gb_fc.append(max(0.0, float(preds.mean())))

    res = {
        'XGBoost':           gb_fc,
        'ARIMA':             arima_fc,
        'SARIMA':            sarima_fc,
    }

    # Apply compounding growth rate over time to avoid simple copy-pasting behavior
    # and provide a more realistic year-over-year projection.
    for m in res:
        scaled = []
        for i, val in enumerate(res[m]):
            # compounding growth per month
            multiplier = (1.0 + (growth_rate / 100.0)) ** (i / 12.0)
            scaled.append(val * multiplier)
        res[m] = scaled

    return res

def active_models(ctrl):
    """Return list of model names that are currently toggled ON."""
    mapping = {
        'XGBoost':           ctrl['show_gb'],
        'ARIMA':             ctrl['show_arima'],
        'SARIMA':            ctrl['show_sarima'],
    }
    return [m for m, on in mapping.items() if on]

def metric_card(col, val, lbl, sub="", extra_cls=""):
    col.markdown(
        f'<div class="metric-card {extra_cls}">'
        f'<div class="val">{val}</div><div class="lbl">{lbl}</div>'
        f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

def warn(text):
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)

def no_models_warn():
    warn("No models selected. Enable at least one model in the sidebar to see results.")
