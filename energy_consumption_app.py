import streamlit as st
from utils import load_data, train_models, MONTH_ORDER

from tabs.overview import tab_overview
from tabs.forecast import tab_forecast
from tabs.models import tab_models
from tabs.features import tab_features
from tabs.predict import tab_predict
from tabs.future import tab_future
from tabs.street import tab_street
from tabs.rawdata import tab_rawdata

st.set_page_config(
    page_title="Noida Energy Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main-header {
        background: #0a3d2b;
        border: 1px solid #1a6b4a;
        padding: 1.8rem 2.2rem;
        border-radius: 14px;
        margin-bottom: 1.6rem;
    }
    .main-header h1 {
        color: #ffffff !important;
        font-size: 1.55rem;
        font-family: 'Space Mono', monospace;
        margin: 0; line-height: 1.35;
    }
    .main-header .sub  { color: rgba(255,255,255,0.75); font-size: 0.88rem; margin-top: 0.4rem; }
    .main-header .byline { color: rgba(255,255,255,0.45); font-size: 0.76rem; margin-top: 0.5rem; }

    .metric-card {
        background: #112b1f; border: 1px solid #1e5c3a;
        border-radius: 12px; padding: 1.15rem 0.9rem; text-align: center;
    }
    .metric-card .val {
        font-size: 1.8rem; font-weight: 700; color: #3DD68C;
        font-family: 'Space Mono', monospace;
    }
    .metric-card .lbl {
        font-size: 0.7rem; color: #7aab90;
        text-transform: uppercase; letter-spacing: 0.55px; margin-top: 5px;
    }
    .metric-card .sub { font-size: 0.7rem; color: #3DD68C; margin-top: 3px; }
    .mc-red   .val, .mc-red   .sub { color: #F07070 !important; }
    .mc-amber .val, .mc-amber .sub { color: #F5BE5A !important; }

    .section-title {
        font-family: 'Space Mono', monospace; font-size: 0.88rem; font-weight: 700;
        color: #3DD68C; border-left: 3px solid #1DB87A;
        padding-left: 11px; margin: 1.3rem 0 0.8rem;
    }
    .insight-box {
        background: #091e15; border-left: 3px solid #1DB87A;
        border-radius: 0 10px 10px 0; padding: 0.9rem 1.15rem;
        font-size: 0.86rem; color: #90c8a8; line-height: 1.8; margin: 0.9rem 0;
    }
    .insight-box b { color: #3DD68C; }

    .warn-box {
        background: #2b1a00; border-left: 3px solid #F5A623;
        border-radius: 0 10px 10px 0; padding: 0.8rem 1rem;
        font-size: 0.83rem; color: #c8a060; margin: 0.5rem 0 0.9rem;
    }

    div[data-testid="stSidebarContent"] { background: #091a11 !important; }
    .footer {
        text-align: center; color: #3a6050; font-size: 0.75rem;
        margin-top: 2.5rem; padding-top: 0.9rem; border-top: 1px solid #1a3a28;
    }
</style>
""", unsafe_allow_html=True)

def sidebar():
    with st.sidebar:
        st.markdown("## Control Panel")
        st.markdown("---")

        st.markdown("### Dataset")
        dataset = st.radio("Select dataset", ["Household", "Commercial"],
                           index=0, horizontal=True)

        st.markdown("### Models to show")
        show_gb     = st.checkbox("XGBoost / GBR",     value=True, key="cb_gb")
        show_arima  = st.checkbox("ARIMA",             value=True, key="cb_arima")
        show_sarima = st.checkbox("SARIMA",            value=True, key="cb_sarima")

        st.markdown("### Forecast horizon")
        horizon = st.slider("Months ahead", 1, 12, 8, step=1)

        st.markdown("### Future forecast years")
        future_years = st.slider("Years to project", 1, 5, 3)

        st.markdown("### Expected Annual Growth")
        growth_rate = st.slider("Annual growth (%)", 0.0, 15.0, 4.0, 0.5)

        st.markdown(f"### Custom predictor ({dataset})")
        if dataset == 'Household':
            rooms      = st.slider("No. of rooms",        1,  6, 3)
            h_type     = st.selectbox("House type",       ["Apartment", "Independent"])
            b_type     = "Office"
            conn_load  = 5
        else:
            rooms      = 3
            h_type     = "Apartment"
            b_type     = st.selectbox("Business type",    ['Office', 'Warehouse', 'Salon', 'Restaurant', 'Coaching Center', 'Retail Shop', 'Clinic'])
            conn_load  = st.slider("Connected Load (kW)", 1, 50, 10)
            
        solar      = st.selectbox("Solar panel?",     ["No", "Yes"])
        sol_cap    = st.slider("Solar capacity (kW)", 0, 20 if dataset == 'Commercial' else 10, 0)
        pred_month = st.selectbox("Predict for month", MONTH_ORDER, index=4)

     
    return dict(dataset=dataset, show_gb=show_gb,
                show_arima=show_arima, show_sarima=show_sarima,
                horizon=horizon, future_years=future_years, growth_rate=growth_rate,
                rooms=rooms, solar=solar, sol_cap=sol_cap,
                h_type=h_type, b_type=b_type, conn_load=conn_load, pred_month=pred_month)

def main():
    st.markdown("""
    <div class="main-header">
        <h1>Machine Learning-Based Prediction of Household Energy Consumption</h1>
        <p class="sub">Greater Noida &amp; Noida Case Study &nbsp;|&nbsp;
            XGBoost &middot; ARIMA &middot; SARIMA</p>
        
    </div>
    """, unsafe_allow_html=True)

    try:
        hh, comm, le_h, le_c = load_data()
    except FileNotFoundError as e:
        st.error(
            f"**Data files not found.** ({e})\n\n"
            "Place `noida_electricity_household_updated.xlsx` and `noida_commercial_updated.xlsx` "
            "in the same directory as this script and restart."
        )
        st.stop()

    models = train_models(hh, comm)
    ctrl   = sidebar()

    tabs = st.tabs([
        "Overview",
        "Forecast",
        "Model Comparison",
        "Feature Analysis",
        "Custom Prediction",
        "Future Forecasting",
        "Street Lights",
        "Raw Data",
    ])

    with tabs[0]: tab_overview(hh, comm, ctrl)
    with tabs[1]: tab_forecast(hh, comm, ctrl)
    with tabs[2]: tab_models(hh, comm, models, ctrl)
    with tabs[3]: tab_features(hh, comm, models, ctrl)
    with tabs[4]: tab_predict(hh, comm, models, ctrl)
    with tabs[5]: tab_future(hh, comm, ctrl)
    with tabs[6]: tab_street()
    with tabs[7]: tab_rawdata(hh, comm, ctrl)
    st.markdown(
        '<div class="footer">'
        'Machine Learning-Based Prediction of Household Energy Consumption &nbsp;|&nbsp; '
        'Greater Noida &amp; Noida Case Study'
        '</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()