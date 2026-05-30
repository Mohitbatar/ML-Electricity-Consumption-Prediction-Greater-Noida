import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from utils import (
    section, active_models, warn
)

def tab_rawdata(hh, comm, ctrl):
    active = active_models(ctrl)
    section("Raw datasets")
    tab_a, tab_b = st.tabs(["Household", "Commercial"])

    with tab_a:
        st.dataframe(hh, use_container_width=True, height=400)
        c1, c2 = st.columns(2)
        with c1: st.write(hh.describe().round(2))
        with c2: st.write(hh.dtypes.rename("dtype").to_frame())

        section("Sample predictions on raw data — active supervised models")
        sup_active = [m for m in active if m in ('XGBoost',)]
        if not sup_active:
            warn("Enable XGBoost to see predictions on raw data.")
        else:
            le   = LabelEncoder().fit(hh['House Type'])
            hh2  = hh.copy()
            hh2['house_type_enc'] = le.transform(hh2['House Type'])
            feat = ['month_num','solar_flag','house_type_enc','Rooms','Solar Capacity (kW)']
            gb_m = XGBRegressor(n_estimators=300, learning_rate=0.05,
                                max_depth=4, subsample=0.8, random_state=42
                                ).fit(hh2[feat].values, hh2['Units Consumed'].values)
            sample = hh2.head(30).copy()
            if 'XGBoost' in sup_active:
                sample['XGBoost Predicted'] = gb_m.predict(sample[feat].values).round(1)
            show_cols = ['Month','House Type','Rooms','Solar','Units Consumed']
            if 'XGBoost'           in sup_active: show_cols.append('XGBoost Predicted')
            st.dataframe(sample[show_cols], use_container_width=True)

    with tab_b:
        st.dataframe(comm, use_container_width=True, height=400)
        c1, c2 = st.columns(2)
        with c1: st.write(comm.describe().round(2))
        with c2: st.write(comm.dtypes.rename("dtype").to_frame())
