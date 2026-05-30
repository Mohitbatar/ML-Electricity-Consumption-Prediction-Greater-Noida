# Machine Learning-Based Prediction of Household and Commercial Energy Consumption
### Greater Noida & Noida Case Study

An interactive Streamlit-based data analytics and forecasting platform designed to predict and analyze electricity consumption in the Noida and Greater Noida regions. The application implements supervised machine learning (XGBoost) and classical statistical time-series forecasting models (ARIMA and SARIMA) to help grid operators and consumers optimize energy loads and solar utilization.

**Live Application:** [Launch Dashboard](https://ml-electricity-consumption-prediction-greater-noida.streamlit.app/)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Architecture & Directory Structure](#architecture--directory-structure)
4. [Dataset Specifications](#dataset-specifications)
5. [Models & Forecasting Methodologies](#models--forecasting-methodologies)
6. [Installation & Setup](#installation--setup)
7. [Running the Application](#running-the-application)

---

## Project Overview
Rapid urbanization in cities like Noida and Greater Noida has caused a surge in electrical load demand, particularly seasonal spikes during summer due to air conditioning. This project provides:
* A data-driven approach to model electricity billing records across Household and Commercial connections.
* A comparative evaluation of machine learning models versus time-series forecasting.
* Custom simulation tools for solar panel adoption, commercial load growth, and municipal street light electrification.

---

## Key Features

The application is structured into 8 modular tabs:
1. **Overview Dashboard:** Renders aggregate metrics, seasonal consumption graphs, commercial distribution by business type, and a breakdown of solar impact.
2. **Forecast Projection:** Allows users to compare historical data side-by-side with 1–12 month forward projections using active models.
3. **Model Comparison:** Displays cross-validated error metrics (MSE, RMSE, MAE, MAPE, $R^2$) and residuals analysis.
4. **Feature Analysis:** Visualizes feature importances from XGBoost, correlation matrices, box plots of housing types, and seasonal radar charts.
5. **Custom Prediction:** Interactive simulation tool that estimates monthly and annual bills (using Rs. 6.50/kWh for residential and Rs. 8.50/kWh for commercial) based on rooms, connected load, business type, and solar panel capacity.
6. **Future Forecasting:** Long-range projections (up to 5 years) compounding with expected annual growth rates, displaying yearly totals and peak/trough analysis.
7. **Street Lights (GNIDA):** A municipal planning module using Greater Noida Industrial Development Authority (GNIDA) road networks (206 km wide roads, 594 km internal roads) to compute LED fixture installation counts, daily/annual kWh consumption, and interactive cost details.
8. **Raw Data Explorer:** Allows users to inspect and download descriptive statistics for the underlying datasets.

---

## Architecture & Directory Structure
```text
├── energy_consumption_app.py      # Streamlit Main App entry point
├── utils.py                       # Helper functions, data-loading, model training & styling
├── requirements.txt               # App dependencies
├── noida_electricity_household_updated.xlsx  # Cleaned residential dataset
├── noida_commercial_updated.xlsx             # Cleaned commercial dataset
└── tabs/                          # Component layouts for application tabs
    ├── __init__.py
    ├── overview.py                # Dashboard tab UI
    ├── forecast.py                # Multi-model plotting tab UI
    ├── models.py                  # Evaluation metrics & diagnostic charts
    ├── features.py                # Feature analysis, heatmaps, & radar charts
    ├── predict.py                 # Single-property billing simulator
    ├── future.py                  # Long-range compounding growth model
    ├── street.py                  # GNIDA Street light estimation layout
    └── rawdata.py                 # Raw dataset summary & preview
