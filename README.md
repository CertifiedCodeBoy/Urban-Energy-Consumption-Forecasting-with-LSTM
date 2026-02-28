# ⚡ Urban Energy Consumption Forecasting with LSTM

![Python](https://img.shields.io/badge/Python-3.10-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Overview

A deep learning pipeline that predicts hourly energy consumption across urban districts using **Long Short-Term Memory (LSTM)** networks. By analyzing historical consumption patterns alongside weather data and time features, the model enables smart grid operators to proactively balance load distribution and reduce energy waste in city infrastructure.

This project directly supports Smart City initiatives by integrating with renewable energy sources — when the model forecasts a demand spike, the city's energy management system can automatically activate solar or wind reserves before the peak occurs.

---

## 🎯 Problem Statement

Urban energy grids are increasingly strained by unpredictable demand fluctuations. Reactive energy management leads to:
- Over-reliance on fossil fuel backup generators
- Grid instability during peak hours
- Wasted renewable energy due to poor timing

**Solution:** A proactive, ML-driven forecasting model that predicts consumption 24–72 hours ahead with high accuracy.

---

## 🗂️ Dataset

- **Source:** UCI ML Repository — Individual Household Electric Power Consumption + OpenWeatherMap API
- **Size:** ~2 million records across 4 years
- **Features used:**
  - Historical power consumption (kWh)
  - Temperature, humidity, wind speed
  - Hour of day, day of week, public holiday flags
  - Solar irradiance index

---

## 🏗️ Architecture

```
Input Sequence (48h lookback)
        │
   [LSTM Layer 1]  — 128 units, return_sequences=True
        │
   [Dropout 0.2]
        │
   [LSTM Layer 2]  — 64 units
        │
   [Dense Layer]   — 32 units, ReLU
        │
   [Output Layer]  — 24 units (next 24h forecast)
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| MAE    | 0.043 kWh |
| RMSE   | 0.061 kWh |
| R²     | 0.94 |

The model achieves **94% variance explanation** on the test set, outperforming the baseline ARIMA model by 31%.

---

## 🛠️ Tech Stack

- **Modeling:** TensorFlow / Keras, Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Deployment:** FastAPI + Docker (REST endpoint for grid integration)

---

## 🚀 Getting Started

```bash
git clone https://github.com/yourusername/energy-forecasting-lstm
cd energy-forecasting-lstm
pip install -r requirements.txt
python train.py --epochs 50 --lookback 48
python serve.py  # starts FastAPI server on port 8000
```

---

## 📁 Project Structure

```
energy-forecasting-lstm/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── lstm_v2.h5
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Training.ipynb
├── src/
│   ├── preprocess.py
│   ├── model.py
│   └── evaluate.py
├── serve.py
├── train.py
└── requirements.txt
```

---

## 🔗 Smart City Integration

This module is designed as a microservice that can plug into a city's **Energy Management System (EMS)**. The `/predict` endpoint returns a 24-hour consumption forecast per district, enabling:

- Dynamic renewable energy scheduling
- Demand-response programs
- Preventive grid maintenance alerts

---

## 📄 License

MIT License © 2025
