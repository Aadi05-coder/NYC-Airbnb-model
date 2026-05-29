# 🗽 NYC Airbnb Price Intelligence

A professional data-science web app built with **Streamlit** that analyses the 2019 New York City Airbnb dataset and predicts nightly prices using **Linear Regression**, **Random Forest**, and **XGBoost**.

---

## ✨ Features

| Page | What it shows |
|---|---|
| **Dashboard** | KPI cards, borough price bar chart, room-type donut, price box-plots, interactive listing density map |
| **EDA & Insights** | Price distributions, outlier analysis, borough/neighbourhood deep-dives, room-type heatmap, correlation matrix |
| **Price Predictor** | Live price estimate — choose borough, neighbourhood, room type, listing attributes; compare 3 models |
| **Model Metrics** | MAE, MSE, RMSE, R² for all 3 models; actual-vs-predicted scatter; full methodology |

---

## 🚀 Running Locally

```bash
git clone https://github.com/<your-username>/nyc-airbnb-app.git
cd nyc-airbnb-app
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Repo · branch `main` · main file `app.py`
4. Click **Deploy** ✅

---

## 📂 Project Structure

```
nyc_airbnb_app/
├── app.py
├── utils.py
├── requirements.txt
├── README.md
├── data/
│   └── AB_NYC_2019_processed01.csv
└── pages/
    ├── __init__.py
    ├── dashboard.py
    ├── eda.py
    ├── predictor.py
    └── metrics.py
```

---

## 🛠 Tech Stack

`Python` · `Streamlit` · `pandas` · `scikit-learn` · `XGBoost` · `Plotly`
