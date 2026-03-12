import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="🌧️ Прогноз опадів",
    page_icon="🌧️",
    layout="wide"
)

st.title("🌧️ Прогноз опадів на основі Open-Meteo")
st.markdown("---")

# ─── Session state ──────────────────────────────────────────────────────────
for key in ["df", "model", "scaler", "feature_cols", "metrics", "model_name"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Налаштування")

lat = st.sidebar.number_input("Широта (latitude)",  value=50.4501, step=0.0001, format="%.4f")
lon = st.sidebar.number_input("Довгота (longitude)", value=30.5234, step=0.0001, format="%.4f")

default_end   = datetime.today().date() - timedelta(days=1)
default_start = default_end - timedelta(days=365)

start_date = st.sidebar.date_input("Дата початку", value=default_start)
end_date   = st.sidebar.date_input("Дата кінця",   value=default_end)

model_choice = st.sidebar.selectbox(
    "Алгоритм ML",
    ["Random Forest", "Logistic Regression", "Обидві моделі"]
)

# ─── Helpers ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "windspeed_10m_max", "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "rain_sum",
]


def fetch_data(lat, lon, start, end):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": str(start),
        "end_date":   str(end),
        "daily": ",".join([
            "precipitation_sum",
            "rain_sum",
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "windspeed_10m_max",
            "shortwave_radiation_sum",
            "et0_fao_evapotranspiration",
        ]),
        "timezone": "Europe/Kiev",
    }
    url  = "https://api.open-meteo.com/v1/forecast"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    daily = data["daily"]
    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    # Цільова змінна
    df["has_precipitation"] = (df["precipitation_sum"] > 0).astype(int)
    return df


def prepare_features(df):
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].copy()
    y = df["has_precipitation"].copy()

    # Заповнюємо пропуски медіаною
    X = X.fillna(X.median())
    valid = y.notna()
    return X[valid], y[valid], cols


def train_models(X_train, X_test, y_train, y_test, scaler, choice):
    results = {}

    if choice in ["Random Forest", "Обидві моделі"]:
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        results["Random Forest"] = {
            "model": rf,
            "pred": pred,
            "proba": rf.predict_proba(X_test)[:, 1],
        }

    if choice in ["Logistic Regression", "Обидві моделі"]:
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(scaler.transform(X_train), y_train)
        pred = lr.predict(scaler.transform(X_test))
        results["Logistic Regression"] = {
            "model": lr,
            "pred": pred,
            "proba": lr.predict_proba(scaler.transform(X_test))[:, 1],
        }

    return results


def compute_metrics(y_true, y_pred):
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred),  4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred,    zero_division=0), 4),
        "F1-score":  round(f1_score(y_true, y_pred,        zero_division=0), 4),
    }


# ════════════════════════════════════════════════════════════════════════════
# BLOCK 1 – DATA
# ════════════════════════════════════════════════════════════════════════════
st.header("1️⃣ Дані")

col1, col2 = st.columns(2)

with col1:
    if st.button("🌐 Отримати дані з Open-Meteo", use_container_width=True):
        with st.spinner("Завантаження даних..."):
            try:
                df = fetch_data(lat, lon, start_date, end_date)
                df.to_csv("weather_daily.csv")
                st.session_state.df = df
                st.success(f"✅ Завантажено {len(df)} днів ({start_date} — {end_date})")
            except Exception as e:
                st.error(f"❌ Помилка: {e}")

with col2:
    uploaded = st.file_uploader("📂 Або завантажити CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded, index_col=0, parse_dates=True)
        if "has_precipitation" not in df.columns:
            df["has_precipitation"] = (df["precipitation_sum"] > 0).astype(int)
        st.session_state.df = df
        st.success(f"✅ Прочитано {len(df)} рядків із CSV")

if st.session_state.df is not None:
    df = st.session_state.df
    st.subheader("Перегляд даних")
    st.dataframe(df.tail(10), use_container_width=True)

    # Statistic
    with st.expander("📊 Статистика та графіки"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Кількість днів", len(df))
        c2.metric("Днів з опадами", int(df["has_precipitation"].sum()))
        c3.metric("% днів без опадів", f"{100 * (1 - df['has_precipitation'].mean()):.1f}%")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        # precipitation_sum histogram
        axes[0].hist(df["precipitation_sum"].fillna(0), bins=30, color="#4e79a7", edgecolor="white")
        axes[0].set_title("Розподіл опадів (мм/день)")
        axes[0].set_xlabel("Опади, мм")

        # class balance
        counts = df["has_precipitation"].value_counts().sort_index()
        axes[1].bar(["Без опадів (0)", "Є опади (1)"], counts.values,
                    color=["#76b7b2", "#f28e2b"], edgecolor="white")
        axes[1].set_title("Баланс класів")
        st.pyplot(fig)

    st.download_button("⬇️ Скачати CSV", df.to_csv().encode(), "weather_daily.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════════════
# BLOCK 2 – TRAINING
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("2️⃣ Навчання моделі")

if st.button("🧠 Навчити модель", use_container_width=True, disabled=(st.session_state.df is None)):
    df = st.session_state.df
    X, y, feat_cols = prepare_features(df)

    if len(X) < 20:
        st.error("Замало даних для навчання (потрібно ≥ 20 днів).")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler().fit(X_train)

        with st.spinner("Навчання..."):
            results = train_models(X_train, X_test, y_train, y_test, scaler, model_choice)

        st.session_state.scaler      = scaler
        st.session_state.feature_cols = feat_cols

        # Pick best model by F1 for forecasting
        best_name, best_model = None, None
        all_metrics = {}
        for name, res in results.items():
            m = compute_metrics(y_test, res["pred"])
            all_metrics[name] = m
            if best_name is None or m["F1-score"] > all_metrics.get(best_name, {}).get("F1-score", 0):
                best_name  = name
                best_model = res["model"]

        st.session_state.model      = best_model
        st.session_state.model_name = best_name
        st.session_state.metrics    = all_metrics

        # Show metrics
        st.subheader("📈 Метрики якості")
        for name, m in all_metrics.items():
            st.markdown(f"**{name}**")
            cols = st.columns(4)
            for col, (k, v) in zip(cols, m.items()):
                col.metric(k, v)

        # Confusion matrix
        with st.expander("🔍 Матриця плутанини та Classification Report"):
            for name, res in results.items():
                st.markdown(f"**{name}**")
                cm   = confusion_matrix(y_test, res["pred"])
                fig2, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["Без опадів", "Є опади"],
                            yticklabels=["Без опадів", "Є опади"], ax=ax)
                ax.set_xlabel("Прогноз"); ax.set_ylabel("Факт")
                st.pyplot(fig2)
                st.text(classification_report(y_test, res["pred"],
                        target_names=["Без опадів", "Є опади"]))

        # Feature importance for RF
        if "Random Forest" in results:
            rf_model = results["Random Forest"]["model"]
            fi = pd.Series(rf_model.feature_importances_, index=feat_cols).sort_values(ascending=True)
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            fi.plot(kind="barh", ax=ax3, color="#4e79a7")
            ax3.set_title("Важливість ознак (Random Forest)")
            st.pyplot(fig3)

        st.success(f"✅ Модель навчена! Найкраща: **{best_name}** (F1 = {all_metrics[best_name]['F1-score']})")

# ════════════════════════════════════════════════════════════════════════════
# BLOCK 3 – FORECAST
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("3️⃣ Прогноз опадів")

model_ready = st.session_state.model is not None and st.session_state.df is not None

if not model_ready:
    st.info("⚠️ Спочатку завантажте дані та навчіть модель.")
else:
    df          = st.session_state.df
    feat_cols   = st.session_state.feature_cols
    model       = st.session_state.model
    scaler      = st.session_state.scaler
    model_name  = st.session_state.model_name

    available_dates = df.index.tolist()
    selected_date   = st.selectbox(
        "Оберіть день із датасету для перевірки прогнозу:",
        options=available_dates,
        index=len(available_dates) - 1,
        format_func=lambda d: d.strftime("%Y-%m-%d"),
    )

    if st.button("☁️ Зробити прогноз", use_container_width=True):
        row = df.loc[[selected_date], feat_cols].fillna(df[feat_cols].median())

        if "Logistic Regression" in model_name:
            X_scaled = scaler.transform(row)
            pred  = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
        else:
            pred  = model.predict(row)[0]
            proba = model.predict_proba(row)[0]

        prob_rain    = proba[1] * 100
        prob_no_rain = proba[0] * 100

        st.subheader(f"Результат прогнозу для **{selected_date.strftime('%d.%m.%Y')}**")

        if pred == 1:
            st.success(f"🌧️ **Очікуються опади!**")
        else:
            st.info(f"☀️ **Опадів не очікується.**")

        col_a, col_b = st.columns(2)
        col_a.metric("☔ Ймовірність опадів",     f"{prob_rain:.1f}%")
        col_b.metric("☀️ Ймовірність без опадів", f"{prob_no_rain:.1f}%")

        # Compare with actual
        if "has_precipitation" in df.columns:
            actual = df.loc[selected_date, "has_precipitation"]
            actual_txt = "✅ Є опади" if actual == 1 else "✅ Без опадів"
            pred_txt   = "🌧️ Є опади" if pred   == 1 else "☀️ Без опадів"
            match = "✔️ Збіг" if actual == pred else "✖️ Помилка"
            st.markdown(
                f"| | |\n|---|---|\n"
                f"|**Модель використовується**|{model_name}|\n"
                f"|**Прогноз моделі**|{pred_txt}|\n"
                f"|**Факт (з датасету)**|{actual_txt}|\n"
                f"|**Результат**|{match}|"
            )

        # Show feature values
        with st.expander("🔢 Вхідні дані для прогнозу"):
            st.dataframe(row.T.rename(columns={selected_date: "Значення"}))
