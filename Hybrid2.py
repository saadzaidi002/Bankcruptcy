# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load data
df = pd.read_excel("final_panel_data.xlsx")
features = ['Return on Asset', 'Asset turnover', 'Real Interest Rate', 
            'GDP growth Annual', 'Electricity Prices']
X = df[features]
y = df['y']

# Preprocessing
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# 1. Random Forest Model
rf_model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=0)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

# 2. ANN Model
ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

ann_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
ann_pred = (ann_model.predict(X_test) > 0.5).astype(int)
print("ANN Accuracy:", accuracy_score(y_test, ann_pred))

# 3. Hybrid Model (Average Probabilities)
rf_proba = rf_model.predict_proba(X_test)[:, 1]
ann_proba = ann_model.predict(X_test).flatten()
hybrid_proba = (rf_proba + ann_proba) / 2
hybrid_pred = (hybrid_proba > 0.5).astype(int)

print("\nHybrid Model Performance:")
print("Accuracy:", accuracy_score(y_test, hybrid_pred))
print("AUC-ROC:", roc_auc_score(y_test, hybrid_proba))
print(classification_report(y_test, hybrid_pred))


import joblib

# Save models
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save ANN model
ann_model.save("ann_model.h5")

import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load trained models and scaler
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
ann_model = load_model("ann_model.h5")

st.set_page_config(page_title="Hybrid Model Prediction", layout="centered")

st.title("📈 Hybrid Model - Bankruptcy Prediction")

st.markdown("Enter the economic indicators below to predict the financial outcome using a hybrid machine learning model combining Random Forest and ANN.")

with st.form("prediction_form"):
    return_on_asset = st.number_input("Return on Asset", min_value=-165.0, max_value=320.0, value=4.8, step=0.1)
    asset_turnover = st.number_input("Asset Turnover", min_value=0.0, max_value=22.0, value=1.1, step=0.1)
    real_interest_rate = st.number_input("Real Interest Rate", min_value=-10.0, max_value=8.0, value=2.0, step=0.1)
    gdp_growth = st.number_input("GDP Growth (Annual %)", min_value=-2.0, max_value=8.0, value=3.8, step=0.1)
    electricity_prices = st.number_input("Electricity Prices", min_value=1.0, max_value=30.0, value=12.8, step=0.1)

    predict_button_clicked = st.form_submit_button("🔍 Predict")

if predict_button_clicked:
    input_data = np.array([[return_on_asset, asset_turnover, real_interest_rate, gdp_growth, electricity_prices]])
    input_scaled = scaler.transform(input_data)

    rf_proba = rf_model.predict_proba(input_scaled)[:, 1]
    ann_proba = ann_model.predict(input_scaled).flatten()
    hybrid_proba = (rf_proba + ann_proba) / 2
    hybrid_pred = (hybrid_proba > 0.5).astype(int)

    st.markdown("---")
    st.subheader("📊 Prediction Result (Hybrid Model)")

    if hybrid_pred[0] == 1:
        st.error(f"Prediction: **Bankrupt** (probability: {hybrid_proba[0]*100:.2f}%)")
    else:
        st.success(f"Prediction: **Not Bankrupt** (probability: {(1 - hybrid_proba[0])*100:.2f}%)")

    st.info(f"Bankruptcy Probability: **{hybrid_proba[0]*100:.2f}%**")
