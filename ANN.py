import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Deep learning libraries
from sklearn.neural_network import MLPClassifier

# Load dataset
df = pd.read_excel('final_panel_data.xlsx')

# Features and target
features = ['Return on Asset', 'Asset turnover', 'Real Interest Rate', 'GDP growth Annual', 'Electricity Prices']
X = df[features]
y = df['y']

# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-Validation-Test Split (70/10/20)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=0)

# Build the model
model = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', max_iter=150, early_stopping=True, random_state=0)

# Train
model.fit(X_train, y_train)

# Evaluate on TEST set (scaled)
test_accuracy = model.score(X_test, y_test)
print(f'\nTest Accuracy: {test_accuracy:.4f}')

# Optional: More detailed metrics
y_pred_probs = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_probs > 0.5).astype(int)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_probs):.4f}")

import joblib

# After scaling the training data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save the fitted scaler
joblib.dump(scaler, 'scaler.pkl')


import joblib

# Load the fitted scaler
scaler = joblib.load('scaler.pkl')

# Set up the Streamlit page
st.set_page_config(page_title="ANN Model Prediction", layout="centered")
st.title("📈 ANN Model - Bankruptcy Prediction")

st.markdown("Enter the financial indicators below to predict bankruptcy risk using a deep learning model ANN.")

with st.form("prediction_form"):
    return_on_asset = st.number_input("Return on Asset", min_value=-164.31, max_value=319.52, step=0.1)
    asset_turnover = st.number_input("Asset Turnover", min_value=0.0, max_value=21.06, step=0.1)
    real_interest_rate = st.number_input("Real Interest Rate", min_value=-8.77, max_value=7.76, step=0.1)
    gdp_growth = st.number_input("GDP Growth Annual", min_value=-1.27, max_value=7.28, step=0.1)
    electricity_prices = st.number_input("Electricity Prices", min_value=1.46, max_value=27.19, step=0.1)

    predict_button_clicked = st.form_submit_button("🔍 Predict")

if predict_button_clicked:
    input_data = np.array([[return_on_asset, asset_turnover, real_interest_rate, gdp_growth, electricity_prices]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict_proba(input_scaled)[:, 1]
    predicted_class = [int(prediction[0] > 0.5)]
    confidence = round(prediction[0] * 100, 2)

    st.markdown("---")
    if predicted_class[0] == 1:
        st.error("Prediction: **Bankrupt**")
    else:
        st.success("Prediction: **Not Bankrupt**")

    st.info(f"Confidence: **{confidence}%**")
