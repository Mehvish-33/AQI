import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
file_path = '/content/delhi_aqi.csv'
data = pd.read_csv(file_path)

# Convert 'date' column to datetime
data['date'] = pd.to_datetime(data['date'])

# Feature Engineering: Extracting datetime features
data['hour'] = data['date'].dt.hour
data['day'] = data['date'].dt.day
data['month'] = data['date'].dt.month
data['season'] = data['month'] % 12 // 3 + 1  # 1: Winter, 2: Spring, 3: Summer, 4: Fall

# Creating Lag Features (Previous Hour Values)
for col in ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']:
    data[f'{col}_lag1'] = data[col].shift(1)

# Creating Rolling Mean Features (24-hour moving average)
for col in ['pm2_5', 'pm10']:
    data[f'{col}_rolling24'] = data[col].rolling(window=24, min_periods=1).mean()

# Drop rows with NaN values due to lag features
data = data.dropna()

# Define Target and Features
target = 'pm2_5'  # Predicting PM2.5 levels
features = [col for col in data.columns if col not in ['date', 'pm2_5']]  # Excluding target and datetime

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Streamlit App
st.title("Delhi AQI Predictor")

# Predict Future PM2.5 Level
latest_data = data.iloc[-1].copy()
latest_data['date'] = latest_data['date'] + pd.Timedelta(hours=1)
latest_data['hour'] = latest_data['date'].hour
latest_data['day'] = latest_data['date'].day
latest_data['month'] = latest_data['date'].month
latest_data['season'] = latest_data['month'] % 12 // 3 + 1

for col in ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']:
    latest_data[f'{col}_lag1'] = latest_data[col]

for col in ['pm2_5', 'pm10']:
    latest_data[f'{col}_rolling24'] = data[col].rolling(window=24, min_periods=1).mean().iloc[-1]

latest_features = latest_data[features].values.reshape(1, -1)
predicted_pm25 = rf_model.predict(latest_features)[0]

# Display AQI Meter
st.subheader("Predicted PM2.5 Level for Next Hour")
st.metric(label="PM2.5 Level (µg/m³)", value=f"{predicted_pm25:.2f}")

# AQI Categories
def get_aqi_category(value):
    if value <= 50:
        return "Good", "#00E400"
    elif value <= 100:
        return "Moderate", "#FFFF00"
    elif value <= 150:
        return "Unhealthy for Sensitive Groups", "#FF7E00"
    elif value <= 200:
        return "Unhealthy", "#FF0000"
    elif value <= 300:
        return "Very Unhealthy", "#8F3F97"
    else:
        return "Hazardous", "#7E0023"

category, color = get_aqi_category(predicted_pm25)
st.markdown(f'<h3 style="color: {color};">{category}</h3>', unsafe_allow_html=True)

# Feature Importance Plot
st.subheader("Feature Importance in AQI Prediction")
feature_importance = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_importance.index, palette='viridis', ax=ax)
plt.xlabel('Importance Score')
plt.ylabel('Features')
st.pyplot(fig)
