import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import os

# Plotting settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("## 1. Load Data")

# Load Streamway Data
try:
    if os.path.exists('streamway_data.csv'):
        df_streamway = pd.read_csv('streamway_data.csv', parse_dates=['created_at'], index_col='created_at')
        # Ensure index is sorted and unique
        df_streamway = df_streamway.sort_index()
        df_streamway = df_streamway[~df_streamway.index.duplicated(keep='first')]
        print(f"Loaded {len(df_streamway)} rows of streamway data.")
        print(f"Range: {df_streamway.index.min()} to {df_streamway.index.max()}")
    else:
        print("Error: streamway_data.csv not found.")
        exit(1)
except Exception as e:
    print(f"Error loading streamway data: {e}")
    exit(1)

# Resample to ensure regular 10-min intervals
df_streamway = df_streamway.resample('10min').mean().interpolate(method='time')

# Fetch Historical Forecast Data (Open-Meteo)
latitude = 51.8258112
longitude = -3.6611301

start_date = df_streamway.index.min().strftime('%Y-%m-%d')
end_date = df_streamway.index.max().strftime('%Y-%m-%d')

print(f"Fetching forecast data from {start_date} to {end_date}...")

forecast_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
forecast_params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": "precipitation_probability,precipitation",
    "timezone": "auto"
}

try:
    response = requests.get(forecast_url, params=forecast_params)
    if response.status_code == 200:
        data = response.json()
        hourly_data = data['hourly']
        
        df_forecast = pd.DataFrame({
            'time': pd.to_datetime(hourly_data['time']),
            'precip_forecast': hourly_data['precipitation'],
            'precip_prob': hourly_data['precipitation_probability']
        })
        df_forecast.set_index('time', inplace=True)
        print(f"Loaded {len(df_forecast)} rows of forecast data.")
    else:
        print(f"Error fetching data: {response.status_code}")
        df_forecast = pd.DataFrame()
except Exception as e:
    print(f"Exception fetching forecast data: {e}")
    df_forecast = pd.DataFrame()

if df_forecast.empty:
    print("Forecast data is empty. Exiting.")
    exit(1)

print("## 2. Data Alignment and Feature Engineering")

# Merge data
# First, reindex forecast to match streamway index (forward fill)
df_combined = df_streamway.join(df_forecast.reindex(df_streamway.index, method='ffill'))

# Feature Engineering
# 1. Lagged features for precipitation (based on 4-hour peak correlation)
# 4 hours = 24 * 10-min intervals
lag_4h = 4 * 6

df_combined['precip_lag_4h'] = df_combined['precip_forecast'].shift(lag_4h)

# Add other lags and rolling windows
for lag in [1, 2, 3, 5, 6]: # Hours
    df_combined[f'precip_lag_{lag}h'] = df_combined['precip_forecast'].shift(lag * 6)

# Rolling sum of precipitation over the last 4 hours
df_combined['precip_roll_sum_4h'] = df_combined['precip_forecast'].rolling(window=lag_4h).sum()
df_combined['precip_roll_mean_4h'] = df_combined['precip_forecast'].rolling(window=lag_4h).mean()

# Drop NaNs created by shifting
df_model = df_combined.dropna()

print(f"Data shape after feature engineering: {df_model.shape}")

print("## 3. Model Development")

# Define features and target
features = [
    'precip_forecast', 'precip_prob', 
    'precip_lag_4h', 'precip_roll_sum_4h', 'precip_roll_mean_4h',
    'precip_lag_1h', 'precip_lag_2h', 'precip_lag_3h', 'precip_lag_5h', 'precip_lag_6h'
]
target = 'streamway_depth_mm'

X = df_model[features]
y = df_model[target]

# Split into train and test (time-based split)
split_index = int(len(df_model) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# Train XGBoost Regressor
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Save plot
plt.figure(figsize=(15, 7))
plt.plot(y_test.index, y_test, label='Actual Depth', alpha=0.7)
plt.plot(y_test.index, y_pred, label='Predicted Depth', alpha=0.7, linestyle='--')
plt.title('Streamway Depth Forecast (Test Set)')
plt.xlabel('Time')
plt.ylabel('Depth (mm)')
plt.legend()
plt.savefig('forecast_plot.png')
print("Plot saved to forecast_plot.png")
