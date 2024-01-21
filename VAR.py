import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler

# Assuming 'DateTime' and 'Zone 1 Power Consumption' are columns in your CSV file
TARGET_VARIABLE = "Zone 1 Power Consumption"

def check_stationarity(series, threshold=0.05):
    result = adfuller(series)
    if result[1] < threshold:
        return True
    else:
        return False

# Load your dataset
data = pd.read_csv("Tetuan/Tetuan_City_power_consumption.csv", delimiter=',')
data.drop(['Zone 2 Power Consumption', 'Zone 3 Power Consumption'], axis=1, inplace=True)

# Set 'DateTime' as index
data['DateTime'] = pd.to_datetime(data['DateTime'], format='%m/%d/%Y %H:%M')
data.set_index('DateTime', inplace=True)

# Create the "dataplots" directory if it doesn't exist
os.makedirs("dataplots", exist_ok=True)

# Apply rolling mean with a window of 7 (assuming daily data)


data = data.resample('D').mean()
#data = data.rolling(7).mean().dropna()
scaler = StandardScaler()

# Scale the data
scaled_data = scaler.fit_transform(data)
df_scaled = pd.DataFrame(scaled_data, index=data.index, columns=data.columns)

# Split the dataset
test_start_date = df_scaled.index[int(0.9 * len(df_scaled))]
train = df_scaled[df_scaled.index < test_start_date]
test = df_scaled[df_scaled.index >= test_start_date]

# Check for stationarity
if train.empty:
    raise ValueError("Training data is empty after preprocessing.")

for column in train.columns:
    if check_stationarity(train[column]):
        print(f"The variable {column} is stationary.")
    else:
        print(f"The variable {column} is not stationary and may need differencing or transformation.")

# Train the VAR model
model = VAR(train)
model_fit = model.fit(maxlags=14)  # Adjust the number of lags as needed

# Forecasting
n_forecast = len(test)  # Number of steps to forecast
forecast = model_fit.forecast(train.values[-model_fit.k_ar:], steps=n_forecast)
df_forecast = pd.DataFrame(forecast, index=test.index, columns=df_scaled.columns)

# Plot the results
fig, ax = plt.subplots(figsize=(15,5))
ax.plot(train.index, train[TARGET_VARIABLE], label='Train', color='blue')
ax.plot(test.index, test[TARGET_VARIABLE], label='Test', color='orange')
ax.plot(df_forecast.index, df_forecast[TARGET_VARIABLE], label='Forecast', color='green')

ax.set_xticklabels([])  # Hide x-axis labels
plt.xlabel('Time')
plt.ylabel('Power Consumption')
plt.title('Train, Test, and Forecast Power Consumption Zone 1')
plt.legend()
plt.show()
