import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

TARGET_VARIABLE = "Zone 1 Power Consumption"

def check_stationarity(series, threshold=0.05):
    result = adfuller(series)
    if result[1] < threshold:
        return True
    else:
        return False


# LoadDataset
data = pd.read_csv("Tetuan/Tetuan_City_power_consumption.csv", delimiter=',')
data.drop(['Zone 2 Power Consumption', 'Zone 3 Power Consumption'], axis=1, inplace=True)


data['DateTime'] = pd.to_datetime(data['DateTime'], format='%m/%d/%Y %H:%M')
data.set_index('DateTime', inplace=True)

os.makedirs("dataplots", exist_ok=True)

#Sample the dataset on a dayli mean.
data = data.resample('D').mean()
scaler = StandardScaler()

# Path to the directory where plots are saved
print("Plots saved in the 'dataplots' directory.")
data.replace('', np.nan, inplace=True)
data.dropna(inplace=True)
data.to_csv("stationaryDataset.csv", index=False)

#apply scaler transformation to make it stationary
scaled_data = scaler.fit_transform(data)
df_scaled = pd.DataFrame(scaled_data, index=data.index, columns=data.columns)


#First approach: VAR
test_VA_start_date = df_scaled.index[int(0.9 * len(df_scaled))]
train_VA = df_scaled[df_scaled.index < test_VA_start_date]
test_VA = df_scaled[df_scaled.index >= test_VA_start_date]

#Check for stationarity
#for col in columns_to_plot:
#    print({col})
#    if not check_stationarity(data[col]):
#        print(f"Series {col} is not stationary")
#        exit(4)


if train_VA.empty:
    raise ValueError("Training data is empty after preprocessing.")

for column in train_VA.columns:
    if check_stationarity(train_VA[column]):
        print(f"The variable {column} is stationary.")
    else:
        print(f"The variable {column} is not stationary and may need differencing or transformation.")

#grangers_test(data, 9)
model = VAR(train_VA)
model_fit = model.fit(maxlags=14)  # Adjust the number of lags as needed

# Convert predictions to DataFrame
n_forecast = len(test_VA)  # Number of steps to forecast
forecast = model_fit.forecast(train_VA.values[-model_fit.k_ar:], steps=n_forecast)
df_forecast = pd.DataFrame(forecast, index=test_VA.index, columns=df_scaled.columns)


target_variable = "Zone 1 Power Consumption"
ts_data = data[target_variable]

# Split the data into training and testing sets
train_size = int(0.9 * len(ts_data))
train, test = ts_data[:train_size], ts_data[train_size:]

# Check for stationarity
def check_stationarity(series, threshold=0.05):
  result = adfuller(series)
  if result[1] < threshold:
    print(f"The time series is stationary (p-value: {result[1]:.4f})")
  else:
    print(f"The time series is not stationary (p-value: {result[1]:.4f})")

# Apply Exponential Smoothing (Holt) model
print(train_size)
print(train)
print(train_VA)
model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=7)
fit_model = model.fit()

# Forecast future values
forecast = fit_model.forecast(steps=n_forecast)

# Plotting actual vs. predicted values
fig, ax = plt.subplots(2,figsize=(12, 8))
ax[0].plot(ts_data, label='Actual Data', color='blue')
ax[0].plot(forecast, label='Forecasted Data', color='red', alpha=0.8)
ax[0].set_title('Exponential Smoothing (Holt) Forecasting')
ax[0].set_xlabel('Date')
ax[0].set_ylabel(target_variable)
ax[0].legend()

# Set line width for both lines
# ax[0].set_rcParams['lines.linewidth'] = 2

# Adjust y-axis limits to fit the data
y_max = ts_data.max()
y_min = ts_data.min()
ax[0].set_ylim([y_min - 0.1 * y_max, y_max + 0.1 * y_max])

# Set x-axis limits to fit the data
x_max = ts_data.index.max()
x_min = ts_data.index.min()
ax[0].set_xlim([x_min, x_max])

# Set grid line styling
# ax[0].set_grid(linestyle='--', alpha=0.7)

# plt.show()

# Check for stationarity in the test data
check_stationarity(test)

# Evaluate the results

# Mean Absolute Error (MAE)
mae = mean_absolute_error(test, forecast)
print(f'Mean Absolute Error ES (MAE): {mae}')
mae_VAR = mean_absolute_error(test_VA, df_forecast)
print(f'Mean Absolute Error VAR (MAE): {mae_VAR}')
# Mean Squared Error (MSE)
mse = mean_squared_error(test, forecast)
print(f'Mean Squared Error ES (MSE): {mse}')
mse_VAR = mean_squared_error(test_VA, df_forecast)
print(f'Mean Squared Error VAR (MSE): {mse_VAR}')
# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error ES (RMSE): {rmse}')
rmse_VAR = np.sqrt(mse_VAR)
print(f'Root Mean Squared Error VAR (RMSE): {rmse_VAR}')
# Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred): 
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(test, forecast)
print(f'Mean Absolute Percentage Error ES (MAPE): {mape}%')
mape_VA = mean_absolute_percentage_error(test_VA, df_forecast)
print(f'Mean Absolute Percentage Error VAR (MAPE): {mape_VA}%')
# Plot the results for "Zone 1 Power Consumption"
# plt.figure(figsize=(10, 6))

ax[1].plot(train_VA.index, train_VA[TARGET_VARIABLE], label='Train', color='blue')
ax[1].plot(test_VA.index, test_VA[TARGET_VARIABLE], label='Test', color='orange')
ax[1].plot(df_forecast.index, df_forecast[TARGET_VARIABLE], label='Forecast', color='green')

ax[1].set_xticklabels([])  # Hide x-axis labels
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Power Consumption')
ax[1].set_title('Vector AutoRegresion Forecasting')
ax[1].legend()
plt.show()
