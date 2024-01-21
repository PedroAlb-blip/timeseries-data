import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller

# Load your dataset
data = pd.read_csv(r'Tetuan\Tetuan_City_power_consumption.csv', delimiter=',')
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.set_index('DateTime', inplace=True)

print(data.columns)

# Choose the target variable for forecasting
target_variable = "Zone 1 Power Consumption"
ts_data = data[target_variable]

# Split the data into training and testing sets
train_size = int(len(ts_data) * 0.8)
train, test = ts_data[:train_size], ts_data[train_size:]

# Check for stationarity
def check_stationarity(series, threshold=0.05):
  result = adfuller(series)
  if result[1] < threshold:
    print(f"The time series is stationary (p-value: {result[1]:.4f})")
  else:
    print(f"The time series is not stationary (p-value: {result[1]:.4f})")

# Apply Exponential Smoothing (Holt) model
model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=143*7)
fit_model = model.fit()

# Forecast future values
forecast = fit_model.forecast(steps=len(test))

# Plotting actual vs. predicted values
fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(ts_data, label='Actual Data', color='blue')
plt.plot(forecast, label='Forecasted Data', color='red', alpha=0.8)
plt.title('Exponential Smoothing (Holt) Forecasting')
plt.xlabel('Date')
plt.ylabel(target_variable)
plt.legend()

# Set line width for both lines
plt.rcParams['lines.linewidth'] = 2

# Adjust y-axis limits to fit the data
y_max = ts_data.max()
y_min = ts_data.min()
plt.ylim([y_min - 0.1 * y_max, y_max + 0.1 * y_max])

# Set x-axis limits to fit the data
x_max = ts_data.index.max()
x_min = ts_data.index.min()
plt.xlim([x_min, x_max])

# Set grid line styling
plt.grid(linestyle='--', alpha=0.7)

plt.show()

# Check for stationarity in the test data
check_stationarity(test)

# Evaluate the results

# Mean Absolute Error (MAE)
mae = mean_absolute_error(test, forecast)
print(f'Mean Absolute Error (MAE): {mae}')

# Mean Squared Error (MSE)
mse = mean_squared_error(test, forecast)
print(f'Mean Squared Error (MSE): {mse}')

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred): 
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(test, forecast)
print(f'Mean Absolute Percentage Error (MAPE): {mape}%')