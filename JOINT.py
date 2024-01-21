import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller


TARGET_VARIABLE = "Zone 1 Power Consumption"
N_PERIOD_TO_PREDICT = 35*140

def check_stationarity(series, threshold=0.05):
    result = adfuller(series)
    if result[1] < threshold:
        return True
    else:
        return False


# Load your dataset
data = pd.read_csv("Tetuan/Tetuan_City_power_consumption.csv", delimiter=',')
#preProcessing(data)

# Set 'DateTime' as index
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.set_index('DateTime', inplace=True)


# Create the "dataplots" directory if it doesn't exist
os.makedirs("dataplots", exist_ok=True)

# List of columns to plot, excluding 'DateTime'
columns_to_plot = [col for col in data.columns if col != 'DateTime'] #
print(data.columns)
seasonality = 143 * 7
# Apply rolling mean and plot each column
# for col in columns_to_plot:
#     # Ensure all values are positive before log transformation
#     min_value = data[col].min()
#     data[col] = data[col] + (-min_value + 1) if min_value <= 0 else data[col]

#     # Apply log transformation
#     data[col] = np.log(data[col])
#     # Apply differencing
#     data[col] = data[col].diff(seasonality)

#     # Apply rolling mean
#     data[col] = data[col].rolling(window=143).mean()

#     # Drop NaN values
#     data_to_plot = data[col].dropna()
#     plt.ylim(0, 100000)

#     # Plotting
#     data_to_plot.plot()
#     plt.savefig(f'dataplots/{col}_rolling_avg.png')
#     plt.close()



# Path to the directory where plots are saved
print("Plots saved in the 'dataplots' directory.")
data.replace('', np.nan, inplace=True)
data.dropna(inplace=True)
data.to_csv("stationaryDataset.csv", index=False)



#APPLY VECTOR AUTO REGRESSION
cols = data.columns
train = data[:int(0.9*(len(data)))]
valid = data[int(0.9*(len(data))):]

#Check for stationarity
#for col in columns_to_plot:
#    print({col})
#    if not check_stationarity(data[col]):
#        print(f"Series {col} is not stationary")
#        exit(4)

# Adjusting the VAR model fitting
'''
model = VAR(train)
for i in [1,2,3,4,5,6,7,8,9,10,20,30]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')
'''
#grangers_test(data, 9)
try:
    model = VAR(endog=train)
    model_fit = model.fit(maxlags=5)  # You can adjust the number of lags
    # Rest of your VAR code
except np.linalg.LinAlgError:
    print("SVD did not converge. Adjust the model or data preprocessing.")

# Forecasting
n_forecast = len(valid)  # Number of steps to forecast
predicted = model_fit.forecast(train.values[-model_fit.k_ar:], steps=n_forecast)

# Convert predictions to DataFrame
predicted_df = pd.DataFrame(predicted, columns=train.columns)
predicted_df.index = valid.index  # Align index with the valid dataset

# Invert transformations for "Zone 1 Power Consumption"
# ... [Invert transformations code specifically for 'Zone 1 Power Consumption'] ...

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
fig, ax = plt.subplots(2,figsize=(10, 6))
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

# Plot the results for "Zone 1 Power Consumption"
# plt.figure(figsize=(10, 6))
ax[1].plot(data['Zone 1 Power Consumption'], label='Actual')

ax[1].plot(pd.to_datetime(predicted_df['Zone 1 Power Consumption']), color='red', label='Predicted')
ax[1].set_title('Prediction vs Actual for Zone 1 Power Consumption')
ax[1].set_xlabel('Date')
ax[1].set_ylabel('Zone 1 Power Consumption')
ax[1].legend()
plt.show()
