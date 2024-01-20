import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates

TARGET_VARIABLE = "Zone 1 Power Consumption"

def check_stationarity(series, threshold=0.05):
    result = adfuller(series)
    if result[1] < threshold:
        return True
    else:
        return False


def df_test_transformation(df, test_start_date, scaler):
    # Apply differencing to make data stationary
    df_diff = df.diff().dropna()

    # Scale data using the previously defined scaler
    df_scaled = pd.DataFrame(scaler.fit_transform(df_diff),
                             columns=df_diff.columns,
                             index=df_diff.index)

    # Select only the data that belongs to the testing set
    df_test_processed = df_scaled[df_scaled.index > test_start_date]

    return df_test_processed


# Load your dataset
data = pd.read_csv("Tetuan/Tetuan_City_power_consumption.csv", delimiter=',')
data.drop(['Zone 2 Power Consumption', 'Zone 3 Power Consumption', 'diffuse flows', 'Wind Speed', 'Humidity'], axis=1, inplace=True)

#preProcessing(data)

# Set 'DateTime' as index
data.set_index('DateTime', inplace=True)

# Create the "dataplots" directory if it doesn't exist
os.makedirs("dataplots", exist_ok=True)

# List of columns to plot, excluding 'DateTime'
columns_to_plot = [col for col in data.columns if col != 'DateTime']

seasonality = 143 * 7

data = data.rolling(seasonality).mean().dropna()

scaler = StandardScaler()

# Transform data
scaled_values = scaler.fit_transform(data)

# Convert to dataframe
df_scaled = pd.DataFrame(scaled_values,
                         columns=data.columns,
                         index=data.index)

train_processed = df_test_transformation(data, df_scaled.index[-1], scaler)

# Apply rolling mean and plot each column
'''
for col in columns_to_plot:
    # Apply rolling mean

    # Ensure all values are positive before log transformation
    #min_value = data[col].min()
    #data[col] = data[col] + (-min_value + 1) if min_value <= 0 else data[col]

    # Apply log transformation
    data[col] = np.log(data[col])
    # Apply differencing
    data[col] = data[col].diff(seasonality)

    # Drop NaN values
    data_to_plot = data[col].dropna()
    plt.ylim(0, 100000)

    # Plotting
    data_to_plot.plot()
    plt.savefig(f'dataplots/{col}_rolling_avg.png')
    plt.close()
'''


# Path to the directory where plots are saved
#print("Plots saved in the 'dataplots' directory.")
#data.replace('', np.nan, inplace=True)
#data.dropna(inplace=True)
#data.to_csv("stationaryDataset.csv", index=False)

#APPLY VECTOR AUTO REGRESSION
cols = data.columns
train = data[:int(0.9*(len(data)))]
valid = data[int(0.9*(len(data))):]

#Check for stationarity

for variable in train_processed.columns:

    # Perform the ADF test
    result = adfuller(train_processed[variable])

    # Extract and print the p-value from the test result
    p_value = result[1]
    print("p-value:", p_value)

    # Interpret the result
    if p_value <= 0.05:
        print(f"The variable {variable} is stationary.\n")
    else:
        print(f"The variable {variable} is not stationary.\n")


#Granger causality test
# List of variables for the test
variables = data.columns.tolist()

# Exclude Zone Power Consumption variables from each other
'''
print("Granger Causality test")
zone_vars = ['Zone 1 Power Consumption']
variables = [var for var in variables if var not in zone_vars]
max_lags = 5
for var1 in variables:
    for var2 in zone_vars:
        if var1 != var2:
            print(f"Results for {var1} causing {var2}:")
            granger_test_result = grangercausalitytests(data[[var2, var1]], max_lags, verbose=True)
            print("\n")

# Granger test shows Temperature Humidity, Wind Speed, General Diffuse Flows, and Diffuse Flows all somehow influence power consuption
# Adjusting the VAR model fitting
'''
#Calculate the lags:
model = VAR(df_scaled)
optimal_lags = model.select_order()
print(f"Lags: {optimal_lags.selected_orders}")

model = VAR(endog=train)
model_fit = model.fit(maxlags=optimal_lags.selected_orders['bic'])  # You can adjust the number of lags

var_model = model_fit.model
#print(model_fit.summary())

# Forecasting
n_forecast = len(valid)  # Number of steps to forecast
forecast = model_fit.forecast(train.values[-optimal_lags.selected_orders['bic']:], steps=n_forecast)
df_forecast = pd.DataFrame(forecast,
                           columns=df_scaled.columns,
                           index=valid.iloc[:n_forecast].index)

# Define the figure and axis
fig, ax = plt.subplots(figsize=(15,5))

# Plot the training data
ax.plot(train.index, train[TARGET_VARIABLE], label='Train', color='blue')

# Plot the test data
ax.plot(valid.index, valid[TARGET_VARIABLE], label='Test', color='orange')

# Plot the forecast data
ax.plot(df_forecast.index, df_forecast[TARGET_VARIABLE], label='Forecast', color='green')

# Set major ticks format
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Beautify the plot
plt.grid(alpha=0.5, which='both')
plt.xlabel('Date')
plt.ylabel('Power Consumption')
plt.title('Train, Test, and Forecast Power Consumption')
plt.legend()

# Show the plot
plt.tight_layout()  # Adjust layout to fit all elements
plt.show()