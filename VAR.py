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
data.set_index('DateTime', inplace=True)

# Create the "dataplots" directory if it doesn't exist
os.makedirs("dataplots", exist_ok=True)

# List of columns to plot, excluding 'DateTime'
columns_to_plot = [col for col in data.columns if col != 'DateTime']

seasonality = 143 * 7
# Apply rolling mean and plot each column
for col in columns_to_plot:
    # Ensure all values are positive before log transformation
    min_value = data[col].min()
    data[col] = data[col] + (-min_value + 1) if min_value <= 0 else data[col]

    # Apply log transformation
    data[col] = np.log(data[col])
    # Apply differencing
    data[col] = data[col].diff(seasonality)

    # Apply rolling mean
    data[col] = data[col].rolling(window=143).mean()

    # Drop NaN values
    data_to_plot = data[col].dropna()

    # Plotting
    #data_to_plot.plot()
    #plt.savefig(f'dataplots/{col}_rolling_avg.png')
    #plt.close()



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
grangers_test(data, 9)
try:
    model = VAR(endog=train)
    model_fit = model.fit(maxlags=9)  # You can adjust the number of lags
    # Rest of your VAR code
except np.linalg.LinAlgError:
    print("SVD did not converge. Adjust the model or data preprocessing.")
