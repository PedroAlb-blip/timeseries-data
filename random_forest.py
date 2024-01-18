# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from math import sqrt

# Read the training data
df = pd.read_csv(r'C:\Users\admin\OneDrive\Documents\CSS 5. Semester\KddM2\Projekt\Tetuan\train_data.csv')

# Convert the date_time column to datetime datatype and perform feature engineering
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['Day'] = df['Datetime'].dt.day
df['Month'] = df['Datetime'].dt.month
df['Hour'] = df['Datetime'].dt.hour
df['Minute'] = df['Datetime'].dt.minute
df['Day of Week'] = df['Datetime'].dt.dayofweek + 1

# Define a function to convert month to quarter
def month_to_quarter(series):
    if series['Month'] <= 3:
        return 1
    elif 3 < series['Month'] <= 6:
        return 2
    elif 6 < series['Month'] <= 9:
        return 3
    elif 9 < series['Month'] <= 12:
        return 4

# Apply the month_to_quarter function to create a 'Quarter of Year' column
df['Quarter of Year'] = df.apply(month_to_quarter, axis='columns')

# Add 'Day of Year' column
df['Day of Year'] = df['Datetime'].dt.strftime('%j').astype(int)

# Drop unnecessary columns
df = df.drop(['Datetime'], axis=1)

# Reorder columns
df = df[['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows',
         'DiffuseFlows', 'Day', 'Month', 'Hour', 'Minute', 'Day of Week',
         'Quarter of Year', 'Day of Year', 'PowerConsumption_Zone1',
         'PowerConsumption_Zone2', 'PowerConsumption_Zone3']]

# Display the preprocessed data
print(df.head())

# Visualize the correlation matrix
corr = df.corr().round(decimals=3)
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(20, 20))
cmap = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=None, center=0, square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .9})

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the training data
scaler.fit(df)
df_scaled = scaler.transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Display the scaled data
print(df_scaled.head())

# Read the test data
test_data = pd.read_csv(r'C:\Users\admin\OneDrive\Documents\CSS 5. Semester\KddM2\Projekt\Tetuan\test_data.csv')

# Repeat the preprocessing steps for the test data
test_data['Datetime'] = pd.to_datetime(test_data['Datetime'])
test_data['Day'] = test_data['Datetime'].dt.day
test_data['Month'] = test_data['Datetime'].dt.month
test_data['Hour'] = test_data['Datetime'].dt.hour
test_data['Minute'] = test_data['Datetime'].dt.minute
test_data['Day of Week'] = test_data['Datetime'].dt.dayofweek + 1
test_data['Quarter of Year'] = test_data.apply(month_to_quarter, axis='columns')
test_data['Day of Year'] = test_data['Datetime'].dt.strftime('%j').astype(int)
test_data = test_data.drop(['Datetime'], axis=1)
test_data = test_data[['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows',
                        'DiffuseFlows', 'Day', 'Month', 'Hour', 'Minute', 'Day of Week',
                        'Quarter of Year', 'Day of Year']]

# Display the preprocessed test data
print(test_data.head())

# Split the data into input features and target variables
X_train = df.drop(['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3'], axis=1)
y_train1 = df['PowerConsumption_Zone1']
y_train2 = df['PowerConsumption_Zone2']
y_train3 = df['PowerConsumption_Zone3']

# Initialize MinMaxScaler for test data
scaler_test = MinMaxScaler()

# Fit and transform the test data
scaler_test.fit(test_data)
test_data_scaled = scaler_test.transform(test_data)
test_data_scaled = pd.DataFrame(test_data_scaled, columns=test_data.columns)

# Display the scaled test data
print(test_data_scaled.head())

# Initialize Random Forest regressor
rf1 = RandomForestRegressor(n_estimators=700, min_samples_split=2, min_samples_leaf=1, max_features=3)

# Train the model on the training data
rf1.fit(X_train, y_train1)

# Use the trained model to make predictions on the training data
predictions = rf1.predict(X_train)
print(predictions)

# Define a function to calculate model performance metrics
def calculate_model_performance(scores, model_name):
    rmse = np.sqrt(-1 * scores['test_neg_mean_squared_error'])
    mse = (-1 * scores['test_neg_mean_squared_error'])
    mae = (-1 * scores['test_neg_mean_absolute_error'])
    r2 = scores['test_r2']

    result = pd.DataFrame({'Model': [model_name, model_name], 'Metric': ['Mean', 'Std.'],
                           'MAE': [mae.mean(), mae.std()], 'MSE': [mse.mean(), mse.std()],
                           'RMSE': [rmse.mean(), rmse.std()], 'R2': [r2.mean(), r2.std()],
                           'Fit Time': [scores['fit_time'].mean(), scores['fit_time'].std()],
                           'Inference Time': [scores['score_time'].mean(), scores['score_time'].std()]})
    return result

# Perform cross-validation on the Random Forest model
scores = cross_validate(rf1, X_train, y_train1, cv=10,
                        scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'])
res = calculate_model_performance(scores, 'Random Forest')

# Display the model performance metrics
print(res)
