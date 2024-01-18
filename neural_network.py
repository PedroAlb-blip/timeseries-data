import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow import keras
from keras import layers

# Load your data
df = pd.read_csv(r'Tetuan\train_data.csv')

# Split the data into input features and target variable
X_train = df.drop(['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3'], axis=1)
y_train1 = df['PowerConsumption_Zone1']
y_train2 = df['PowerConsumption_Zone2']
y_train3 = df['PowerConsumption_Zone3']

# Standardize the input features
scaler = StandardScaler()
numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
X_train_scaled = scaler.fit_transform(X_train[numerical_cols])


# Split the data into training and validation sets
X_train_nn, X_val_nn, y_train1_nn, y_val1_nn, y_train2_nn, y_val2_nn, y_train3_nn, y_val3_nn = train_test_split(
    X_train_scaled, y_train1, y_train2, y_train3, test_size=0.1, random_state=42
)

# Define a simple feedforward neural network model
def create_nn_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')  # Adjust output neurons based on your problem
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the neural network models
nn_model1 = create_nn_model()
nn_model1.fit(X_train_nn, y_train1_nn, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

nn_model2 = create_nn_model()
nn_model2.fit(X_train_nn, y_train2_nn, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

nn_model3 = create_nn_model()
nn_model3.fit(X_train_nn, y_train3_nn, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Load your test data
# Replace 'your_test_data.csv' with the actual file path
test_data = pd.read_csv(r'Tetuan\test_data.csv')

# Preprocess the test data (similar to training data preprocessing)
X_test = df.drop(['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3'], axis=1)
y_test1 = df['PowerConsumption_Zone1']
y_test2 = df['PowerConsumption_Zone2']
y_test3 = df['PowerConsumption_Zone3']


# Standardize the test data
scaler = StandardScaler()
X_test_scaled = scaler.transform(test_data)

# Make predictions using the neural network models
pred_nn1 = nn_model1.predict(X_test_scaled)
pred_nn2 = nn_model2.predict(X_test_scaled)
pred_nn3 = nn_model3.predict(X_test_scaled)

# Create a DataFrame with the predictions
df_preds_nn = pd.DataFrame({'predicted_Zone1': pred_nn1.flatten(), 'predicted_Zone2': pred_nn2.flatten(), 'predicted_Zone3': pred_nn3.flatten()})

# Load your sample data
# Replace 'your_sample_data.csv' with the actual file path
sample = pd.read_csv('your_sample_data.csv')

# Get the datetime column from the sample DataFrame
datetime_col = sample['Datetime']

# Concatenate the two DataFrames
df_result_nn = pd.concat([datetime_col, df_preds_nn], axis=1)

# Save the result DataFrame to a CSV file
df_result_nn.to_csv('NeuralNetworkPredictions.csv', index=False)

# Display the result DataFrame
print(df_result_nn)
