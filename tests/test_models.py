import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

# Load preprocessed data
data = pd.read_csv('data/processed_data.csv', index_col='Date', parse_dates=True)
close_prices = data['Close'].values
scaled_close_prices = data['Scaled_Close'].values

# Split into train/test sets (80/20 split)
train_size = int(len(close_prices) * 0.8)
test_data = close_prices[train_size:]
scaled_test_data = scaled_close_prices[train_size:]

# Load trained LSTM model
lstm_model = load_model('models/lstm_model.h5')

# Load trained ARIMA model
with open('models/arima_model.pkl', 'rb') as f:
    arima_model = pickle.load(f)

# Create sequences for LSTM prediction (assuming sequence length is 60)
sequence_length = 60
X_test_lstm = create_sequences(scaled_test_data, sequence_length)

# Generate LSTM predictions for the test data
lstm_predictions_scaled = lstm_model.predict(X_test_lstm)

# Inverse scale the LSTM predictions to return them to the original price range
scaler = MinMaxScaler(feature_range=(0, 1))  # Use the same scaler used during preprocessing
scaler.fit(close_prices.reshape(-1, 1))  # Fit on the entire close price dataset
lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)

# Flatten the LSTM predictions
lstm_predictions = lstm_predictions.flatten()

# Generate ARIMA predictions with the correct index
start_index = train_size  # Start from the end of the training data
end_index = len(close_prices) - 1  # End at the last point of the full dataset

arima_predictions = arima_model.predict(start=start_index, end=end_index)

# Align test data with LSTM predictions (remove the first `sequence_length` points)
adjusted_test_data = test_data[sequence_length:]

# Ensure predictions match the length of the test data
lstm_predictions = lstm_predictions[:len(adjusted_test_data)]
arima_predictions = arima_predictions[:len(adjusted_test_data)]

# Calculate hybrid predictions (average of LSTM and ARIMA)
hybrid_predictions = (lstm_predictions + arima_predictions) / 2

# Calculate metrics for LSTM, ARIMA, and Hybrid models
lstm_mse = mean_squared_error(adjusted_test_data, lstm_predictions)
lstm_mae = mean_absolute_error(adjusted_test_data, lstm_predictions)
lstm_rmse = np.sqrt(lstm_mse)

arima_mse = mean_squared_error(adjusted_test_data, arima_predictions)
arima_mae = mean_absolute_error(adjusted_test_data, arima_predictions)
arima_rmse = np.sqrt(arima_mse)

hybrid_mse = mean_squared_error(adjusted_test_data, hybrid_predictions)
hybrid_mae = mean_absolute_error(adjusted_test_data, hybrid_predictions)
hybrid_rmse = np.sqrt(hybrid_mse)

# Print the evaluation metrics
print(f"LSTM MSE: {lstm_mse:.4f}, MAE: {lstm_mae:.4f}, RMSE: {lstm_rmse:.4f}")
print(f"ARIMA MSE: {arima_mse:.4f}, MAE: {arima_mae:.4f}, RMSE: {arima_rmse:.4f}")
print(f"Hybrid (LSTM + ARIMA) MSE: {hybrid_mse:.4f}, MAE: {hybrid_mae:.4f}, RMSE: {hybrid_rmse:.4f}")

# Save predictions to CSV for visualization
predictions_df = pd.DataFrame({
    'Date': data.index[train_size + sequence_length:],  # Align with LSTM predictions
    'Actual': adjusted_test_data,
    'LSTM_Predictions': lstm_predictions,
    'ARIMA_Predictions': arima_predictions,
    'Hybrid_Predictions': hybrid_predictions
})
predictions_df.to_csv('data/test_predictions_with_metrics.csv', index=False)
