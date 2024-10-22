def create_sequences(data, sequence_length):
    """
    Create sequences of the specified length from the dataset for LSTM prediction.
    """
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

def evaluate_models(data_path, lstm_model_path, arima_model_path):
    """
    Evaluate and compare LSTM and ARIMA models.
    """

    data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    close_prices = data['Close'].values
    scaled_close = data['Scaled_Close'].values
    
  
    train_size = int(len(close_prices) * 0.8)
    test_size = len(close_prices) - train_size
    train_data, test_data = close_prices[:train_size], close_prices[train_size:]
    scaled_train_data, scaled_test_data = scaled_close[:train_size], scaled_close[train_size:]

    lstm_model = load_lstm_model(lstm_model_path)
    arima_model = load_arima_model(arima_model_path)

    sequence_length = 60
    X_test_lstm = create_sequences(scaled_test_data, sequence_length)
    
    if X_test_lstm.shape[0] == 0:
        raise ValueError(f"Not enough test data for LSTM prediction with sequence length {sequence_length}.")
    
    lstm_predictions = lstm_model.predict(X_test_lstm)
    lstm_predictions = lstm_predictions.reshape(-1)

    arima_predictions = arima_model.forecast(steps=test_size)


    lstm_mse = evaluate_model(test_data[-len(lstm_predictions):], lstm_predictions, model_name="LSTM")
    arima_mse = evaluate_model(test_data, arima_predictions, model_name="ARIMA")

    # Combine predictions 
    combined_predictions = (lstm_predictions + arima_predictions[-len(lstm_predictions):]) / 2
    combined_mse = evaluate_model(test_data[-len(lstm_predictions):], combined_predictions, model_name="Hybrid (LSTM + ARIMA)")

    return lstm_mse, arima_mse, combined_mse
