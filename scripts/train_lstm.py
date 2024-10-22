import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import argparse

def create_sequences(data, sequence_length):
    """
    Create sequences of the specified length from the dataset for LSTM training.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def train_lstm(data_path, sequence_length=60, epochs=10, batch_size=32):
    """
    Train the LSTM model on the processed data.
    """
    
    data = pd.read_csv(data_path)
    scaled_data = data['Scaled_Close'].values.reshape(-1, 1)

    
    X, y = create_sequences(scaled_data, sequence_length)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1)) 

    model.compile(optimizer='adam', loss='mean_squared_error')

  
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    
    model.save('../models/lstm_model.h5')

    print(f"Model saved to '../models/lstm_model.h5'")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LSTM model for stock price prediction")
    parser.add_argument('--data', type=str, required=True, help="Path to the processed data CSV file")
    parser.add_argument('--sequence_length', type=int, default=60, help="Length of the input sequences for LSTM")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    
    args = parser.parse_args()

    train_lstm(data_path=args.data, sequence_length=args.sequence_length, epochs=args.epochs, batch_size=args.batch_size)
