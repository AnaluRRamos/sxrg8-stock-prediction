import pandas as pd
import matplotlib.pyplot as plt

def plot_predictions(data_path):
    """
    Plot the actual stock prices vs the predictions from LSTM, ARIMA, and Hybrid models.
    
    :param data_path: Path to the CSV file containing actual and predicted prices
    """
    data = pd.read_csv(data_path)
    

    actual_prices = data['Actual']
    lstm_predictions = data['LSTM_Predictions']
    arima_predictions = data['ARIMA_Predictions']
    hybrid_predictions = (lstm_predictions + arima_predictions) / 2

    
    plt.figure(figsize=(14, 8))
    plt.plot(data['Date'], actual_prices, label='Actual Prices', color='blue')
    plt.plot(data['Date'], lstm_predictions, label='LSTM Predictions', color='green')
    plt.plot(data['Date'], arima_predictions, label='ARIMA Predictions', color='orange')
    plt.plot(data['Date'], hybrid_predictions, label='Hybrid Predictions', color='red')
    plt.xticks(rotation=45)
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    plot_predictions('data/test_predictions.csv')
