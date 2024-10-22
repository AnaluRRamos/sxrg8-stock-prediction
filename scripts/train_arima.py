import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import argparse
import pickle
import os

def train_arima(data_path, order=(5, 1, 0), output_dir='../models', output_file='arima_model.pkl'):
    """
    Train the ARIMA model on the processed data and save the trained model.
    
    :param data_path: Path to the processed data CSV file
    :param order: The order of the ARIMA model (p, d, q)
    :param output_dir: The directory where the trained model will be saved
    :param output_file: The name of the file for saving the ARIMA model
    """
    data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    
    close_prices = data['Close']

    print(f"Training ARIMA model with order {order} on data from {data_path}")
    model = ARIMA(close_prices, order=order)
    model_fit = model.fit()


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'wb') as f:
        pickle.dump(model_fit, f)
    
    print(f"ARIMA model trained and saved to {output_path}")
    return model_fit

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train ARIMA model for stock prediction")
    parser.add_argument('--data', type=str, required=True, help="Path to the processed data CSV file")
    parser.add_argument('--order', type=int, nargs=3, default=[5, 1, 0], help="Order of the ARIMA model (p, d, q)")
    parser.add_argument('--output_dir', type=str, default='../models', help="Directory where the ARIMA model will be saved")
    parser.add_argument('--output_file', type=str, default='arima_model.pkl', help="Output file name for the ARIMA model")


    args = parser.parse_args()


    train_arima(data_path=args.data, order=tuple(args.order), output_dir=args.output_dir, output_file=args.output_file)
