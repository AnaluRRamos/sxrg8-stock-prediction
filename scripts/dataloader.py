import yfinance as yf
import pandas as pd
import os

def download_data(ticker='EXS1.DE', start_date='2000-01-01', end_date='2024-01-01'):
    """
    Download historical data from Yahoo Finance and save it to a CSV file.
    """
    
    if not os.path.exists('../data'):
        os.makedirs('../data')
    
    
    data = yf.download(ticker, start=start_date, end=end_date)
    
    
    data.to_csv('../data/raw_data.csv')  
    return data

if __name__ == "__main__":
    download_data()

