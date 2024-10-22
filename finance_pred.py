import subprocess

# Step 1: Download data
subprocess.call(['python', 'scripts/data_loader.py'])

# Step 2: Preprocess data
subprocess.call(['python', 'scripts/preprocess.py', '--file', 'data/raw_data.csv'])

# Step 3: Train LSTM model
subprocess.call(['python', 'scripts/train_lstm.py', '--data', 'data/processed_data.csv'])

# Step 4: Train ARIMA model
subprocess.call(['python', 'scripts/train_arima.py', '--data', 'data/processed_data.csv'])

# Step 5: Evaluate models and combine predictions
subprocess.call(['python', 'scripts/model_evaluation.py', '--lstm', 'models/lstm_model.h5', '--arima', 'models/arima_model.pkl'])

# Step 6: Run Streamlit app for real-time prediction (optional)
# subprocess.call(['streamlit', 'run', 'streamlit_app/app.py'])
