import pandas as pd
import os
from datetime import datetime, timedelta
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, Informer, NHITS, DLinear
from neuralforecast.losses.pytorch import MSE
from neuralforecast.losses.pytorch import DistributionLoss
from pytorch_forecasting import MAE

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np


def loaddataset():
    consumption = pd.read_csv(
        'ConsumptionIndustry.csv', sep=';')
    spot_prices = pd.read_csv(
        '/content/ELSpotPrices.csv', sep=';')

    # Convert comma decimal format to float
    consumption['ConsumptionkWh'] = consumption['ConsumptionkWh'].str.replace(
        ',', '.').astype(float)
    spot_prices['SpotPriceDKK'] = spot_prices['SpotPriceDKK'].str.replace(
        ',', '.').astype(float)

    # Remove first row, since the measurement at that time is not present in other dataset
    spot_prices = spot_prices.iloc[1:]

    # Merge datasets on HourDK
    combined_data = pd.merge(consumption, spot_prices,
                             on='HourDK', how='inner')

    # Drop unnecessary columns
    combined_data = combined_data.drop(
        ['HourUTC_x', 'HourUTC_y', 'SpotPriceEUR', 'MunicipalityNo', 'Branche', 'PriceArea'], axis=1)

    # Set HourDK as index
    combined_data.index = pd.to_datetime(
        combined_data['HourDK'])  # Ensure index is datetime

    combined_data['HourDK'] = pd.to_datetime(combined_data['HourDK'])
    combined_data['Hour'] = combined_data['HourDK'].dt.hour
    combined_data['DayOfWeek'] = combined_data['HourDK'].dt.dayofweek
    combined_data['IsWeekend'] = combined_data['DayOfWeek'].isin([
                                                                 5, 6]).astype(int)

    return combined_data

# Prepare the data for neuralforecast


def prepare_neuralforecast_data(combined_data):
    # Reset index while avoiding duplicates
    combined_data = combined_data.reset_index(drop=True)

    # Rename columns to fit neuralforecast conventions
    combined_data = combined_data.rename(
        columns={'HourDK': 'ds', 'ConsumptionkWh': 'y'})

    # Add unique_id for a single time series
    # Single series; use unique values if there are multiple series
    combined_data['unique_id'] = 1

    combined_data.index = pd.to_datetime(combined_data['ds'])

    return combined_data[['unique_id', 'ds', 'y'] + [col for col in combined_data.columns if col not in ['unique_id', 'ds', 'y']]]


def sample_data_with_train_window(df, start_date, end_date, train_window_size):
    # Ensure the index is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        # Use the 'ds' column as the datetime index
        df.index = pd.to_datetime(df['ds'])

    # Adjust start_date to account for the training window
    start_date = datetime.strptime(
        start_date, '%Y-%m-%d') - timedelta(hours=train_window_size) + timedelta(hours=24)
    end_date = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(hours=24)

    # Filter the DataFrame based on the datetime range
    return df[(df.index >= start_date) & (df.index <= end_date)]


def sample_data_with_train_window(df, start_date, end_date, train_window_size):
    start_date = datetime.strptime(
        start_date, '%Y-%m-%d') - timedelta(hours=train_window_size) + timedelta(hours=24)
    end_date = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(hours=24)
    return df[(df.index >= start_date) & (df.index <= end_date)]

# Get training, validation, and testing windows


def get_next_window(data, train_window_size, validation_window_size, forecast_horizon):
    train_window_size = int(train_window_size)
    validation_window_size = int(validation_window_size)
    forecast_horizon = int(forecast_horizon)

    train_data = data[:train_window_size]
    val_data = data[train_window_size:train_window_size +
                    validation_window_size]
    test_data = data[train_window_size + validation_window_size:
                     train_window_size + validation_window_size + forecast_horizon]

    return train_data, val_data, test_data

# Function to train the LSTM model


# def train_lstm_model(train_data, val_data, forecast_horizon):
#     model = LSTM(h=forecast_horizon, input_size=-1,
#                  loss=DistributionLoss(distribution='Normal', level=[80, 90]),
#                  scaler_type='robust',
#                  encoder_n_layers=2,
#                  encoder_hidden_size=128,
#                  context_size=10,
#                  decoder_hidden_size=128,
#                  decoder_layers=2,
#                  max_steps=200,
#                  )

#     nf = NeuralForecast(models=[model], freq='H', )
#     nf.fit(train_data)
#     forecasts = nf.predict(val_data)
#     return forecasts

def train_models(train_data, val_data, models):
    forecasts = []

    # Get the validation size
    val_size = len(val_data)

    for model in models:
        # Initialize NeuralForecast model
        nf = NeuralForecast(models=[model], freq='H')

        # Fit the model, passing validation size during fitting
        nf.fit(train_data, val_size=val_size)

        # Forecast using the validation data
        forecast = nf.predict(val_data)

        # Store the forecast
        forecasts.append(forecast)

    return forecasts


# Function to calculate RMSE
def calculate_rmse(actual, forecast):
    return np.sqrt(mean_squared_error(actual, forecast))

# Function to plot forecasts and RMSE comparison


def plot_forecasts_and_rmse(forecasts, actual, model_names):
    # Plotting the forecasts vs actual consumption
    plt.figure(figsize=(12, 6))
    for i, forecast in enumerate(forecasts):
        plt.plot(forecast['ds'], forecast['yhat'],
                 label=f'{model_names[i]} Forecast')

    # Plot actual consumption data
    plt.plot(actual['ds'], actual['y'],
             label='Actual Consumption', color='black', linestyle='--')

    plt.xlabel('Date')
    plt.ylabel('Consumption (kWh)')
    plt.title('Model Forecasts vs Actual Consumption')
    plt.legend()
    plt.show()

    # Plotting RMSE for each model
    rmse_values = [calculate_rmse(actual['y'], forecast['yhat'])
                   for forecast in forecasts]
    plt.figure(figsize=(8, 5))
    plt.bar(model_names, rmse_values, color='skyblue')
    plt.ylabel('RMSE')
    plt.title('RMSE Comparison of Models')
    plt.show()

# Modified main pipeline


def main():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    # Load and preprocess the data
    combined_data = loaddataset()
    neuralforecast_data = prepare_neuralforecast_data(combined_data)

    # Define parameters
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    train_window_size = 336  # 2 weeks of hourly data
    validation_window_size = 168  # 1 week
    forecast_horizon = 24  # 1 day

    # Sample data
    sampled_data = sample_data_with_train_window(
        neuralforecast_data, start_date, end_date, train_window_size)
    train_data, val_data, test_data = get_next_window(
        sampled_data, train_window_size, validation_window_size, forecast_horizon)

    # Define models to compare
    lstm_model = LSTM(h=forecast_horizon, input_size=1,
                      loss=DistributionLoss(
                          distribution='Normal', level=[80, 90]),
                      scaler_type='robust',
                      encoder_n_layers=2,
                      encoder_hidden_size=128,
                      context_size=10,
                      decoder_hidden_size=128,
                      decoder_layers=2,
                      max_steps=200,
                      )

    informer_model = Informer(
        h=forecast_horizon,
        input_size=24,
        hidden_size=16,
        conv_hidden_size=32,
        n_head=2,
        loss=DistributionLoss(
            distribution='Normal', level=[80, 90]),
        scaler_type='robust',
        learning_rate=1e-3,
        max_steps=5,
    )

    nhits_model = NHITS(
        h=forecast_horizon, input_size=1,
        loss=DistributionLoss(distribution='Normal', level=[80, 90]),
        max_steps=200,
    )

    dlinear_model = DLinear(
        h=forecast_horizon, input_size=1,
        loss=DistributionLoss(distribution='Normal', level=[80, 90]),
        max_steps=200,
    )

    # List of models for comparison
    models = [lstm_model, informer_model, nhits_model, dlinear_model]
    model_names = ['LSTM', 'Informer', 'NHITS', 'DLinear']

    # Train and forecast
    forecasts = train_models(train_data, val_data, models)


# Execute the main function
if __name__ == "__main__":
    main()
