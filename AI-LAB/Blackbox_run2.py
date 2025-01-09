import os
import math
import numpy as np
import pandas as pd
import time

from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, Informer, NHITS, DLinear
from neuralforecast.losses.pytorch import RMSE

from datetime import datetime, timedelta

from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

import warnings
warnings.filterwarnings('once')

os.environ['NIXTLA_ID_AS_COL'] = '1'

df = pd.read_csv('../Dataset/ConsumptionIndustry.csv', sep=';')
df['HourDK'] = pd.to_datetime(df['HourDK'])
df['ConsumptionkWh'] = df['ConsumptionkWh'].str.replace(",", ".").astype(float)
df.index = df['HourDK']
df.drop(columns=['HourUTC', 'HourDK', 'MunicipalityNo', 'Branche'], inplace=True)

def loaddataset():
    consumption = pd.read_csv('ConsumptionIndustry.csv', sep=';')
    spot_prices = pd.read_csv('ELSpotPrices.csv', sep=';')

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


def prepare_neuralforecast_data(combined_data):
    combined_data = combined_data.reset_index(drop=True)

    combined_data = combined_data.rename(
        columns={'HourDK': 'ds', 'ConsumptionkWh': 'y'})

    combined_data['unique_id'] = 1
    combined_data.index = pd.to_datetime(combined_data['ds'])

    return combined_data[['unique_id', 'ds', 'y'] + [col for col in combined_data.columns if col not in ['unique_id', 'ds', 'y']]]


def sample_data(df, start_date, end_date):
    end_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(hours=25)
    return df[(df.index >= start_date) & (df.index <= end_date)]


def sample_data_with_train_window(df, start_date, end_date, train_window_size):
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df['ds'])

    start_date = datetime.strptime(
        start_date, '%Y-%m-%d') - timedelta(hours=train_window_size) + timedelta(hours=24)
    end_date = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(hours=24)

    return df[(df.index >= start_date) & (df.index <= end_date)]


def get_next_window(data, train_window_size, forecast_horizon):
    return data[:train_window_size], data[train_window_size:train_window_size + forecast_horizon]


def forecast_blackbox_model(model, model_name):
    nf = NeuralForecast(models=[model], freq='H')
    nf.fit(data_train)
    return nf.predict(data_test)[model_name]

def save_prediction_and_stats(runtime, config_name, df_predictions, df_true, prediction_path, stats_path):
    df_predictions.to_csv(prediction_path, header=False)

    try:
        df_stats = pd.read_csv(stats_path)
    except:
        df_stats = pd.DataFrame(columns=['model', 'runtime', 'mse', 'rmse', 'mae', 'mape'])

    new_row = {'model': config_name, 'runtime': runtime,
               'mse': mean_squared_error(df_predictions, df_true),
               'rmse': root_mean_squared_error(df_predictions, df_true),
               'mae': mean_absolute_error(df_predictions, df_true),
               'mape': mean_absolute_percentage_error(df_predictions, df_true)}
    new_row_df = pd.DataFrame([new_row]).dropna(axis=1, how='all')
    df_stats = pd.concat([df_stats, new_row_df], ignore_index=True)
    df_stats = df_stats.sort_values(
        by=['model', 'rmse'], ascending=True).reset_index(drop=True)

    df_stats.to_csv(stats_path, index=False)


if __name__ == '__main__':
    model_name = 'NHITS'
    date_start = '2023-11-01'
    date_end = '2024-11-01'

    # List of (window_train_size, forecast_horizon, model_config) tuples
    scenarios = [
        (336, 24, {'input_size': 24, 'max_steps': 3000, 'val_check_steps': 100, 'batch_size': 32, 'step_size': 5, 'scaler_type': 'robust'}),
        (1440, 336, {'input_size': 24, 'max_steps': 200, 'val_check_steps': 500, 'batch_size': 128, 'step_size': 1, 'scaler_type': 'minmax'}),
        (17520, 8760, {'input_size': 24, 'max_steps': 200, 'val_check_steps': 10, 'batch_size': 128, 'step_size': 5, 'scaler_type': 'robust'})
    ]

    combined_data = loaddataset()
    neuralforecast_data = prepare_neuralforecast_data(combined_data)

    for window_train_size, forecast_horizon, model_config in scenarios:
        config_name = f'{model_name}_{window_train_size}_{forecast_horizon}'
        warnings.filterwarnings("ignore")

        start_time = time.time()

        data = sample_data_with_train_window(neuralforecast_data, date_start, date_end, window_train_size)
        results = np.array([])
        iterations = 0
        max_iterations = math.ceil(8760 / forecast_horizon)

        while len(results) < 8760:
            iterations += 1
            print(f'{config_name}: Iteration {iterations}/{max_iterations}')

            if (len(results) + forecast_horizon) > 8760:
                forecast_horizon = 8760 - len(results)

            data_train, data_test = get_next_window(
                data, window_train_size, forecast_horizon)
            model = NHITS(h=forecast_horizon, loss=RMSE(), input_size=model_config['input_size'], max_steps=model_config['max_steps'], 
                          val_check_steps=model_config['val_check_steps'], batch_size=model_config['batch_size'], step_size=model_config['step_size'], 
                          scaler_type=model_config['scaler_type'])
            try:
                predictions = forecast_blackbox_model(model, model_name)
            except Exception as e:
                raise RuntimeError(f'Model failed to fit and forecast at iteration {iterations}')

            results = np.append(results, predictions.values)
            data = data.iloc[forecast_horizon:]

        end_time = time.time()

        warnings.filterwarnings("default")

        df_true = sample_data(df, date_start, date_end)
        df_predictions = pd.DataFrame(results)
        df_predictions.index = pd.date_range(start=date_start, periods=len(results), freq='h')

        save_prediction_and_stats(runtime=end_time - start_time, config_name=config_name, df_predictions=df_predictions, df_true=df_true,
                                  prediction_path=f'{config_name}.csv',
                                  stats_path=f'blackbox_run_stats.csv')
