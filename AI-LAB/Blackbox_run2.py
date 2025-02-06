import os
import math
import numpy as np
import pandas as pd
import time

from ray import tune
import optuna

from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, Informer, NHITS, DLinear
from neuralforecast.auto import AutoNHITS, AutoDLinear
from neuralforecast.losses.pytorch import RMSE, MAE

from utilsforecast.plotting import plot_series

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

from datetime import datetime, timedelta

from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

import warnings
warnings.filterwarnings('once')


os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['NIXTLA_ID_AS_COL'] = '1'


df = pd.read_csv('../Dataset/ConsumptionIndustry.csv', sep=';')
df['HourDK'] = pd.to_datetime(df['HourDK'])
df['ConsumptionkWh'] = df['ConsumptionkWh'].str.replace(",", ".").astype(float)
df.index = df['HourDK']
df.drop(columns=['HourUTC', 'HourDK',
        'MunicipalityNo', 'Branche'], inplace=True)


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

    combined_data['HourDK'] = pd.to_datetime(combined_data['HourDK'])
    combined_data['Hour'] = combined_data['HourDK'].dt.hour
    combined_data['DayOfWeek'] = combined_data['HourDK'].dt.dayofweek
    combined_data['IsWeekend'] = combined_data['DayOfWeek'].isin([
                                                                 5, 6]).astype(int)
    return combined_data

def prepare_neuralforecast_data(combined_data):
    combined_data = combined_data.reset_index(drop=True)
    combined_data = combined_data.rename(columns={'HourDK': 'ds', 'ConsumptionkWh': 'y'})
    combined_data['unique_id'] = 1
    return combined_data

def prepare_statsforecast_data(combined_data):
    combined_data = combined_data.reset_index(drop=True)

    combined_data = combined_data.rename(
        columns={'HourDK': 'ds', 'ConsumptionkWh': 'y'})

    combined_data['unique_id'] = "1"
    combined_data['ds'] = combined_data['ds'].astype('object')
    combined_data['unique_id'] = combined_data['unique_id'].astype('object')
    combined_data['ds'] = pd.to_datetime(combined_data['ds'])
    return combined_data[['unique_id', 'ds', 'y']]

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


def forecast_blackbox_model(model, model_name, data_train, data_test):
    nf = NeuralForecast(models=[model], freq='H')
    nf.fit(data_train)
    return nf.predict(data_test)[model_name]

def forecast_statsforecast_model(model):
    sf = StatsForecast(models=[model], freq='H')
    data_train.reset_index(drop=True, inplace=True)
    data_test.reset_index(drop=True, inplace=True)
    sf.fit(df=data_train)
    return sf.predict(h=len(data_test))['AutoARIMA']

def save_prediction_and_stats(runtime, config_name, df_predictions, df_true, prediction_path, stats_path):
    df_predictions.to_csv(prediction_path, header=False)

    try:
        df_stats = pd.read_csv(stats_path)
    except:
        df_stats = pd.DataFrame(
            columns=['model', 'runtime', 'mse', 'rmse', 'mae', 'mape'])

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

def config_nhits(trial):
    return {
        "input_size": trial.suggest_categorical(          # Length of input window
            "input_size", (48, 48*2, 48*3)                
        ),                                                
        "start_padding_enabled": True,                                          
        "n_blocks": 5 * [1],                              # Length of input window
        "mlp_units": 5 * [[64, 64]],                      # Length of input window
        "n_pool_kernel_size": trial.suggest_categorical(  # MaxPooling Kernel size
            "n_pool_kernel_size",
            (5*[1], 5*[2], 5*[4], [8, 4, 2, 1, 1])
        ),     
        "n_freq_downsample": trial.suggest_categorical(   # Interpolation expressivity ratios
            "n_freq_downsample",
            ([8, 4, 2, 1, 1],  [1, 1, 1, 1, 1])
        ),     
        "learning_rate": trial.suggest_float(             # Initial Learning rate
            "learning_rate",
            low=1e-4,
            high=1e-2,
            log=True,
        ),            
        "scaler_type": None,                              # Scaler type
        "max_steps": 1000,                                # Max number of training iterations
        "batch_size": trial.suggest_categorical(          # Number of series in batch
            "batch_size",
            (1, 4, 10),
        ),                   
        "windows_batch_size": trial.suggest_categorical(  # Number of windows in batch
            "windows_batch_size",
            (128, 256, 512),
        ),      
        "random_seed": trial.suggest_int(                 # Random seed   
            "random_seed",
            low=1,
            high=20,
        ),                      
    }

if __name__ == '__main__':
    model_name = 'NHITS'
    date_start = '2021-01-15'
    date_end = '2021-02-01'

    window_train_size = 336
    forecast_horizon = 24
    config_name = f'{model_name}_{window_train_size}_{forecast_horizon}'
    results = np.array([])

    combined_data = loaddataset()
    shorthand_data = prepare_neuralforecast_data(combined_data)
    historic_exog = combined_data[['SpotPriceDKK']]
    future_exog = combined_data[['Hour', 'DayOfWeek', 'IsWeekend']]

    warnings.filterwarnings("ignore")

    start_time = time.time()

    data_train, data_test = get_next_window(shorthand_data, window_train_size, forecast_horizon)

    model = AutoNHITS(h=forecast_horizon, config=config_nhits, loss=MAE(), backend='optuna', num_samples=50)
    model2 = NHITS(h=forecast_horizon, input_size=2,  loss=MAE(), hist_exog_list=historic_exog, futr_exog_list=future_exog)

    try:
        nf = NeuralForecast(models=[model2], freq='H')
        nf.fit(data_train)
        predictions = nf.predict(futr_df=future_exog)
        predictions.columns = predictions.columns.str.replace('-median', '')
    except Exception as e:
        raise RuntimeError(e)

    results = np.append(results, predictions[model_name].values)

    end_time = time.time()

    warnings.filterwarnings("default")

    df_true = df.loc[(df.index >= '2021-01-15 00:00:00') & (df.index <= '2021-01-15 23:00:00')]
    df_predictions = pd.DataFrame(results)
    df_predictions.index = pd.date_range(start=date_start, periods=len(results), freq='h')

    save_prediction_and_stats(runtime=end_time - start_time, config_name=config_name, df_predictions=df_predictions, df_true=df_true, prediction_path=f'{config_name}.csv', stats_path=f'blackbox_run_stats.csv')

    plot_series(shorthand_data.head(window_train_size + len(predictions)), predictions) #plot_random=False, max_insample_length=48 * 3, level=[80, 90]
