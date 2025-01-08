import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, Informer, NHITS, DLinear
from neuralforecast.losses.pytorch import RMSE
from neuralforecast.losses.pytorch import DistributionLoss
from pytorch_forecasting import MAE

from datetime import datetime, timedelta

from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('once')

os.environ['NIXTLA_ID_AS_COL'] = '1'


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


def sample_data_with_train_window(df, start_date, end_date, train_window_size):
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df['ds'])

    start_date = datetime.strptime(
        start_date, '%Y-%m-%d') - timedelta(hours=train_window_size) + timedelta(hours=24)
    end_date = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(hours=24)

    return df[(df.index >= start_date) & (df.index <= end_date)]


def get_next_window(data, train_window_size, forecast_horizon):
    return data[:train_window_size], data[train_window_size:train_window_size + forecast_horizon]


def objective_LSTM(trial, data_train, data_test, forecast_horizon):
    nf = NeuralForecast(
        models=[LSTM(h=forecast_horizon, input_size=-1, loss=RMSE(),
                     encoder_n_layers=trial.suggest_categorical(
                         'encoder_n_layers', [1, 2, 5, 10]),
                     encoder_hidden_size=trial.suggest_categorical(
                         'encoder_hidden_size', [100, 200, 300, 400]),
                     context_size=trial.suggest_categorical(
                         'context_size', [5, 10, 15, 20]),
                     decoder_hidden_size=trial.suggest_categorical(
                         'decoder_hidden_size', [100, 200, 300, 400]),
                     decoder_layers=trial.suggest_categorical(
                         'decoder_layers', [1, 2, 5, 10]),
                     max_steps=trial.suggest_categorical(
                         'max_steps', [200, 500, 1000, 3000]),
                     val_check_steps=trial.suggest_categorical(
                         'val_check_steps', [10, 20, 50, 100, 250, 500]),
                     batch_size=trial.suggest_categorical(
                         'batch_size', [16, 32, 64, 128]),
                     scaler_type=trial.suggest_categorical(
                         'scaler_type', ['standard', 'minmax', 'robust']),
                     )
                ],
        freq='H'
    )
    nf.fit(data_train)
    predictions = nf.predict(data_test)
    return root_mean_squared_error(data_test['y'], predictions['LSTM'])


def objective_Informer(trial, data_train, data_test, forecast_horizon):
    nf = NeuralForecast(
        models=[Informer(h=forecast_horizon, loss=RMSE(),
                         input_size=trial.suggest_categorical(
                             'input_size', [1, 2, 6, 12, 24, 48]),
                         hidden_size=trial.suggest_categorical(
                             'hidden_size', [128/2, 128, 128*2, 128*3]),
                         n_head=trial.suggest_categorical(
                             'n_head', [1, 2, 4, 8]),
                         conv_hidden_size=trial.suggest_categorical(
                             'conv_hidden_size', [16, 32, 64, 128, 256]),
                         encoder_layers=trial.suggest_categorical(
                             'encoder_layers', [1, 2, 3, 4]),
                         decoder_layers=trial.suggest_categorical(
                             'decoder_layers', [1, 2, 3, 4]),
                         max_steps=trial.suggest_categorical(
                             'max_steps', [200, 500, 1000, 3000]),
                         val_check_steps=trial.suggest_categorical(
                             'val_check_steps', [10, 20, 50, 100, 250, 500]),
                         batch_size=trial.suggest_categorical(
                             'batch_size', [16, 32, 64, 128]),
                         scaler_type=trial.suggest_categorical(
                             'scaler_type', ['standard', 'minmax', 'robust']),
                         )
                ],
        freq='H'
    )
    nf.fit(data_train)
    predictions = nf.predict(data_test)
    return root_mean_squared_error(data_test['y'], predictions['Informer'])


def objective_DLinear(trial, data_train, data_test, forecast_horizon):
    nf = NeuralForecast(
        models=[DLinear(h=forecast_horizon, loss=RMSE(),
                        input_size=trial.suggest_categorical(
                            'input_size', [1, 2, 6, 12, 24, 48]),
                        moving_avg_window=trial.suggest_categorical(
                            'moving_avg_window', [17, 21, 25, 33, 37, 41]),
                        max_steps=trial.suggest_categorical(
                            'max_steps', [200, 500, 1000, 3000]),
                        val_check_steps=trial.suggest_categorical(
                            'val_check_steps', [10, 20, 50, 100, 250, 500]),
                        batch_size=trial.suggest_categorical(
                            'batch_size', [16, 32, 64, 128]),
                        scaler_type=trial.suggest_categorical(
                            'scaler_type', ['standard', 'minmax', 'robust']),
                        )
                ],
        freq='H'
    )
    nf.fit(data_train)
    predictions = nf.predict(data_test)
    return root_mean_squared_error(data_test['y'], predictions['DLinear'])


def objective_NHITS(trial, data_train, data_test, forecast_horizon):
    nf = NeuralForecast(
        models=[NHITS(h=forecast_horizon, loss=RMSE(),
                      input_size=trial.suggest_categorical(
                          'input_size', [1, 2, 6, 12, 24, 48]),
                      max_steps=trial.suggest_categorical(
                          'max_steps', [200, 500, 1000, 3000]),
                      val_check_steps=trial.suggest_categorical(
                          'val_check_steps', [10, 20, 50, 100, 250, 500]),
                      batch_size=trial.suggest_categorical(
                          'batch_size', [16, 32, 64, 128]),
                      step_size=trial.suggest_categorical(
                          'step_size', [1, 2, 3, 4, 5]),
                      scaler_type=trial.suggest_categorical(
                          'scaler_type', ['standard', 'minmax', 'robust', 'identity']),
                      )
                ],
        freq='H'
    )
    nf.fit(data_train)
    predictions = nf.predict(data_test)
    return root_mean_squared_error(data_test['y'], predictions['NHITS'])


if __name__ == '__main__':
    date_start = '2023-11-01'
    date_end = '2024-11-01'
    window_train_size = 17520  # hours
    forecast_horizon = 8760  # hours
    # 336_24, 1440_336, 17520_8760
    trials = 20
    model_name = f'NHITS_{window_train_size}_{forecast_horizon}'

    combined_data = loaddataset()
    neuralforecast_data = prepare_neuralforecast_data(combined_data)
    data = sample_data_with_train_window(
        neuralforecast_data, date_start, date_end, window_train_size)
    data_train, data_test = get_next_window(
        data, window_train_size, forecast_horizon)

    def safe_objective(trial):
        try:
            return objective_NHITS(trial, data_train, data_test, forecast_horizon)
        except Exception as e:
            print(f"Failed trial: {e}. Skipped this trial.")
            return float('inf')

    warnings.filterwarnings("ignore")
    study1 = optuna.create_study(direction='minimize')
    study1.optimize(safe_objective, n_trials=trials)

    trial = study1.best_trial
    print(f"Accuracy: {trial.value}")
    print(f"best params for {model_name}: {trial.params}")
    warnings.filterwarnings("default")

    # Save the results in CSV
    if trial.value != float('inf'):
        try:
            df_tuning = pd.read_csv('blackbox_tuning.csv')
        except:
            df_tuning = pd.DataFrame(columns=['model', 'accuracy', 'params'])

        new_row = {'model': model_name, 'accuracy': trial.value,
                   'params': str(trial.params)}
        new_row_df = pd.DataFrame([new_row]).dropna(axis=1, how='all')
        df_tuning = pd.concat([df_tuning, new_row_df], ignore_index=True)
        df_tuning = df_tuning.sort_values(
            by=['model', 'accuracy', 'params'], ascending=True).reset_index(drop=True)
        df_tuning.to_csv('blackbox_tuning.csv', index=False)
