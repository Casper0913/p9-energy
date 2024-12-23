import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import optuna

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
from statsmodels.tsa.forecasting.theta import ThetaModel
from datetime import datetime, timedelta

from sklearn.metrics import root_mean_squared_error

import warnings
warnings.filterwarnings('once')

# Consumption data
df = pd.read_csv('ConsumptionIndustry.csv', sep=';')

df['HourDK'] = pd.to_datetime(df['HourDK'])
df['ConsumptionkWh'] = df['ConsumptionkWh'].str.replace(",", ".").astype(float)
df.index = df['HourDK']

df.drop(columns=['HourUTC', 'HourDK',
        'MunicipalityNo', 'Branche'], inplace=True)

# El-spot prices
df2 = pd.read_csv('ELSpotPrices.csv', sep=';')
df2['HourDK'] = pd.to_datetime(df2['HourDK'])
df2['SpotPriceDKK'] = df2['SpotPriceDKK'].str.replace(",", ".").astype(float)
df2.index = df2['HourDK']
# remove first row, since the measurement at that time is not present in other dataset
df2 = df2.iloc[1:]
df2.drop(columns=['HourUTC', 'HourDK', 'PriceArea',
         'SpotPriceEUR'], inplace=True)


def plot_data(data_train, data_test, predictions, save_at=''):
    plt.figure(figsize=(7, 3))
    plt.plot(data_train.index, data_train,
             label=f'Train ({data_train.index[0]} - {data_train.index[-1]})')
    plt.plot(data_test.index, data_test,
             label=f'Test ({data_test.index[0]} - {data_test.index[-1]})')
    plt.plot(data_test.index, predictions, label='Prediction')
    plt.title('Consumption in danish private households with prediction')
    plt.xlabel('Measurements')
    plt.ylabel('Power (kW / charger)')
    plt.legend()
    if save_at:
        plt.savefig(save_at)
    plt.show()


def sample_data_with_train_window(df, start_date, train_window_size):
    start_date = datetime.strptime(
        start_date, '%Y-%m-%d') - timedelta(hours=train_window_size)
    end_date = df.index[-1]
    return df[(df.index >= start_date) & (df.index <= end_date)]


def get_next_window(data, train_window_size, forecast_horizon):
    return data[:train_window_size], data[train_window_size:train_window_size + forecast_horizon]


def forecast_whitebox_model(model, forecast_horizon, model_name, exog_data_test=None):
    model_res = model.fit()

    if "SARIMA" in model_name and "STL" not in model_name:
        return model_res.get_forecast(steps=forecast_horizon, exog=exog_data_test).predicted_mean
    else:
        return model_res.forecast(steps=forecast_horizon)


def objective_SARIMAX(trial, data_train, data_test, forecast_horizon, exog_data_train=None, exog_data_test=None):
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    pdqs = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    order = trial.suggest_categorical('order', pdq)
    seasonal_order = trial.suggest_categorical('seasonal_order', pdqs)
    trend = trial.suggest_categorical('trend', ['n', 'c', 't', 'ct', None])
    model = SARIMAX(data_train, order=order,
                    seasonal_order=seasonal_order, exog=exog_data_train, trend=trend)
    mdl = model.fit(disp=0)
    predictions = mdl.forecast(steps=forecast_horizon, exog=exog_data_test)
    return root_mean_squared_error(data_test, predictions)


def objective_SARIMA(trial, data_train, data_test, forecast_horizon):
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    pdqs = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    order = trial.suggest_categorical('order', pdq)
    seasonal_order = trial.suggest_categorical('seasonal_order', pdqs)
    model = SARIMAX(data_train, order=order, seasonal_order=seasonal_order)
    mdl = model.fit(disp=0)
    predictions = mdl.forecast(steps=forecast_horizon)
    return root_mean_squared_error(data_test, predictions)


def safe_objective(trial):
    try:
        return objective_SARIMA(trial, data_train, data_test, forecast_horizon)
    except Exception as e:
        print(f"Failed trial: {e}. Skipped this trial.")
        return float('inf')


if __name__ == '__main__':
    date_start = '2023-11-01'
    window_train_size = 365*24*2  # hours
    forecast_horizon = 365*24  # hours
    # 336_24, 1440_336, 17520_8760

    trials = 100
    model_name = f'SARIMA_{window_train_size}_{forecast_horizon}'

    # start: date_start - window_train_size, end: last date in df
    data = sample_data_with_train_window(df, date_start, window_train_size)
    # exog_data = sample_data_with_train_window(df2, date_start, window_train_size)

    data_train, data_test = get_next_window(
        data, window_train_size, forecast_horizon)
    # exog_data_train, exog_data_test = get_next_window(exog_data, window_train_size, forecast_horizon)

    warnings.filterwarnings("ignore")
    study1 = optuna.create_study(direction='minimize')
    study1.optimize(safe_objective, n_trials=trials)

    trial = study1.best_trial
    print(f"Accuracy: {trial.value}")
    print(f"best params for {model_name}: {trial.params}")

    warnings.filterwarnings("default")

    # Save the results in CSV
    df_tuning = pd.DataFrame(columns=['model', 'accuracy', 'params'])

    new_row = {'model': model_name, 'accuracy': trial.value,
               'params': str(trial.params)}
    df_tuning = pd.concat(
        [df_tuning, pd.DataFrame([new_row])], ignore_index=True)
    df_tuning = df_tuning.sort_values(
        by=['model', 'accuracy', 'params'], ascending=True).reset_index(drop=True)
    df_tuning.to_csv('blackbox_tuning.csv', index=False)
