import math
import numpy as np
import pandas as pd
import time

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
from statsmodels.tsa.forecasting.theta import ThetaModel
from datetime import datetime, timedelta

from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('once')

#consumption data
df = pd.read_csv('ConsumptionIndustry.csv', sep=';')
df['HourDK'] = pd.to_datetime(df['HourDK'])
df['ConsumptionkWh'] = df['ConsumptionkWh'].str.replace(",", ".").astype(float)
df.index = df['HourDK']
df.drop(columns=['HourUTC', 'HourDK', 'MunicipalityNo', 'Branche'], inplace=True)

#spot prices
df2 = pd.read_csv('ELSpotPrices.csv', sep=';')
df2['HourDK'] = pd.to_datetime(df2['HourDK'])
df2['SpotPriceDKK'] = df2['SpotPriceDKK'].str.replace(",", ".").astype(float)
df2.index = df2['HourDK']
df2 = df2.iloc[1:]
df2.drop(columns=['HourUTC', 'HourDK', 'PriceArea', 'SpotPriceEUR'], inplace=True)

def sample_data(df, start_date, end_date):
  end_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(hours=25)
  return df[(df.index >= start_date) & (df.index <= end_date)]

def sample_data_with_train_window(df, start_date, end_date, train_window_size):
  start_date = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(hours=train_window_size) + timedelta(hours=24)
  end_date = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(hours=24)
  return df[(df.index >= start_date) & (df.index <= end_date)]

def get_next_window(data, train_window_size, forecast_horizon):
  return data[:train_window_size], data[train_window_size:train_window_size + forecast_horizon]

def forecast_whitebox_model(model, forecast_horizon, model_name, exog_data_test=None):
  model_res = model.fit(disp=0)

  if "SARIMA" in model_name:
    return model_res.get_forecast(steps=forecast_horizon, exog=exog_data_test).predicted_mean
  else:
    return model_res.forecast(steps=forecast_horizon)

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
  df_stats = df_stats.sort_values(by=['model', 'rmse'], ascending=True).reset_index(drop=True)

  df_stats.to_csv(stats_path, index=False)

if __name__ == '__main__':
  model_name = 'SARIMA'
  date_start = '2023-11-01'
  date_end = '2024-11-01'

  # List of (window_train_size, forecast_horizon, model_config) tuples
  scenarios = [
    (336, 24, {'order': (2, 1, 2), 'seasonal_order': (2, 2, 2, 12)}),
    (1440, 336, {'order': (0, 0, 1), 'seasonal_order': (2, 0, 2, 12)}),
    (17520, 8760, {'order': (0, 0, 0), 'seasonal_order': (1, 1, 2, 12)})
  ]

  for window_train_size, forecast_horizon, model_config in scenarios:
    config_name = f'{model_name}_{window_train_size}_{forecast_horizon}'

    warnings.filterwarnings("ignore")

    start_time = time.time()
    scaler = MinMaxScaler()

    data = sample_data_with_train_window(df, date_start, date_end, window_train_size)
    # exog_data = sample_data_with_train_window(df2, date_start, date_end, window_train_size)
    results = np.array([])
    iterations = 0
    max_iterations = math.ceil(8760 / forecast_horizon)

    while len(results) < 8760:
      iterations += 1
      print(f'{config_name}: Iteration {iterations}/{max_iterations}')

      if (len(results) + forecast_horizon) > 8760: 
        forecast_horizon = 8760 - len(results)
      
      data_train, data_test = get_next_window(data, window_train_size, forecast_horizon)
      # exog_data_train, exog_data_test = get_next_window(exog_data, window_train_size, forecast_horizon)

      data_train_scaled = scaler.fit_transform(data_train[['ConsumptionkWh']])
      data_train = pd.DataFrame(data_train_scaled, columns=['ConsumptionkWh'], index=data_train.index)
      model = SARIMAX(data_train, order=model_config['order'], seasonal_order=model_config['seasonal_order'])
      try:
        predictions_scaled = forecast_whitebox_model(model, forecast_horizon, model_name)
        predictions = scaler.inverse_transform(predictions_scaled.values.reshape(-1, 1))
        predictions = pd.Series(predictions.flatten(), index=data_test.index)
      except Exception as e:
          raise RuntimeError(f'Model failed to fit and forecast at iteration {iterations}')

      results = np.append(results, predictions.values)

      data = data.iloc[forecast_horizon:]
      # exog_data = exog_data.iloc[forecast_horizon:]

    end_time = time.time()

    warnings.filterwarnings("default")

    df_true = sample_data(df, date_start, date_end)
    df_predictions = pd.DataFrame(results)
    df_predictions.index = pd.date_range(start=date_start, periods=len(results), freq='h')

    save_prediction_and_stats(runtime=end_time - start_time, config_name=config_name, df_predictions=df_predictions, df_true=df_true,
                              prediction_path=f'{config_name}.csv',
                              stats_path=f'whitebox_run_stats.csv')