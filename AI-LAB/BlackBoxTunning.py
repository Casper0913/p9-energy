""" Code for black box tuning of the hyperparameters of different models """

from datetime import datetime, timedelta
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import optuna  # Hyperparameter optimization
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.multiprocessing as mp


class EnergyTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, output_size, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(EnergyTransformer, self).__init__()
        self.embedding = nn.Embedding(input_size, d_model)
        self.d_model = d_model
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, output_size)

        # Positional encoding initialization
        self.register_buffer("positional_encoding",
                             self.get_positional_encoding(512, d_model))

    def get_positional_encoding(self, max_seq_len, d_model):
        pos = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        # print(f"x shape: {x.shape}, x dtype: {x.dtype}")
        seq_len = x.size(1)
        x = x.long()  # or x = x.int()
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.positional_encoding[:, :seq_len, :]
        x = self.transformer(x, x)  # Assuming input is both src and tgt
        return self.fc_out(x[:, -1, :])  # Output for the last sequence step


class EnergyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(EnergyLSTM, self).__init__()

        if num_layers == 1:
            dropout = 0  # Set dropout to 0 if only one layer is used

        self.gru = nn.LSTM(input_size, hidden_size,
                           num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward pass through GRU
        gru_out, _ = self.gru(x)
        if gru_out.dim() == 2:
            last_output = gru_out
        else:
            last_output = gru_out[:, -1, :]
        output = self.fc_out(last_output)
        return output


# class EnergyGRU(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
#         super(EnergyGRU, self).__init__()
#         self.gru = nn.GRU(input_size, hidden_size,
#                           num_layers=num_layers, dropout=dropout, batch_first=True)
#         self.fc_out = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         gru_out, _ = self.gru(x)
#         if gru_out.dim() == 2:
#             last_output = gru_out
#         else:
#             last_output = gru_out[:, -1, :]
#         last_output = self.dropout(last_output)  # Apply dropout here
#         output = self.fc_out(last_output)
#         return output


class EnergyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(EnergyGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # Dropout only applies if num_layers > 1
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout layer (applied after GRU outputs)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # Initialize hidden state for first input
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Pass through GRU layer
        out, _ = self.gru(x, h0)
        
        # Apply dropout to the last output
        last_output = self.dropout(out[:, -1, :])  # Take the last output in the sequence
        
        # Pass through fully connected layer
        out = self.fc(last_output)
        
        return out


def read_data():
    """ Function to read the data """
    df = pd.read_csv(
        '/Users/casper/Documents/GitHub/p9-energy/Dataset/ConsumptionIndustry.csv', sep=';')
    df2 = pd.read_csv(
        '/Users/casper/Documents/GitHub/p9-energy/Dataset/ElspotPrices.csv', sep=';')

    # Load the dataset for colab
    # df = pd.read_csv('ConsumptionIndustry.csv', sep=';')

    # Convert HourDK to datetime
    df['HourDK'] = pd.to_datetime(df['HourDK'])

    # Convert ConsumptionkWh to numeric
    df['ConsumptionkWh'] = df['ConsumptionkWh'].str.replace(
        ",", ".").astype(float)

    # El spot prices
    df2['HourDK'] = pd.to_datetime(df2['HourDK'])
    df2['SpotPriceDKK'] = df2['SpotPriceDKK'].str.replace(
        ",", ".").astype(float)
    df2.index = df2['HourDK']
    # remove first row, since the measurement at that time is not present in other dataset
    df2 = df2.iloc[1:]
    df2.drop(columns=['HourUTC', 'HourDK', 'PriceArea',
                      'SpotPriceEUR'], inplace=True)

    # Merge the two datasets
    dfcombined = pd.merge(df, df2, on='HourDK', how='inner')

    return dfcombined


def feature_engineering(df):
    """ Function to create features from the datetime column """

    df.drop(columns=['HourUTC',
            'MunicipalityNo', 'Branche'], inplace=True)

    # df['HourDK'] = pd.to_datetime(df['HourDK'])

    # # Lag features
    df['ConsumptionkWh_lag1'] = df['ConsumptionkWh'].shift(1)
    df['ConsumptionkWh_lag24'] = df['ConsumptionkWh'].shift(24)
    df['ConsumptionkWh_lag168'] = df['ConsumptionkWh'].shift(168)

    # Lag features for SpotPriceDKK
    df['SpotPriceDKK_lag1'] = df['SpotPriceDKK'].shift(1)
    df['SpotPriceDKK_lag24'] = df['SpotPriceDKK'].shift(24)
    df['SpotPriceDKK_lag168'] = df['SpotPriceDKK'].shift(168)

    # Rolling Average
    df['ConsumptionkWh_roll24'] = df['ConsumptionkWh'].rolling(
        window=24).mean()
    df['ConsumptionkWh_roll168'] = df['ConsumptionkWh'].rolling(
        window=168).mean()

    # Rolling Average for SpotPriceDKK
    df['SpotPriceDKK_roll24'] = df['SpotPriceDKK'].rolling(
        window=24).mean()
    df['SpotPriceDKK_roll168'] = df['SpotPriceDKK'].rolling(
        window=168).mean()

    # Holidays in Denmark from 2021 to 2024 (source: https://publicholidays.dk/)
    holidays = ['2021-01-01', '2021-04-01', '2021-04-02', '2021-04-05', '2021-05-13', '2021-05-21', '2021-06-01', '2021-06-24', '2021-12-24', '2021-12-25', '2021-12-26', '2021-12-31', '2022-01-01', '2022-04-14', '2022-04-15', '2022-04-18', '2022-05-05', '2022-05-13', '2022-05-26', '2022-06-05', '2022-06-24', '2022-12-24', '2022-12-25', '2022-12-26',
                '2022-12-31', '2023-01-01', '2023-03-24', '2023-03-25', '2023-03-26', '2023-04-07', '2023-05-05', '2023-05-13', '2023-05-26', '2023-06-05', '2023-06-24', '2023-12-24', '2023-12-25', '2023-12-26', '2023-12-31', '2024-01-01', '2024-03-28', '2024-03-29', '2024-03-30', '2024-04-05', '2024-05-05', '2024-05-13', '2024-05-26', '2024-06-05', '2024-06-24']
    holidays = pd.to_datetime(holidays)
    df['is_holiday'] = df['HourDK'].dt.date.isin(holidays.date)

    # Weekday and weekend flag
    df['day_of_week'] = df['HourDK'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Hour of the Day (0-23) to sine/cosine transformation
    df['hour_sin'] = np.sin(2 * np.pi * df['HourDK'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['HourDK'].dt.hour / 24)

    # Day of the Week (0-6) to sine/cosine transformation
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Month of the Year (1-12) to sine/cosine transformation
    df['month_sin'] = np.sin(2 * np.pi * df['HourDK'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['HourDK'].dt.month / 12)

    # drop Nan values
    df = df.dropna()

    return df


def sample_data_with_train_window(df, start_date, train_window_size):
    start_date = datetime.strptime(
        start_date, '%Y-%m-%d') - timedelta(hours=train_window_size)
    end_date = df.index[-1]
    return df[(df.index >= start_date) & (df.index <= end_date)]


# def get_next_window(data, train_window_size, validation_window_size, forecast_horizon):
#     return data[:train_window_size], data[train_window_size:validation_window_size+train_window_size], data[train_window_size+validation_window_size:train_window_size + forecast_horizon + validation_window_size]

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


def normalize_dataset(train_df, val_df, test_df):
    # Make explicit copies to avoid modifying slices
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # Apply scaling to features (.loc for Explicit Indexing)
    train_df.loc[:, feature_cols] = scaler.fit_transform(
        train_df[feature_cols])
    val_df.loc[:, feature_cols] = scaler.transform(val_df[feature_cols])
    test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])

    # Apply scaling to the target column
    train_df.loc[:, target_col] = scaler.fit_transform(train_df[[target_col]])
    val_df.loc[:, target_col] = scaler.transform(val_df[[target_col]])
    test_df.loc[:, target_col] = scaler.transform(test_df[[target_col]])

    return train_df, val_df, test_df


class EnergyDataset(Dataset):
    def __init__(self, data, feature_cols, target_col):
        self.features = torch.tensor(
            data[feature_cols].values, dtype=torch.float32)
        self.targets = torch.tensor(
            data[target_col].values, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def create_dataset(train_df, val_df, test_df, batch_size=128):
    train_dataset = EnergyDataset(train_df, feature_cols, target_col)
    val_dataset = EnergyDataset(val_df, feature_cols, target_col)
    test_dataset = EnergyDataset(test_df, feature_cols, target_col)

    # Create dataloaders
    batch_size = batch_size
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


scaler = MinMaxScaler()
feature_cols = ['ConsumptionkWh_lag1', 'ConsumptionkWh_lag24', 'ConsumptionkWh_lag168',
                'ConsumptionkWh_roll24', 'ConsumptionkWh_roll168', 'hour_sin', 'hour_cos',
                'day_sin', 'day_cos', 'month_sin', 'month_cos', 'SpotPriceDKK',
                'SpotPriceDKK_lag1', 'SpotPriceDKK_lag24', 'SpotPriceDKK_lag168',
                'SpotPriceDKK_roll24', 'SpotPriceDKK_roll168']
target_col = 'ConsumptionkWh'


def objective(trial, data_train, data_val, data_test, model_type):
    """
    Objective function for hyperparameter tuning with Optuna.
    """
    # Use GPU or MPS if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("CUDA is USED")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS device")
    else:
        print("Using CPU device")

    # Shared hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2)
    batch_size = 128
    epochs = trial.suggest_categorical('epochs', [100, 200, 500])
    train, val, test = create_dataset(
        data_train, data_val, data_test, batch_size)
    print("Model type: ", model_type)

    if model_type == 'Transformer':
        # Transformer-specific hyperparameters
        valid_combinations = [
            (dm, nh) for dm in range(64, 513, 64)
            for nh in [2, 4, 6, 8] if dm % nh == 0
        ]
        valid_combinations_str = [
            f"{dm},{nh}" for dm, nh in valid_combinations]
        # Suggest from valid combinations (as strings)
        selected_combination_str = trial.suggest_categorical(
            'd_model_nhead', valid_combinations_str)
        # Decode back to integers
        d_model, nhead = map(int, selected_combination_str.split(','))
        num_encoder_layers = trial.suggest_int('num_encoder_layers', 2, 6)
        num_decoder_layers = trial.suggest_int('num_decoder_layers', 2, 6)
        dim_feedforward = trial.suggest_int(
            'dim_feedforward', 128, 512, step=128)
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        print("Hyperparameters: ", d_model, nhead, num_encoder_layers,
              num_decoder_layers, dim_feedforward, dropout)
        model = EnergyTransformer(
            input_size=len(feature_cols),
            d_model=d_model,
            nhead=nhead,
            output_size=1,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
    elif model_type == 'LSTM':
        # LSTM-specific hyperparameters
        hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
        num_layers = trial.suggest_int('num_layers', 1, 4)
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        model = EnergyLSTM(
            input_size=len(feature_cols),
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1,
            dropout=dropout
        )
    elif model_type == 'GRU':
        # GRU-specific hyperparameters
        hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
        num_layers = trial.suggest_int('num_layers', 1, 4)
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        model = EnergyGRU(
            input_size=len(feature_cols),
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1,
            dropout=dropout
        )

    model.to(device)
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    model.train()
    train_loss = 0
    for epoch in range(epochs):  # Dynamic epoch count
        for features, targets in train:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(-1), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation step
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for features, targets in val:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs.squeeze(-1), targets)
                val_loss += loss.item()

    # Return validation loss
    # return val_loss / len(val)
    # Test the model
    model.eval()
    predictions = []
    with torch.no_grad():
        for features, targets in test:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            predictions.extend(outputs.squeeze(-1).cpu().numpy())

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(data_test[target_col], predictions))
    print(f"RMSE: {rmse}")
    return rmse


if __name__ == '__main__':
    mp.set_start_method('spawn')  # Required for parallelism in macOS
    # Read the data
    df = read_data()

    # # Feature Engineering
    df = feature_engineering(df)

    # Set model type as a variable
    model_type = 'LSTM'  # Change to 'LSTM' or 'GRU' as needed

    # Set HourDK as the index
    df = df.set_index('HourDK')

    # Data splitting
    # date_start = '2023-11-01'
    # window_train_size = 24*7*2  # 2 weeks in hours
    # window_val_size = 24*7      # 1 week in hours
    # forecast_horizon = 24       # 1 day in hours

    date_start = '2021-01-01'

    # Training window size = 2 years in hours
    window_train_size = 365 * 24 * 2  # 2 years = 17,520 hours

    # Validation window size = 1 year in hours
    window_val_size = 365 * 24  # 1 year = 8,760 hours

    # Forecast horizon (testing period) = 1 year in hours
    forecast_horizon = 365 * 24  # 1 year = 8,760 hours

    # Sample data
    data = sample_data_with_train_window(df, date_start, window_train_size)
    data_train, data_val, data_test = get_next_window(
        data, window_train_size, window_val_size, forecast_horizon
    )

    # Ensure dataframes are not empty
    if data_train.empty or data_val.empty or data_test.empty:
        raise ValueError("One of the dataframes (train, val, test) is empty.")

    data_trainN, data_valN, data_testN = normalize_dataset(
        data_train, data_val, data_test
    )

    # Optuna study for hyperparameter tuning
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(
        trial, data_trainN, data_valN, data_testN, model_type=model_type), n_trials=100, n_jobs=5)

    # print("Best hyperparameters: ", study.best_params)
    # print("Best value: ", study.best_value)
    # print("Total time for optimization: ",
    #       study.best_trial.duration.resolution)

    # # Save the study
    # study_name = f"{model_type}_study.pkl"

    # warnings.filterwarnings("default")

    # # Save hyperparameters in csv file
    # trial = study.best_trial
    # print(f"Accuracy: {trial.value}")
    # print(f"best params for {model_type}: {trial.params}")

    # # Save the results in CSV
    # df_tuning = pd.DataFrame(columns=['model', 'accuracy', 'params', 'time'])
    # new_row = {'model': model_type, 'accuracy': trial.value,
    #            'params': str(trial.params), 'time': study.best_trial.duration}
    # df_tuning = pd.concat(
    #     [df_tuning, pd.DataFrame([new_row])], ignore_index=True)
    # df_tuning = df_tuning.sort_values(
    #     by=['model', 'accuracy', 'params'], ascending=True).reset_index(drop=True)
    # df_tuning.to_csv('whitebox_tuning.csv', index=False)

    # Display results
    best_trial = study.best_trial
    print("Best hyperparameters: ", best_trial.params)
    print("Best value: ", best_trial.value)
    print("Total time for optimization: ", best_trial.duration)

    # Save the study
    study_name = f"{model_type}_study.pkl"

    # Save hyperparameters in CSV file
    df_tuning = pd.DataFrame(columns=['model', 'accuracy', 'params', 'time'])
    new_row = {
        'model': model_type,
        'accuracy': best_trial.value,
        'params': str(best_trial.params),
        'time': best_trial.duration
    }

    # Fix the FutureWarning by removing empty columns if any
    df_tuning = pd.concat(
        [df_tuning.dropna(how='all'), pd.DataFrame([new_row])], ignore_index=True
    )

    df_tuning = df_tuning.sort_values(
        by=['model', 'accuracy', 'params'], ascending=True
    ).reset_index(drop=True)
    df_tuning.to_csv('whitebox_tuningLong.csv', index=False)

    print(f"Accuracy: {best_trial.value}")
    print(f"Best params for {model_type}: {best_trial.params}")
