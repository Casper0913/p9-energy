""" Code for black box tuning of the hyperparameters of different models """

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import optuna  # Hyperparameter optimization
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class EnergyTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, output_size, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(EnergyTransformer, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.embedding = nn.Linear(input_size, d_model)  # Embedding layer
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, 1000, d_model))  # Replace nn.Embedding
        # self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, d_model)) # Positional encoding
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        output = self.transformer(x, x)
        return self.fc_out(output[:, -1, :])


class EnergyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(EnergyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        if len(lstm_out.shape) == 2:
            # No squeeze for single sequence
            return self.fc_out(lstm_out[:, -1])
        else:
            # Squeeze the batch dimension if present
            # Squeeze the extra dimension
            return self.fc_out(lstm_out[:, -1, :].squeeze(1))


class EnergyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
        super(EnergyGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.fc_out(gru_out[:, -1, :])


def read_data():
    """ Function to read the data """
    df = pd.read_csv(
        '/Users/casper/Documents/GitHub/p9-energy/Dataset/ConsumptionIndustry.csv', sep=';')

    # Load the dataset for colab
    # df = pd.read_csv('ConsumptionIndustry.csv', sep=';')

    # Convert HourDK to datetime
    df['HourDK'] = pd.to_datetime(df['HourDK'])

    # Convert ConsumptionkWh to numeric
    df['ConsumptionkWh'] = df['ConsumptionkWh'].str.replace(
        ",", ".").astype(float)

    return df


def feature_engineering(df):
    """ Function to create features from the datetime column """
    # df['HourDK'] = pd.to_datetime(df['HourDK'])

    # # Lag features
    df['ConsumptionkWh_lag1'] = df['ConsumptionkWh'].shift(1)
    df['ConsumptionkWh_lag24'] = df['ConsumptionkWh'].shift(24)
    df['ConsumptionkWh_lag168'] = df['ConsumptionkWh'].shift(168)

    # Rolling Average
    df['ConsumptionkWh_roll24'] = df['ConsumptionkWh'].rolling(
        window=24).mean()
    df['ConsumptionkWh_roll168'] = df['ConsumptionkWh'].rolling(
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


def get_next_window(data, train_window_size, validation_window_size, forecast_horizon):
    return data[:train_window_size], data[train_window_size:validation_window_size+train_window_size], data[train_window_size+validation_window_size:train_window_size + forecast_horizon + validation_window_size]


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
                'day_sin', 'day_cos', 'month_sin', 'month_cos']
target_col = 'ConsumptionkWh'


def objective(trial, data_train, data_val, model_type):
    """
    Objective function for hyperparameter tuning with Optuna.
    """
    # Select model type (static or trial-specific for flexibility)
    # model_type = trial.suggest_categorical(
    #     'model_type', ['Transformer', 'LSTM', 'GRU'])

    # Shared hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', 64, 128, 256)
    epochs = trial.suggest_categorical('epochs', 100, 200, 500)

    print("Trial parameters: ", trial.params)

    train, val, train = create_dataset(
        data_train, data_val, data_val, batch_size)

    print("Model type: ", model_type)
    print("Train data: ", train)
    print("Validation data: ", val)

    if model_type == 'Transformer':

        # Transformer-specific hyperparameters
        # d_model multiples of 64
        d_model = trial.suggest_int('d_model', 64, 512)
        nhead = trial.suggest_int('nhead', 2, 8)
        num_encoder_layers = trial.suggest_int('num_encoder_layers', 2, 6)
        num_decoder_layers = trial.suggest_int('num_decoder_layers', 2, 6)
        dim_feedforward = trial.suggest_int(
            'dim_feedforward', 128, 512, step=128)
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)

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

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    model.train()
    train_loss = 0
    for epoch in range(epochs):  # Dynamic epoch count
        for features, targets in train:
            optimizer.zero_grad()
            outputs = model(features)
            print("Fejler den her?")
            loss = criterion(outputs.squeeze(-1), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

    # Validation step
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for features, targets in val:
            outputs = model(features)
            loss = criterion(outputs.squeeze(-1), targets)
            val_loss += loss.item()

    # Return validation loss
    return val_loss / len(val)


if __name__ == '__main__':
    # Read the data
    df = read_data()

    # Feature Engineering
    df = feature_engineering(df)

    # Set model type as a variable
    model_type = 'Transformer'  # Change to 'LSTM' or 'GRU' as needed

    # Set HourDK as the index
    df = df.set_index('HourDK')

    # Data splitting
    date_start = '2023-11-01'
    window_train_size = 24*7*2  # 2 weeks in hours
    window_val_size = 24*7      # 1 week in hours
    forecast_horizon = 24       # 1 day in hours

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
        trial, data_trainN, data_valN, model_type=model_type), n_trials=50)

    print("Best hyperparameters: ", study.best_params)
    print("Best value: ", study.best_value)
    print("Best trial: ", study.best_trial)
    print("Best trial parameters: ", study.best_trial.params)
