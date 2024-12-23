from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


def checking_device():
    if torch.cuda.is_available():
        print("CUDA is USED")
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS device")
    else:
        print("Using CPU device")
    return device


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


def read_data():
    """ Function to read the data """
    df = pd.read_csv('AI-LAB/ConsumptionIndustry.csv', sep=';')
    df2 = pd.read_csv('AI-LAB/ELSpotPrices.csv', sep=';')

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


def splitting_dataset(df):
    """ Function to split the dataset into train, validation and test sets """
    train_start = "2021-01-01"
    train_end = "2023-06-30"
    val_start = "2023-07-01"
    val_end = "2023-11-30"
    test_start = "2023-12-01"
    test_end = "2024-11-10"

    # Filter the dataset based on the date ranges
    train = df[(df['HourDK'] >= train_start) & (df['HourDK'] <= train_end)]
    val = df[(df['HourDK'] >= val_start) & (df['HourDK'] <= val_end)]
    test = df[(df['HourDK'] >= test_start) & (df['HourDK'] <= test_end)]

    return train, val, test


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


def log_tensorboard(writer, model, optimizer, epoch, train_loss, val_loss, train_loader, val_loader, best_val_loss, input_size, device):
    # Log Traning and Validation Loss
    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Loss/Validation", val_loss, epoch)

    # Log Learning Rate
    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar("Learning Rate", current_lr, epoch)

    # Log Best Validation Loss
    writer.add_scalar("Loss/Best Validation", best_val_loss, epoch)

    # Log Gradient Norms
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        writer.add_scalar('Gradients/Norm', total_norm, epoch)

    # Log model weights and gradients
    for name, param in model.named_parameters():
        writer.add_histogram(f'Weights/{name}', param, epoch)
        if param.grad is not None:
            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

    # Log validation predictions vs targets (every 10 epochs)
    if epoch % 10 == 0:
        sample_features, sample_targets = next(iter(val_loader))
        sample_features, sample_targets = sample_features.to(
            device), sample_targets.to(device)
        sample_predictions = model(sample_features.unsqueeze(1)).squeeze()
        writer.add_scalars(
            'Predictions vs Targets',
            {f'Prediction_{i}': pred.item()
             for i, pred in enumerate(sample_predictions[:5])},
            epoch
        )

    # Log validation MAE
    val_predictions = []
    val_targets = []
    with torch.no_grad():
        for batch in val_loader:
            features, targets = batch
            features, targets = features.to(device), targets.to(device)
            predictions = model(features.unsqueeze(1)).squeeze()
            val_predictions.append(predictions.cpu())
            val_targets.append(targets.cpu())

    val_predictions = torch.cat(val_predictions).numpy()
    val_targets = torch.cat(val_targets).numpy()
    rmse = root_mean_squared_error(val_targets, val_predictions)
    writer.add_scalar('Metrics/RMSE', rmse, epoch)

    # Log model graph (first epoch only)
    if epoch == 0:
        dummy_input = torch.randn(1, 10, input_size).to(
            device)  # Example: batch_size=1, seq_len=10
        try:
            scripted_model = torch.jit.script(model)
            writer.add_graph(scripted_model, dummy_input)
        except Exception as e:
            print(f"Error adding scripted model graph: {e}")

    # Log input data distributions (first epoch only)
    if epoch == 0:
        sample_features, _ = next(iter(train_loader))
        writer.add_histogram('Data/Input', sample_features, epoch)

    # Log timing metrics
    import time
    epoch_time = time.time()
    writer.add_scalar('Timing/Epoch Duration', epoch_time, epoch)


def train_transformer(train_loader, val_loader, epochs=100):

    # Create dataloaders
    best_val_loss = float('inf')  # Initialize the best validation loss

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Define device
    device = checking_device()

    # Move model to device
    model.to(device)

    # Setting up Tensorboard
    writer = SummaryWriter('Training_data/runs/energy_transformer')

    # Traning loop
    for epochs in range(epochs):
        # Set the model to training mode
        model.train()

        train_loss = 0

        for batch in train_loader:
            features, targets = batch
            features, targets = features.to(device), targets.to(device)

            # Forward Pass
            optimizer.zero_grad()
            # Add a dummy sequence length dimension
            targets_pred = model(features.unsqueeze(1))
            # Squeeze the output to match the target shape
            loss = criterion(targets_pred.squeeze(), targets)

            # Backward Pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Accumulate the loss for monitoring
            train_loss += loss.item()

        # Calculate the average loss over the entire training data
        train_loss /= len(train_loader)

    # Validation Loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            features, targets = batch
            features, targets = features.to(device), targets.to(device)
            targets_pred = model(features.unsqueeze(1))
            loss = criterion(targets_pred.squeeze(), targets)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    # Log the metrics
    log_tensorboard(
        writer=writer,
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        train_loss=train_loss,
        val_loss=val_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        best_val_loss=best_val_loss,
        input_size=input_size,
        device=device
    )

    print(
        f"Epoch {epochs+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Close the Tensorboard writer
    writer.close()

    # Save the model
    torch.save(model.state_dict(), 'Training_data/Transfomer_long.pth')


def unnormalize_data(data, scaler):
    return scaler.inverse_transform(data.reshape(-1, 1)).flatten()


def perform_predictions(data_loader):
    print("Evaluating Transformer Model")

    # Define device
    device = checking_device()

    # Load the model
    model.load_state_dict(torch.load(
        'Training_data/Transfomer_long.pth', weights_only=True))

    # Set model to eval mode
    model.eval()

    # Colelct the predictions
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            features, targets = batch
            features, targets = features.to(device), targets.to(device)
            targets_pred = model(features.unsqueeze(1))
            predictions.append(targets_pred.squeeze().cpu().numpy())

    return predictions


def evaluate_model(test_set, predictions):

    # Unnormalize the data
    predictions = unnormalize_data(predictions, scaler)

    # Print test set and predictions
    print(f"Test Set: {test_set[:5]}")
    print(f"Predictions: {predictions[:5]}")

    # Calculate the performance metrics
    mae = mean_absolute_error(test_set, predictions)
    mse = mean_squared_error(test_set, predictions)
    rmse = np.sqrt(mse)
    mase = mean_absolute_error(test_set, predictions) / \
        np.mean(np.abs(np.diff(test_set)))
    sMAPE = 100 * np.mean(2 * np.abs(predictions - test_set) /
                          (np.abs(test_set) + np.abs(predictions)))
    mape = 100 * np.mean(np.abs(predictions -
                         test_set) / np.abs(test_set))
    r2 = r2_score(test_set, predictions)

    # Print the performance metrics
    print("Energy Transformer Model Performance:")
    print("-------------------------------------")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    print('\n')
    print(f"Mean Absolute Scaled Error: {mase:.2f}")
    print(f"Symmetric Mean Absolute Percentage Error: {sMAPE:.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}")

    # Save the results to a CSV file
    results = {'mae': mae, 'mse': mse, 'rmse': rmse,
               'r2': r2, 'mase': mase, 'sMAPE': sMAPE, 'mape': mape}
    results_df = pd.DataFrame(results, index=[0])
    results_df.to_csv(
        'Training_data/energy_transformer_results.csv', index=False)


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
        plt.savefig(save_at, dpi=1200)
    plt.show()


# Global variables
scaler = MinMaxScaler()
feature_cols = ['ConsumptionkWh_lag1', 'ConsumptionkWh_lag24', 'ConsumptionkWh_lag168',
                'ConsumptionkWh_roll24', 'ConsumptionkWh_roll168', 'hour_sin', 'hour_cos',
                'day_sin', 'day_cos', 'month_sin', 'month_cos', 'SpotPriceDKK',
                'SpotPriceDKK_lag1', 'SpotPriceDKK_lag24', 'SpotPriceDKK_lag168',
                'SpotPriceDKK_roll24', 'SpotPriceDKK_roll168']
target_col = 'ConsumptionkWh'

# Hyperparameters
input_size = len(feature_cols)
d_model = 128
nhead = 4
output_size = 1
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 512
dropout = 0.1
batch_size = 128
epochs = 100
learning_rate = 0.0001

model = EnergyTransformer(input_size, d_model, nhead, output_size, num_encoder_layers,
                          num_decoder_layers, dim_feedforward, dropout)


if __name__ == '__main__':

    # Read data
    df = read_data()

    # Feature Engineering
    df = feature_engineering(df)

    # Splitting the dataset
    train, val, test = splitting_dataset(df)

    # Normalize the dataset
    train, val, test = normalize_dataset(train, val, test)

    # Create PyTorch Dataset
    train_loader, val_loader, test_loader = create_dataset(
        train, val, test, batch_size=batch_size)

    # Training the Transformer model
    train_transformer(train, val, epochs=epochs)

    # Perform predictions
    predictions = perform_predictions(test=test)

    # Evaluate the Transformer model
    evaluate_model(test['ConsumptionkWh'], predictions)

    # Plot the results
    plot_data(data_train=train['ConsumptionkWh'], data_test=test['ConsumptionkWh'],
              predictions=predictions, save_at='AI-LAB/transformer.png')
