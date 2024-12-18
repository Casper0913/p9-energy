import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter


# Correct the file path to the actual location of the CSV file
df = pd.read_csv('Dataset/ConsumptionIndustry.csv', sep=';')

# Load the dataset for colab
# df = pd.read_csv('ConsumptionIndustry.csv', sep=';')


# Convert HourDK to datetime
df['HourDK'] = pd.to_datetime(df['HourDK'])

# Convert ConsumptionkWh to numeric
df['ConsumptionkWh'] = df['ConsumptionkWh'].str.replace(",", ".").astype(float)

df['HourDK'] = pd.to_datetime(df['HourDK'])

# Lag features
df['ConsumptionkWh_lag1'] = df['ConsumptionkWh'].shift(1)
df['ConsumptionkWh_lag24'] = df['ConsumptionkWh'].shift(24)
df['ConsumptionkWh_lag168'] = df['ConsumptionkWh'].shift(168)


# Rolling Average
df['ConsumptionkWh_roll24'] = df['ConsumptionkWh'].rolling(window=24).mean()
df['ConsumptionkWh_roll168'] = df['ConsumptionkWh'].rolling(window=168).mean()

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

train_start = "2021-01-01"
train_end = "2023-06-30"
val_start = "2023-07-01"
val_end = "2023-11-30"
test_start = "2023-12-01"
test_end = "2024-11-10"

# Filter the dataset based on the date ranges
train_df = df[(df['HourDK'] >= train_start) & (df['HourDK'] <= train_end)]
val_df = df[(df['HourDK'] >= val_start) & (df['HourDK'] <= val_end)]
test_df = df[(df['HourDK'] >= test_start) & (df['HourDK'] <= test_end)]


# Normalize the features
scaler = MinMaxScaler()
feature_cols = ['ConsumptionkWh_lag1', 'ConsumptionkWh_lag24', 'ConsumptionkWh_lag168',
                'ConsumptionkWh_roll24', 'ConsumptionkWh_roll168', 'hour_sin', 'hour_cos',
                'day_sin', 'day_cos', 'month_sin', 'month_cos']
target_col = 'ConsumptionkWh'

# Make explicit copies to avoid modifying slices
train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

# Apply scaling to features (.loc for Explicit Indexing)
train_df.loc[:, feature_cols] = scaler.fit_transform(train_df[feature_cols])
val_df.loc[:, feature_cols] = scaler.transform(val_df[feature_cols])
test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])

# Apply scaling to the target column
train_df.loc[:, target_col] = scaler.fit_transform(train_df[[target_col]])
val_df.loc[:, target_col] = scaler.transform(val_df[[target_col]])
test_df.loc[:, target_col] = scaler.transform(test_df[[target_col]])


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


# Create datasets
train_dataset = EnergyDataset(train_df, feature_cols, target_col)
val_dataset = EnergyDataset(val_df, feature_cols, target_col)
test_dataset = EnergyDataset(test_df, feature_cols, target_col)

# Create dataloaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class EnergyTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, output_size, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(EnergyTransformer, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.embedding = nn.Linear(input_size, d_model)  # Embedding layer
        self.positional_encoding = nn.Embedding(
            1000, d_model)  # Positional encoding
        # self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, d_model)) # Positional encoding
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x):
        # Compute the embeddings
        x = self.embedding(x)

        # Generate positional encodings
        position_ids = torch.arange(
            x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        x = x + self.positional_encoding(position_ids)

        # Pass through the Transformer and final output layer
        output = self.transformer(x, x)
        return self.fc_out(output[:, -1, :])


# Model parameters
input_size = len(feature_cols)  # Number of features
d_model = 128  # Embedding dimension
nhead = 4  # Number of attention heads
output_size = 1  # Single output value
num_encoder_layers = 3  # Number of encoder layers
num_decoder_layers = 3  # Number of decoder layers
dim_feedforward = 256  # Feedforward dimension
dropout = 0.1  # Dropout rate

# Initialize the model
model = EnergyTransformer(input_size, d_model, nhead, output_size, num_encoder_layers,
                          num_decoder_layers, dim_feedforward, dropout)


criterion = nn.MSELoss()

# Adam Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


def log_tensorboard(writer, model, optimizer, epoch, train_loss, val_loss, train_loader, val_loader, best_val_loss, patience_counter, input_size, device):
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
    mae = mean_absolute_error(val_targets, val_predictions)
    writer.add_scalar('Metrics/MAE', mae, epoch)

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

    # Log early stopping metrics if patience is exceeded
    if patience_counter > 0:
        writer.add_scalar('EarlyStopping/Patience Counter',
                          patience_counter, epoch)
        writer.add_scalar('EarlyStopping/Best Validation Loss',
                          best_val_loss, epoch)

    # Log timing metrics
    import time
    epoch_time = time.time()
    writer.add_scalar('Timing/Epoch Duration', epoch_time, epoch)


# Check if GPU is available and move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Setting up Tensorboard
writer = SummaryWriter('Training_data/runs/energy_transformer')

# Define early stopping
patience = 50  # Number of epochs to wait before early stopping
best_val_loss = float('inf')  # Initialize the best validation loss
# patience counter =  number of epochs since the last best validation loss
patience_counter = 0

# Training Loop
num_epochs = 1000
for epochs in range(num_epochs):
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
        patience_counter=patience_counter,
        input_size=input_size,
        device=device
    )

    print(
        f"Epoch {epochs+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Early Stopping Check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'Training_data/Transfomer_long.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Best Validation Loss: {best_val_loss}")
            torch.save(model.state_dict(), 'Training_data/Transfomer_long.pth')
            # torch.save(model.state_dict(), '/content/drive/My Drive/energy_transformer.pth')
            break

    if patience_counter >= patience:
        print("Early Stopping")
        torch.save(model.state_dict(), 'Training_data/Transfomer_long.pth')
        # torch.save(model.state_dict(), '/content/drive/My Drive/energy_transformer.pth')
        break


writer.close()
torch.save(model.state_dict(), 'Training_data/Transfomer_long.pth')

# %% [markdown]
# # Evaluate the model

# %%
# Load the model
model.load_state_dict(torch.load(
    'Training_data/Transfomer_long.pth', weights_only=True))

# Set model to eval mode
model.eval()

# Collect predictions and targets
predictions, true_values = [], []

with torch.no_grad():  # Disable gradient tracking
    for batch in test_loader:
        inputs, targets = batch  # Get the inputs and targets
        inputs, targets = inputs.to(device), targets.to(
            device)  # Move to GPU if available

        # Forward pass
        # Add a dummy sequence length dimension
        outputs = model(inputs.unsqueeze(1))

        # Store predictions and true values
        # Move to CPU and convert to NumPy (from PyTorch Tensor)
        predictions.append(outputs.cpu().numpy())
        true_values.append(targets.cpu().numpy())


# Flatten the list of NumPy arrays (from multiple batches) into a single array
# predictions = np.concatenate(predictions, axis=0)
# true_values = np.concatenate(true_values, axis=0)

# Convert predictions and true values to NumPy arrays if they are lists
predictions = np.concatenate(predictions, axis=0)
true_values = np.concatenate(true_values, axis=0)

# Reshape to 2D for inverse_transform
predictions_reshaped = predictions.reshape(-1, 1)
true_values_reshaped = true_values.reshape(-1, 1)

# Use the inverse_transform method to unnormalize
predictions = scaler.inverse_transform(predictions_reshaped)
true_values = scaler.inverse_transform(true_values_reshaped)

# Optionally, flatten the arrays after inverse_transform for easier comparison
predictions = predictions.flatten()
true_values = true_values.flatten()

mae = mean_absolute_error(true_values, predictions)
mse = mean_squared_error(true_values, predictions)
rmse = np.sqrt(mse)
mase = mean_absolute_error(true_values, predictions) / \
    np.mean(np.abs(np.diff(true_values)))
sMAPE = 100 * np.mean(2 * np.abs(predictions - true_values) /
                      (np.abs(true_values) + np.abs(predictions)))
mape = 100 * np.mean(np.abs(predictions - true_values) / np.abs(true_values))
r2 = r2_score(true_values, predictions)

# Print the performance metrics
print("Energy Transformer Model Performance:")
print("Total Traning time: 9min 23sec")
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
results_df.to_csv('Training_data/energy_transformer_results.csv', index=False)

# %% [markdown]
# ### Visualize Model Performance

# %%

# Plot the first week of predictions
plt.figure(figsize=(14, 7))
plt.plot(true_values[:24*7], label='Actual Consumption')
plt.plot(predictions[:24*7], label='Predicted Consumption', linestyle='--')
plt.title("Energy Consumption Prediction for the First Week")
plt.xlabel("Hour")
plt.ylabel("Energy Consumption (kWh)")
plt.legend()
plt.grid(True)
plt.savefig('Training_data/energy_transformer_predictions.png', dpi=900)

# Plot the entire test set
plt.figure(figsize=(14, 7))
plt.plot(true_values, label='Actual', color='blue', alpha=0.6)
plt.plot(predictions, label='Predicted',
         linestyle='--', color='red', alpha=0.6)
plt.title("Predicted vs Actual Energy Consumption")
plt.xlabel("Hour")
plt.ylabel("Energy Consumption (kWh)")
plt.legend()
plt.grid(True)
plt.savefig('Training_data/energy_transformer_predictions_full.png', dpi=900)


# Scatter plot of the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(true_values, predictions, alpha=0.5)
plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)],
         color='red', linewidth=2, label="Ideal (y = x)")
plt.xlabel("Actual Consumption (kWh)")
plt.ylabel("Predicted Consumption (kWh)")
plt.title("Prediction vs Actual")
plt.legend()
plt.grid(True)
plt.savefig('Training_data/energy_transformer_scatter.png', dpi=900)


# Save the predictions and true values to a CSV file
results = pd.DataFrame({'Actual': true_values.flatten(),
                       'Predicted': predictions.flatten()})
results.to_csv('Training_data/energy_transformer_predictions.csv', index=False)
