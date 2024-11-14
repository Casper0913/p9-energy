import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter

# Load the data and parse the 'Timestamp' column explicitly
data = pd.read_csv(
    '/Users/casper/Documents/GitHub/p9-energy/Examples/test-data/maj2023_2024.csv', sep=';')

# Convert 'Timestamp' column to datetime format
data['Timestamp'] = pd.to_datetime(
    data['Timestamp'], format='%d.%m.%Y %H.%M.%S')

# Separate the training and evaluation datasets
train_data = data[data['Timestamp'].dt.year == 2023]
eval_data = data[data['Timestamp'].dt.year == 2024]

# Convert usage to numpy arrays
train_usage = train_data['Usage'].values.reshape(-1, 1)
eval_usage = eval_data['Usage'].values.reshape(-1, 1)

# Normalize the usage
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_usage)
eval_scaled = scaler.transform(eval_usage)

# Prepare sequences for GRU


def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append((seq, label))
    return np.array(sequences, dtype=object)


seq_length = 48
train_sequences = create_sequences(train_scaled, seq_length)
eval_sequences = create_sequences(eval_scaled, seq_length)

# Convert to NumPy arrays and then tensors
train_x, train_y = zip(*train_sequences)
train_x = np.array(train_x)
train_y = np.array(train_y)
train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)

train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Convert eval data
eval_x, eval_y = zip(*eval_sequences)
eval_x = np.array(eval_x)
eval_y = np.array(eval_y)
eval_x = torch.tensor(eval_x, dtype=torch.float32)
eval_y = torch.tensor(eval_y, dtype=torch.float32)


class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=200, num_layers=5):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])  # Take the output of the last GRU cell
        return out


device = torch.device('mps')  # Use MPS for Mac M1
model = GRUModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter(log_dir='runs/gru_experiment')

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    start_time = time.time()

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time

    # Log the training loss to TensorBoard
    writer.add_scalar('Loss/train', loss.item(), epoch)

    print(
        f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Time: {train_time:.2f}s')

# Close the writer after training
writer.close()

# Evaluate on May 2024
model.eval()
with torch.no_grad():
    eval_x = eval_x.to(device)
    predictions = model(eval_x).cpu()

# Inverse scale predictions
predictions = scaler.inverse_transform(predictions.numpy())

# Compare predictions with actual usage
actual_usage = scaler.inverse_transform(eval_y.numpy())

# Convert actual usage to tensor
actual_tensor = torch.tensor(actual_usage, dtype=torch.float32)

# Calculate MSE using PyTorch
criterion = nn.MSELoss()
eval_loss = criterion(torch.tensor(
    predictions, dtype=torch.float32), actual_tensor)

print(f'Evaluation MSE Loss: {eval_loss.item():.4f}')

# Optional: If you want to store the MSE value for later use
mse_value = eval_loss.item()

date_range = pd.date_range(
    start='01.05.2024', periods=len(actual_usage), freq='h')

# Plot the actual vs predicted usage
plt.figure(figsize=(10, 6))
plt.plot(date_range, actual_usage, label='Actual Usage', color='blue')
plt.plot(date_range, predictions, label='Predicted Usage',
         color='orange', linestyle='--')
plt.title('Predicted vs Actual Electricity Usage for May 2024')
plt.xlabel('Date')
plt.ylabel('Usage (kWh)')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
