import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np

# 1. Load and preprocess the data
df = pd.read_csv(
    '/Users/casper/Documents/GitHub/p9-energy/Examples/test-data/sep2024.csv', sep=';')

# Convert the timestamp to datetime and sort the data
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d.%m.%Y %H.%M.%S')
df = df.sort_values('Timestamp')

# Normalize the usage data
scaler = MinMaxScaler()
df['Usage'] = scaler.fit_transform(df[['Usage']])

# 2. Prepare the data for the transformer model


def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    return torch.tensor(np.array(sequences), dtype=torch.float32), torch.tensor(np.array(targets), dtype=torch.float32)


# Parameters
sequence_length = 24  # Use the past 24 hours to predict the next value
train_size = int(0.8 * len(df))

# Split data into training and testing sets
train_data = df['Usage'].values[:train_size]
test_data = df['Usage'].values[train_size:]

train_sequences, train_targets = create_sequences(train_data, sequence_length)
test_sequences, test_targets = create_sequences(test_data, sequence_length)

# 3. Define a simple Transformer model


class TransformerModel(nn.Module):
    def __init__(self, input_dim, dim_model, num_heads, num_encoder_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, dim_model)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, sequence_length, dim_model))
        encoder_layers = nn.TransformerEncoderLayer(
            dim_model, num_heads, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_encoder_layers)
        self.decoder = nn.Linear(dim_model, 1)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1])  # Output the last timestep
        return x


# Hyperparameters
input_dim = 1  # Since we have only 1 feature: usage
dim_model = 64
num_heads = 4
num_encoder_layers = 2
dropout = 0.1
learning_rate = 0.001
epochs = 10

# 4. Instantiate the model, loss function, and optimizer
model = TransformerModel(input_dim, dim_model, num_heads,
                         num_encoder_layers, dropout)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 5. Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# Add input dimension
train_sequences = train_sequences.unsqueeze(-1).to(device)
train_targets = train_targets.unsqueeze(-1).to(device)

start_time = time.time()
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(train_sequences)
    loss = criterion(output, train_targets)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
end_time = time.time()
training_time = end_time - start_time
print(f"Training Time: {training_time:.2f} seconds")

# 6. Evaluate the model
model.eval()
test_sequences = test_sequences.unsqueeze(-1).to(device)
test_targets = test_targets.unsqueeze(-1).to(device)

with torch.no_grad():
    start_time = time.time()
    predictions = model(test_sequences)
    prediction_time = time.time() - start_time
    test_loss = criterion(predictions, test_targets)
    print(f"Test Loss: {test_loss.item():.6f}")
    print(f"Prediction Time: {prediction_time:.2f} seconds")

# Inverse transform the predictions and targets to original scale
predictions = scaler.inverse_transform(predictions.cpu().numpy())
test_targets = scaler.inverse_transform(test_targets.cpu().numpy())

# 7. Log the results (train/test time and loss)
print(f"Training Time: {training_time:.2f} seconds")
print(f"Prediction Time: {prediction_time:.2f} seconds")
