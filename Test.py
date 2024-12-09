import os
import subprocess
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, accuracy_score,
                             confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support)

#Paths to files
model_save_path = "rnn_model.pickle"
test_features_path = "data/test_features.npy"

#Function for loading model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Use the last output of the RNN
        return out

def load_model_from_pickle(file_path, model_class, device):
    checkpoint = torch.load(file_path, map_location=device, weights_only=False)
    model = model_class(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        output_size=checkpoint['output_size'],
        num_layers=checkpoint['num_layers']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

#Load test features
test_features = np.load(test_features_path)
print(f"Loaded Test Features Shape: {test_features.shape}")

#Use cuda if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Load the trained model
loaded_model, loaded_optimizer = load_model_from_pickle(model_save_path, SimpleRNN, device)
print(f"Model Loaded From {model_save_path}")
loaded_model.eval()

#Process test features to create sequences
def prepare_test_sequences(features, input_years, target_year):
    sequences = []
    targets = []
    unique_ids = np.unique(features[:, 0])

    for soil_id in unique_ids:
        #Filter rows for ID and sort by year
        soil_data = features[features[:, 0] == soil_id]
        soil_data = soil_data[np.argsort(soil_data[:, 1])]

        #Extract features and years
        years = soil_data[:, 1]
        feature_values = soil_data[:, 2:]

        if all(year in years for year in input_years) and target_year in years:
            #Create input sequence and target
            input_sequence = np.array([feature_values[years == year][0] for year in input_years])
            target_value = feature_values[years == target_year][0]

            sequences.append(torch.tensor(input_sequence, dtype=torch.float32))
            targets.append(torch.tensor(target_value, dtype=torch.float32))

    return sequences, targets

input_years = [2006, 2011, 2016]
target_year = 2021

#Prepare test sequences and targets
test_sequences, test_targets = prepare_test_sequences(test_features, input_years, target_year)

#Dataset for the test data
class TestDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

#Create the dataset and dataloader
test_dataset = TestDataset(test_sequences, test_targets)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

criterion = nn.MSELoss()

#Evaluate the model
test_loss = 0
predictions = []
ground_truth = []

with torch.no_grad():
    for batch_features, batch_targets in test_loader:
        batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
        outputs = loaded_model(batch_features)
        predictions.append(outputs.cpu().numpy())
        ground_truth.append(batch_targets.cpu().numpy())
        loss = criterion(outputs, batch_targets)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f"Test Loss (MSE): {test_loss:.4f}")

# Convert predictions and ground truth to NumPy arrays
predictions = np.vstack(predictions)
ground_truth = np.vstack(ground_truth)

#Optionally save predictions for analysis
np.save("predictions.npy", predictions)
