import os
import subprocess
import sys
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, accuracy_score,
                             confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support)
import torch.nn.functional as F

#Paths to files
model_save_path = "rnn_model.pickle"
test_features_path = "data/test_features.npy"

#Function for loading model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.2):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        #out = F.relu(self.fc(out))
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
loaded_model, loaded_optimizer = load_model_from_pickle(model_save_path, SimpleLSTM, device)
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


# ------------------------------ Evaluation ------------------------------
if not os.path.exists("results"):
    os.makedirs("results")

# Load test features and predictions
test_features = np.load("data/test_features.npy")
predictions = np.load("predictions.npy")

print(f"Loaded Test Features Shape: {test_features.shape}")
print(f"Loaded Predictions Shape: {predictions.shape}")

# Filter test features for 2021 data
target_year = 2021
target_columns = [2, 3, 4, 5]
#uncomment this to eliminate the eri of wind since it missing more than 80% of values
#target_columns = [2, 3, 4]

# Extract features for 2021
test_2021 = test_features[test_features[:, 1] == target_year]
print(f"Test Data for 2021: {test_2021.shape}")

# Actual targets from test_features (assuming erosion values are in certain columns)
actual_targets = test_2021[:, target_columns]
print(f"Actual Targets for 2021: {actual_targets.shape}")

# Ensure predictions and actual values are aligned
assert predictions.shape[0] == actual_targets.shape[0], "Mismatch between predictions and actual targets"


# Calculate MSE, MAE, R2 for each target
results = {}
for i, target_column in enumerate(target_columns):
    actual = actual_targets[:, i]
    predicted = predictions[:, i]

    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    results[f"Target {i+1}"] = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2
    }

# Display regression results
results_df = pd.DataFrame(results).T

# Visualization of predictions vs. actual
plt.figure(figsize=(14, 10))
for i in range(len(target_columns)):
    actual = actual_targets[:, i]
    predicted = predictions[:, i]
    plt.subplot(2, 2, i+1)
    plt.scatter(range(len(actual)), actual, label='Actual', color='b')
    plt.scatter(range(len(predicted)), predicted, label='Predicted', color='r')
    plt.title(f"Actual vs Predicted for Target {i+1}")
    plt.xlabel("Sample Index")
    plt.ylabel("Erosion Value")
    plt.legend()
    plt.tight_layout()
plt.savefig("results/actual_vs_predicted.png")
plt.show()

# Load scaler parameters
with open('data/scaler_params.json', 'r') as f:
    scaler_params = json.load(f)

# Define category boundaries
category_boundaries = [6, 11, 22, 33]

# Scale boundaries for each column using loaded parameters
category_boundaries_scaled = {}
for col, params in scaler_params.items():
    # Calculate scaled boundaries
    mean, std = params['mean'], params['std']
    scaled_boundaries = [(b - mean) / std for b in category_boundaries]
    category_boundaries_scaled[col] = scaled_boundaries

# Map normalized values to categories
def map_to_category(value, boundaries):
    if value < boundaries[0]:
        return "Very Low"
    elif boundaries[0] <= value <= boundaries[1]:
        return "Low"
    elif boundaries[1] < value <= boundaries[2]:
        return "Moderate"
    elif boundaries[2] < value <= boundaries[3]:
        return "High"
    else:
        return "Very High"


# Convert actual and predicted values to categories
classification_results = []
for i, target_column in enumerate(target_columns):
    actual_2021_category = [map_to_category(value,category_boundaries_scaled[f'Erosion_Value{i+1}N']) for value in actual_targets[:, i]]
    predicted_category = [map_to_category(value,category_boundaries_scaled[f'Erosion_Value{i+1}N']) for value in predictions[:, i]]

    # Compute confusion matrix
    cm = confusion_matrix(actual_2021_category, predicted_category, labels=["Very Low", "Low", "Moderate", "High", "Very High"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Very Low", "Low", "Moderate", "High", "Very High"])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Confusion Matrix for Target {i+1}")
    plt.tight_layout()
    plt.savefig(f"results/confusion_matrix_target_{i+1}.png")
    plt.show()

    # Compute Precision, Recall, F1 Score, and Accuracy
    precision, recall, f1, _ = precision_recall_fscore_support(actual_2021_category, predicted_category, average='weighted')
    accuracy = accuracy_score(actual_2021_category, predicted_category)

    # Store results
    classification_results.append({
        "Target": f"Target {i+1}",
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Accuracy": accuracy
    })

# Convert classification results to DataFrame
classification_df = pd.DataFrame(classification_results).set_index("Target")

# Merge regression and classification results
final_results_df = pd.concat([results_df, classification_df], axis=1)

# Display and save final results
print("Final Evaluation Results:")
print(final_results_df)

# Save final evaluation results to CSV
final_results_df.to_csv("results/evaluation_results.csv", index=True)
print("Final evaluation results saved to results/evaluation_results.csv")
