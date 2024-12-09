import os
import subprocess
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from functools import reduce

# Load data
file_path = "data/ENV_AEI_SOIL_ERI.csv"
file_path2 = "data/ENV_AEI_SOIL_ERI_WIND.csv"
file_path3 = "data/ENV_AEI_SOIL_ERI_WATER.csv"
file_path4 = "data/ENV_AEI_SOIL_ERI_TILLAGE.csv"

print("Running Data PreProcess: -----")
with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
    data = pd.read_csv(f)
with open(file_path2, 'r', encoding='utf-8', errors='replace') as f2:
    data2 = pd.read_csv(f2)
with open(file_path3, 'r', encoding='utf-8', errors='replace') as f3:
    data3 = pd.read_csv(f3)
with open(file_path4, 'r', encoding='utf-8', errors='replace') as f4:
    data4 = pd.read_csv(f4)

def process_data(data, n):
    text_columns = [col for idx, col in enumerate(data.columns) if (idx % 4 == 3 or idx % 4 == 0) and idx != 0]
    data = data.drop(columns=text_columns)

    value_columns = [col for col in data.columns if '_VAL' in col and '_CHG' not in col]
    class_columns = [col for col in data.columns if '_CLASS' in col and '_CHG' not in col]
    
    #melt
    values_long = pd.melt(data, id_vars=['SOIL_LANDSCAPE_ID'], value_vars=value_columns,
                        var_name="Year", value_name=f"Erosion_Value{n}")
    class_long = pd.melt(data, id_vars=['SOIL_LANDSCAPE_ID'], value_vars=class_columns,
                        var_name="Year", value_name=f"Erosion_Class{n}")
    
    #extract year
    values_long['Year'] = values_long['Year'].str.extract(r'(\d{4})').astype(int)
    class_long['Year'] = class_long['Year'].str.extract(r'(\d{4})').astype(int)
    
    return pd.merge(values_long, class_long, on=['SOIL_LANDSCAPE_ID', 'Year'], how='inner')

combined_data1 = process_data(data, 1)
combined_data2 = process_data(data2, 2)
combined_data3 = process_data(data3, 3)
combined_data4 = process_data(data4, 4)

all_data = [combined_data1, combined_data2, combined_data3, combined_data4]
final_data = reduce(lambda left, right: pd.merge(left, right, on=['SOIL_LANDSCAPE_ID', 'Year'], how='inner'), all_data)

#This is used for interpolating missing value for some years
def interpolate_missing_values(data, group_col, value_cols):
    data = data.copy()
    for col in value_cols:
        # Replace zeros with NaN for meaningful interpolation
        data[col] = data[col].replace(0, pd.NA)
        # Group by ID and interpolate missing values
        data[col] = (
            data.groupby(group_col)[col]
            .apply(lambda x: x.interpolate(method='linear').bfill().ffill())
            .reset_index(level=0, drop=True)
        )
    data[value_cols] = data[value_cols].fillna(0)
    return data

#Handling missing values
value_columns = ['Erosion_Value1', 'Erosion_Value2', 'Erosion_Value3', 'Erosion_Value4']
class_columns = ['Erosion_Class1', 'Erosion_Class2', 'Erosion_Class3', 'Erosion_Class4']

final_data = interpolate_missing_values(final_data, 'SOIL_LANDSCAPE_ID', value_columns)

bins = [0, 6, 11, 22, 33, float('inf')]
labels = [1, 2, 3, 4, 5]

for value_col, class_col in zip(value_columns, class_columns):
    final_data[class_col] = pd.cut(final_data[value_col], bins=bins, labels=labels, right=False).astype('Int64')
    final_data.loc[final_data[value_col] == 0, class_col] = 0

    final_data[class_col] = final_data[class_col].astype(int)

'''
# Normalize Erosion_Value
scaler = MinMaxScaler()
for col in value_columns:
    final_data[col] = final_data[col].astype(float).fillna(0)
    final_data[f'{col}N'] = scaler.fit_transform(final_data[[col]])
'''

train_data = final_data[final_data['Year'] <= 2016]
test_data = final_data[final_data['Year'] >= 2006]

#Extract features and save as .npy
train_features = train_data[['SOIL_LANDSCAPE_ID', 'Year'] + value_columns].values
test_features = test_data[['SOIL_LANDSCAPE_ID', 'Year'] + value_columns].values

np.save("data/train_features.npy", train_features)
np.save("data/test_features.npy", test_features)

#Output preprocessed data
final_data = final_data.drop(columns = value_columns)

#Save final processed data as CSV
final_data.to_csv("data/processed_data.csv", index=False)

print("Data preprocess complete -------")

#-------------------------------------------------------

# Load features.npy file
train_features = np.load("data/train_features.npy")
test_features = np.load("data/test_features.npy")

#use cuda if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#This function is used to create time-series sequences, so that RNN is meaningful
def create_sequences(features, window_size):
    sequences = []
    targets = []

    #group by ID
    unique_ids = np.unique(features[:, 0])
    for soil_id in unique_ids:
        #Filter rows belonging to the same SOIL_LANDSCAPE_ID
        soil_data = features[features[:, 0] == soil_id]
        #Sort by Year
        soil_data = soil_data[np.argsort(soil_data[:, 1])]  

        #Separate feature columns and target column
        feature_data = soil_data[:, 2:]

        for i in range(len(feature_data) - window_size):
            sequences.append(torch.tensor(feature_data[i:i + window_size], dtype=torch.float32))
            targets.append(torch.tensor(feature_data[i + window_size], dtype=torch.float32))

    return sequences, targets

window_size = 3

train_sequences, train_targets = create_sequences(train_features, window_size)

#make it PyTorch Dataset
class ErosionDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


#training datasets
train_dataset = ErosionDataset(train_sequences, train_targets)

#Build RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

#Initializing
input_size = train_sequences[0].shape[1]
hidden_size = 64
output_size = train_targets[0].shape[0] 
num_layers = 2
model = SimpleRNN(input_size, hidden_size, output_size, num_layers).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

'''
#Perform K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(train_dataset)))):
    print(f"Fold {fold + 1}")

    #split data
    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    val_subset = torch.utils.data.Subset(train_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    #train
    model.train()
    for epoch in range(20):
        train_loss = 0
        for batch_features, batch_targets in train_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

    #validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_features, batch_targets in val_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)

            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f"Fold {fold + 1}, Validation Loss: {val_loss:.4f}")
'''

#Perform Time Series Split
tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(tscv.split(range(len(train_dataset)))):
    print(f"Fold {fold + 1}")

    #split data
    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    val_subset = torch.utils.data.Subset(train_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    #train
    model.train()
    for epoch in range(20):
        train_loss = 0
        for batch_features, batch_targets in train_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

    #validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_features, batch_targets in val_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)

            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f"Fold {fold + 1}, Validation Loss: {val_loss:.4f}")

#Function to save model as pickle
model_save_path = "rnn_model.pickle"

def save_model(model, optimizer, file_path, input_size, hidden_size, output_size, num_layers):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_size': output_size,
        'num_layers': num_layers
    }, file_path)
    print(f"Model Saved To {file_path}")

if not os.path.exists(model_save_path):
    save_model(model, optimizer, model_save_path, input_size, hidden_size, output_size, num_layers)

'''
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

loaded_model, loaded_optimizer = load_model_from_pickle(model_save_path, SimpleRNN, device)
print(f"Model Loaded From {model_save_path}")

test_sequences, test_targets = create_sequences(data, "SOIL_LANDSCAPE_ID", feature_columns, target_columns, training_years, target_year)
test_dataset = ErosionDataset(test_sequences, test_targets)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

loaded_model.eval()
test_loss = 0
predictions = []
with torch.no_grad():
    for batch_features, batch_targets in test_loader:
        batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
        outputs = loaded_model(batch_features)
        predictions.append(outputs.cpu().numpy())
        loss = criterion(outputs, batch_targets)
        test_loss += loss.item()
test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")


# De-normalize predictions
scalers = {}
for col in feature_columns:
    scalers[col] = MinMaxScaler()
    scalers[col].fit(data[data['Year'] <= 2016][[col]])

def denormalize(predictions, scalers, columns):
    de_normalized = []
    for i, col in enumerate(columns):
        de_normalized.append(scalers[col].inverse_transform(predictions[:, i].reshape(-1, 1)))
    return np.hstack(de_normalized)

predictions = np.array(predictions).squeeze()
de_normalized_predictions = denormalize(predictions, scalers, feature_columns)

test_targets = np.array(test_targets)
mae = mean_absolute_error(test_targets, de_normalized_predictions)
rmse = np.sqrt(mean_squared_error(test_targets, de_normalized_predictions))
print(f"De-Normalized MAE: {mae:.4f}")
print(f"De-Normalized RMSE: {rmse:.4f}")


#Make a DataFrame for review
predictions = np.array(predictions).squeeze()
predicted_df = pd.DataFrame(predictions, columns=target_columns)
predicted_df.to_csv("data/predicted_ERI_2021.csv", index=False)
'''