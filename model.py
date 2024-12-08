import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
data_path = "data/merged_ERI.csv"
data = pd.read_csv(data_path)

#This function is used to create time-series sequences, so that RNN is meaningful
def create_sequences(data, group_col, feature_cols, target_cols, years, target_year):
    sequences = []
    targets = []

    #group by ID
    grouped = data.groupby(group_col)
    for gid, group in grouped:
        group = group.sort_values(by="Year")

        feature_data = group[group["Year"].isin(years)][feature_cols].values
        target_data = group[group["Year"] == target_year][target_cols].values

        if len(feature_data) == len(years) and len(target_data) == 1:
            sequences.append(torch.tensor(feature_data, dtype=torch.float32))
            targets.append(torch.tensor(target_data[0], dtype=torch.float32))

    return sequences, targets

#use 1981 - 2016 as training dataset, 2021 as testing dataset
training_years = [1981, 1986, 1991, 1996, 2001, 2006, 2011, 2016]
target_year = 2021

#the features will be all four erosion value from 1981 - 2016, and target will be the four erosion value on 2021
feature_columns = ['Erosion_Value1N', 'Erosion_Value2N', 'Erosion_Value3N', 'Erosion_Value4N']
target_columns = ['Erosion_Value1N', 'Erosion_Value2N', 'Erosion_Value3N', 'Erosion_Value4N']

#make training and testing to sequences
train_sequences, train_targets = create_sequences(data, "SOIL_LANDSCAPE_ID", feature_columns, target_columns, training_years, target_year)

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
input_size = len(feature_columns)
hidden_size = 64
output_size = len(target_columns)
num_layers = 2
model = SimpleRNN(input_size, hidden_size, output_size, num_layers).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


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

#Test
test_sequences, test_targets = create_sequences(data, "SOIL_LANDSCAPE_ID", feature_columns, target_columns, training_years, target_year)
test_dataset = ErosionDataset(test_sequences, test_targets)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model.eval()
test_loss = 0
predictions = []
with torch.no_grad():
    for batch_features, batch_targets in test_loader:
        batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
        outputs = model(batch_features)
        predictions.append(outputs.cpu().numpy())
        loss = criterion(outputs, batch_targets)
        test_loss += loss.item()
test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")

#Make a DataFrame for review
predictions = np.array(predictions).squeeze()
predicted_df = pd.DataFrame(predictions, columns=target_columns)
predicted_df.to_csv("data/predicted_ERI_2021.csv", index=False)