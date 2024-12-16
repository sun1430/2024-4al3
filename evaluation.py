import os
import subprocess
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, accuracy_score,
                             confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support)

# Define paths to the data files and scripts
merged_eri_path = "data/merged_ERI.csv"
predicted_eri_2021_path = "data/predicted_ERI_2021.csv"
dataprocess_script = "dataprocess.py"
model_script = "model.py"
results_folder = "results"

# Check if the results folder exists, if not, create it
if not os.path.exists(results_folder):
    print(f"{results_folder} not found. Creating it.")
    os.makedirs(results_folder)

# Check if merged_ERI exists, if not, run dataprocess.py
if not os.path.exists(merged_eri_path):
    print(f"{merged_eri_path} not found. Running {dataprocess_script} to generate it.")
    subprocess.run([sys.executable, dataprocess_script])

# Check if predicted_ERI_2021 exists, if not, run model.py
if not os.path.exists(predicted_eri_2021_path):
    print(f"{predicted_eri_2021_path} not found. Running {model_script} to generate it.")
    subprocess.run([sys.executable, model_script])

# Load actual and predicted data
actual_data_path = "data/merged_ERI.csv"
predicted_data_path = "data/predicted_ERI_2021.csv"

# Load data
actual_data = pd.read_csv(actual_data_path)
predicted_data = pd.read_csv(predicted_data_path)

# Filter actual 2021 data
target_columns = ['Erosion_Value1N', 'Erosion_Value2N', 'Erosion_Value3N', 'Erosion_Value4N']
actual_2021 = actual_data[actual_data['Year'] == 2021][target_columns].reset_index(drop=True)

# Ensure alignment
if len(actual_2021) != len(predicted_data):
    raise ValueError("Mismatch between actual and predicted data lengths.")

# Evaluate performance
results = {}
for col in target_columns:
    actual = actual_2021[col]
    predicted = predicted_data[col]
    
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    results[col] = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2
    }

# Display evaluation results
results_df = pd.DataFrame(results).T
print("Evaluation Results:")
print(results_df)


# Visualization of predictions vs. actual
plt.figure(figsize=(14, 10))
for i, col in enumerate(target_columns, 1):
    plt.subplot(2, 2, i)
    plt.scatter(range(len(actual_2021[col])), actual_2021[col], label='Actual', color='b')
    plt.scatter(range(len(predicted_data[col])), predicted_data[col], label='Predicted', color='r')
    plt.title(f"Actual vs Predicted for {col}")
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
    min_val, max_val = params['min'], params['max']
    scaled_boundaries = [(b - min_val) / (max_val - min_val) for b in category_boundaries]
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


classification_results = []
for col in target_columns:
    boundaries = category_boundaries_scaled[col]
    actual_2021[f"{col}_Category"] = actual_2021[col].apply(lambda x: map_to_category(x, boundaries))
    predicted_data[f"{col}_Category"] = predicted_data[col].apply(lambda x: map_to_category(x, boundaries))

    y_true = actual_2021[f"{col}_Category"]
    y_pred = predicted_data[f"{col}_Category"]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=["Very Low", "Low", "Moderate", "High", "Very High"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Very Low", "Low", "Moderate", "High", "Very High"])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Confusion Matrix for {col}")
    plt.tight_layout()
    plt.savefig(f"results/confusion_matrix_{col}.png")
    plt.show()
    
    # Compute Precision, Recall, F1 Score, and Accuracy
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)  # Calculate accuracy
    
    # Add results to classification_results
    classification_results.append({
        "Target": col,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Accuracy": accuracy  # Add accuracy to the results
    })

# Convert classification results to DataFrame
classification_df = pd.DataFrame(classification_results).set_index("Target")

# Merge classification metrics with regression metrics
final_results_df = pd.concat([results_df, classification_df], axis=1)

# Display and save final results
print("Final Evaluation Results:")
print(final_results_df)

evaluation_results_path = "results/evaluation_results.csv"
final_results_df.to_csv(evaluation_results_path, index=True)
print(f"Final evaluation results saved to {evaluation_results_path}")
