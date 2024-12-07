
# Save the cleaned dataset to a new file
output_path = "data/ERI_clean.csv"
data.to_csv(output_path, index=False)
print(f"Cleaned data saved to {output_path}")

import pandas as pd

# Load data
file_path = "data/ENV_AEI_SOIL_ERI.csv"
data = pd.read_csv(file_path)

# Drop SOIL_LANDSCAPE_ID and text columns
text_columns = [col for col in data.columns if '_CLASS_EN' in col or '_CLASS_FR' in col]
data = data.drop( text_columns)

# Convert wide format to long format
value_columns = [col for col in data.columns if '_VAL' in col]
class_columns = [col for col in data.columns if '_CLASS' in col and '_CHG' not in col]

# Melting the data for numerical values
values_long = pd.melt(data, id_vars=[], value_vars=value_columns, 
                      var_name="Year", value_name="Erosion_Value")
# Melting the data for classification
class_long = pd.melt(data, id_vars=[], value_vars=class_columns, 
                     var_name="Year", value_name="Erosion_Class")

# Combine value and class data
values_long['Year'] = values_long['Year'].str.extract(r'(\d{4})').astype(int)
class_long['Year'] = class_long['Year'].str.extract(r'(\d{4})').astype(int)
combined_data = pd.merge(values_long, class_long, on="Year")

# Handle missing values
combined_data['Erosion_Value'] = combined_data['Erosion_Value'].interpolate()  # Linear interpolation
combined_data['Erosion_Class'] = combined_data['Erosion_Class'].fillna(method='ffill')  # Forward fill

# Normalize numerical features (Erosion_Value)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
combined_data['Erosion_Value_Normalized'] = scaler.fit_transform(
    combined_data[['Erosion_Value']]
)

# Split data into training (1981-2016) and testing (2021)
train_data = combined_data[combined_data['Year'] <= 2016]
test_data = combined_data[combined_data['Year'] == 2021]

# Display processed data
print(train_data.head())
print(test_data.head())
