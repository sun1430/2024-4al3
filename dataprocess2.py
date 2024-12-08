import pandas as pd
from functools import reduce

# Load data
file_path = "data/ENV_AEI_SOIL_ERI.csv"
file_path2 = "data/ENV_AEI_SOIL_ERI_WIND.csv"
file_path3 = "data/ENV_AEI_SOIL_ERI_WATER.csv"
file_path4 = "data/ENV_AEI_SOIL_ERI_TILLAGE.csv"
output_file_path = "data/merged_ERI.csv"

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
    
    # melt
    values_long = pd.melt(data, id_vars=['SOIL_LANDSCAPE_ID'], value_vars=value_columns,
                          var_name="Year", value_name=f"Erosion_Value{n}")
    class_long = pd.melt(data, id_vars=['SOIL_LANDSCAPE_ID'], value_vars=class_columns,
                         var_name="Year", value_name=f"Erosion_Class{n}")
    
    # extract year
    values_long['Year'] = values_long['Year'].str.extract(r'(\d{4})').astype(int)
    class_long['Year'] = class_long['Year'].str.extract(r'(\d{4})').astype(int)
    
    return pd.merge(values_long, class_long, on=['SOIL_LANDSCAPE_ID', 'Year'], how='inner')

combined_data1 = process_data(data, 1)
combined_data2 = process_data(data2, 2)
combined_data3 = process_data(data3, 3)
combined_data4 = process_data(data4, 4)

all_data = [combined_data1, combined_data2, combined_data3, combined_data4]
final_data = reduce(lambda left, right: pd.merge(left, right, on=['SOIL_LANDSCAPE_ID', 'Year'], how='inner'), all_data)

# Handle missing values
value_columns = ['Erosion_Value1', 'Erosion_Value2', 'Erosion_Value3', 'Erosion_Value4']
class_columns = ['Erosion_Class1', 'Erosion_Class2', 'Erosion_Class3', 'Erosion_Class4']

for col in value_columns:
    final_data[col] = final_data[col].interpolate().fillna(0)

for col in class_columns:
    final_data[col] = final_data[col].ffill().fillna(0)

# Add original values to Erosion_ValueXN columns
for i, col in enumerate(value_columns, 1):
    final_data[f'Erosion_Value{i}N'] = final_data[col]

# Output the final data
output = final_data.drop(columns=value_columns)

# Split data into training (1981-2016) and testing (2021)
X = output[output['Year'] <= 2016]
Y = output[output['Year'] == 2021]

# Display processed data
print(X.head())
print(Y.head())
print("Missing values per column:")
print(output.isnull().sum())

print("Rows with all missing values:")
print(output.isnull().all(axis=1).sum())

# Save the final output to CSV
output.to_csv(output_file_path, index=False)
