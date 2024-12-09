import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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
with open(file_path4, 'r',encoding='utf-8', errors='replace') as f4:
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
        # Ensure all missing values are handled
    data[value_cols] = data[value_cols].fillna(0)
    return data

# Handle missing values
value_columns = ['Erosion_Value1', 'Erosion_Value2', 'Erosion_Value3', 'Erosion_Value4']
class_columns = ['Erosion_Class1', 'Erosion_Class2', 'Erosion_Class3', 'Erosion_Class4']

final_data = interpolate_missing_values(final_data, 'SOIL_LANDSCAPE_ID', value_columns)

bins = [0, 6, 11, 22, 33, float('inf')]
labels = [1, 2, 3, 4, 5]

for value_col, class_col in zip(value_columns, class_columns):
    final_data[class_col] = pd.cut(final_data[value_col], bins=bins, labels=labels, right=False).astype('Int64')
    final_data.loc[final_data[value_col] == 0, class_col] = 0

    final_data[class_col] = final_data[class_col].astype(int)


# Normalize Erosion_Value
scaler = MinMaxScaler()
for col in value_columns:
    final_data[col] = final_data[col].astype(float).fillna(0)
    final_data[f'{col}N'] = scaler.fit_transform(final_data[[col]])

output = final_data.drop(columns = value_columns)

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
output.to_csv(output_file_path, index=False)
