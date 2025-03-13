import json
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Load JSON file to read
with open("chip_layouts.json", "r") as f:
    data = json.load(f)
   
   
df = pd.json_normalize(data) # Flattens data into table


print(df)


# Numerical columns to scale
numerical_features = ["power_metric", "area", "wire_length"]


# Scales data using StandardScaler
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])


# Power metric, area and wire length are standardized
print(df)


# if mean is approx. 0 and std is approx. 1, data was standardized properly
print("\nMean: ", df[numerical_features].mean())
print("\nStandard Dev: ", df[numerical_features].std())


# Split data set into train and test sets
# Define features(X) and labels(Y)
X = df[numerical_features]
y = df["design_name"]


# Peform train - test - split (20% test and 80% train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Display dataset sizes
print("\nTraining set X size:", X_train.shape[0])
print("Test set X size:", X_test.shape[0])


print("\nTraining set y size:", y_train.shape[0])
print("Test set y size:",y_test.shape[0])


# Display X and y datasets
print("\nTraining X Set:\n", X_train)
print("Test Set X:\n", X_test)


print("\nTraining Set y:\n", y_train)
print("Test Set y:\n", y_test)
