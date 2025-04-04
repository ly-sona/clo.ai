import pandas as pd
import numpy as np
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------------------
# 1. Define Feature Extraction Process
# -------------------------------------------

def extract_statistical_features(df):
    """
    Extract statistical summaries for each layout’s 2D maps.
    This includes mean, std, min, max, skewness, and kurtosis for each feature.
    """
    statistical_features = df.describe().transpose()[['mean', 'std', 'min', 'max', '25%', '75%']].reset_index()
    statistical_features['skew'] = df.skew()
    statistical_features['kurtosis'] = df.kurtosis()

    return statistical_features

# -------------------------------------------
# 2. Data Ingestion and Feature Engineering
# -------------------------------------------

# Load your JSON file (chip layouts)
with open('chip_layouts.json', 'r') as f:
    data = json.load(f)

# Normalize the JSON data into a flat table
df = pd.json_normalize(data)

# Feature extraction
statistical_features = extract_statistical_features(df)

# -------------------------------------------
# 3. Data Preprocessing (Standardization)
# -------------------------------------------

# Normalize the features using StandardScaler
scaler = StandardScaler()
statistical_features_scaled = scaler.fit_transform(statistical_features)

# Prepare training data (X) and labels (y)
X = statistical_features_scaled
y = df['power_metric']  # Assuming power_metric is the label column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------
# 4. Train and Evaluate XGBoost Model
# -------------------------------------------

# Define and train the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared: {r2}")

# -------------------------------------------
# 5. Feature Importance Visualization
# -------------------------------------------

# Display feature importance using matplotlib
xgb.plot_importance(model)
plt.show()

# -------------------------------------------
# 6. Define the Threshold for Power Consumption
# -------------------------------------------

# Set a threshold for power consumption (e.g., ~0.5mW)
threshold = 0.5  # Example in mW

# Identify layouts that meet the power consumption threshold
optimized_layouts = y_pred[y_pred < threshold]
print(f"Number of optimized layouts: {len(optimized_layouts)}")

# -------------------------------------------
# 7. Save the Trained Model (optional)
# -------------------------------------------

# Save the model to disk for future use
model.save_model('xgboost_power_model.json')

# -------------------------------------------
# 8. Documenting Threshold Rationale
# -------------------------------------------

# Documentation of Threshold Rationale:
threshold_rationale = """
The threshold for power consumption (~0.5mW) is selected to ensure the chip layout optimizations achieve acceptable power efficiency 
without compromising design constraints. A lower threshold can be applied in power-sensitive designs, while the threshold can 
also be normalized for consistency across varying power units. This threshold integrates with the genetic algorithm to optimize 
layout placements until power consumption meets the target, resulting in more efficient VLSI designs.
"""

print(threshold_rationale)

# -------------------------------------------
# 9. Save the Processed Data (Optional)
# -------------------------------------------

# Save the processed statistical features as a DataFrame and also as a JSON file
processed_df = pd.DataFrame(statistical_features_scaled, columns=statistical_features.columns)
processed_df.to_json('processed_chip_layout_features.json', orient='records', lines=True)
