import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

class DataCleaner:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.df = None
        self.outliers_info = {}
        self.feature_columns = [str(i) for i in range(256)]  # For features 0-255

    def load_json_data(self):
        """Load JSON data with 256 numerical features"""
        json_path = os.path.join(self.data_dir, 'chips_layout.json')
        
        records = []
        with open(json_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                records.append(record)
        
        self.df = pd.DataFrame(records)
        
        # Convert all features to numeric type
        self.df[self.feature_columns] = self.df[self.feature_columns].apply(pd.to_numeric, errors='coerce')
        print(f"Loaded {len(self.df)} records with {len(self.feature_columns)} features each")

    def clean_missing_values(self):
        """Handle missing values using median imputation"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_json_data() first")

        # Check for missing values
        missing = self.df[self.feature_columns].isna().sum().sum()
        print(f"Found {missing} missing values in total")

        # Median imputation
        imputer = SimpleImputer(strategy='median')
        self.df[self.feature_columns] = imputer.fit_transform(self.df[self.feature_columns])
        print("Missing values imputed using median strategy")

    def detect_and_cap_outliers(self):
        """Identify and cap outliers using IQR method for each feature"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_json_data() first")

        self.outliers_info = {}
        
        for col in self.feature_columns:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Detect and cap outliers
            outliers = self.df[col].between(lower_bound, upper_bound, inclusive='both').sum()
            self.outliers_info[col] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers_count': len(self.df) - outliers
            }
            
            # Cap values
            self.df[col] = self.df[col].clip(lower_bound, upper_bound)
        
        print("Outlier capping complete")

    def normalize_data(self):
        """Normalize all features to [0, 1] range"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_json_data() first")

        scaler = MinMaxScaler()
        self.df[self.feature_columns] = scaler.fit_transform(self.df[self.feature_columns])
        print("Normalization complete")

    def visualize_feature_distribution(self, feature_num):
        """Visualize distribution for a specific feature"""
        if feature_num not in range(256):
            raise ValueError("Feature number must be between 0-255")
            
        col = str(feature_num)
        plt.figure(figsize=(12, 6))
        sns.histplot(self.df[col], kde=True, bins=30)
        plt.title(f"Distribution of Feature {col}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()

    def visualize_feature_outliers(self, feature_num):
        """Visualize outliers for a specific feature"""
        if feature_num not in range(256):
            raise ValueError("Feature number must be between 0-255")
        
        col = str(feature_num)
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=self.df[col])
        plt.title(f"Boxplot for Feature {col}")
        plt.xlabel("Value")
        plt.show()

    def save_clean_data(self, output_dir):
        """Save cleaned data to CSV"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_json_data() first")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path = os.path.join(output_dir, 'cleaned_layout_data.csv')
        self.df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")

# Example usage
if __name__ == "__main__":
    cleaner = DataCleaner(data_dir='path/to/data')
    cleaner.load_json_data()
    
    # Data cleaning pipeline
    cleaner.clean_missing_values()
    cleaner.detect_and_cap_outliers()
    cleaner.normalize_data()
    
    # Visualize sample features
    cleaner.visualize_feature_distribution(0)
    cleaner.visualize_feature_outliers(127)
    cleaner.visualize_feature_distribution(255)
    
    cleaner.save_clean_data(output_dir='cleaned_data')