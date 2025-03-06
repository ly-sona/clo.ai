import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataCleaner:
    def __init__(self, data_dir, feature_type):
        self.data_dir = data_dir
        self.feature_type = feature_type
        self.dataframes = []
        self.outliers_info = {}

    def load_data(self):
        """
        Load all CSV files from the specified directory into pandas DataFrames.
        """
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.csv') and self.feature_type in root:
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path)
                    self.dataframes.append(df)
        print(f"Loaded {len(self.dataframes)} data files from {self.data_dir}")

    def clean_missing_values(self):
        """
        Identify and impute missing values with the median.
        """
        imputer = SimpleImputer(strategy='median')
        for i, df in enumerate(self.dataframes):
            self.dataframes[i] = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        print("Missing values imputed with median")

    def detect_outliers(self):
        """
        Detect outliers using IQR method and cap them at upper and lower bounds.
        """
        for i, df in enumerate(self.dataframes):
            self.outliers_info[i] = {}
            for column in df.select_dtypes(include=[np.number]).columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
                self.outliers_info[i][column] = outliers
                # Cap outliers
                df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
                df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        print("Outliers detected and capped")

    def normalize_data(self):
        """
        Normalize numerical features to [0, 1] range using MinMaxScaler.
        """
        scaler = MinMaxScaler()
        for i, df in enumerate(self.dataframes):
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = scaler.fit_transform(df[num_cols])
        print("Data normalized")

    def visualize_outliers(self):
        """
        Plot boxplots for each numerical feature to visualize outliers.
        """
        for i, df in enumerate(self.dataframes):
            for column in df.select_dtypes(include=[np.number]).columns:
                plt.figure(figsize=(10, 4))
                sns.boxplot(x=df[column])
                plt.title(f"Feature: {column} - Dataset {i}")
                plt.show()

    def save_cleaned_data(self, save_path):
        """
        Save cleaned datasets to the specified path.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i, df in enumerate(self.dataframes):
            file_name = f"cleaned_dataset_{i}.csv"
            df.to_csv(os.path.join(save_path, file_name), index=False)
        print(f"Cleaned data saved to {save_path}")

# Example Usage
# cleaner = DataCleaner(data_dir='data', feature_type='routability_features')
# cleaner.load_data()
# cleaner.clean_missing_values()
# cleaner.detect_outliers()
# cleaner.normalize_data()
# cleaner.visualize_outliers()
# cleaner.save_cleaned_data(save_path='cleaned_data')
