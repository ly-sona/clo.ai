#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Engineering for VLSI Chip Layout Optimization
This script extracts and processes features from the CircuitNet dataset
to optimize VLSI chip component placement for power consumption.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class VLSIFeatureEngineering:
    """
    Class for extracting and processing features from the CircuitNet dataset
    for VLSI chip layout optimization.
    """
    
    def __init__(self, data_path, output_path="./processed_features/"):
        """
        Initialize the feature engineering class.
        
        Args:
            data_path (str): Path to CircuitNet dataset
            output_path (str): Path to save processed features
        """
        self.data_path = data_path
        self.output_path = output_path
        self.create_output_directory()
        
    def create_output_directory(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            print(f"Created output directory: {self.output_path}")
    
    def load_design_data(self, design_id):
        """
        Load necessary data for a specific design from CircuitNet.
        
        Args:
            design_id (str): Design identifier (e.g., 'RISCY-a-1-c2-u0.7-m1-p1-f0')
            
        Returns:
            dict: Dictionary containing loaded feature maps and metadata
        """
        base_path = os.path.join(self.data_path, design_id)
        design_data = {}
        
        # Load tile-based feature maps
        feature_maps = [
            'macro_region', 'cell_density', 'rudy', 'pin_rudy', 
            'congestion', 'instance_power'
        ]
        
        for feature in feature_maps:
            feature_path = os.path.join(base_path, f"{feature}.npy")
            if os.path.exists(feature_path):
                design_data[feature] = np.load(feature_path)
            else:
                print(f"Warning: {feature} not found for {design_id}")
        
        # Extract metadata from the design_id
        metadata = self.extract_metadata_from_id(design_id)
        design_data['metadata'] = metadata
        
        return design_data
    
    def extract_metadata_from_id(self, design_id):
        """
        Extract metadata from the design ID based on CircuitNet naming convention.
        
        Args:
            design_id (str): Design identifier
            
        Returns:
            dict: Extracted metadata
        """
        # Parse the CircuitNet design ID format
        # Example: RISCY-a-1-c2-u0.7-m1-p1-f0
        parts = design_id.split('-')
        
        metadata = {}
        
        # Handle different CircuitNet naming conventions
        if len(parts) >= 7 and parts[0] in ['RISCY', 'RISCY-FPU', 'zero-riscy']:
            # CircuitNet-N28 format
            metadata['design'] = '-'.join(parts[0:2])
            metadata['macros'] = int(parts[2])
            
            # Extract clock (c2 -> 200MHz, c1 -> 500MHz, c3 -> 50MHz)
            clock_part = parts[3]
            clock_map = {'c1': 500, 'c2': 200, 'c3': 50}
            metadata['frequency'] = clock_map.get(clock_part, 0)
            
            # Extract utilization (u0.7 -> 70%)
            util_part = parts[4]
            metadata['utilization'] = float(util_part[1:]) if util_part.startswith('u') else 0
            
            # Extract macro placement, power mesh, and filler insertion
            metadata['macro_placement'] = int(parts[5][1:]) if parts[5].startswith('m') else 0
            metadata['power_mesh'] = int(parts[6][1:]) if parts[6].startswith('p') else 0
            metadata['filler_insertion'] = int(parts[7][1:]) if len(parts) > 7 and parts[7].startswith('f') else 0
        
        elif '_freq_' in design_id:
            # CircuitNet-N14 format
            # Example: RISCY_freq_50_mp_1_fpu_60_fpa_1.0_p_7_fi_ar
            parts = design_id.split('_')
            metadata['design'] = parts[0]
            
            freq_idx = parts.index('freq') if 'freq' in parts else -1
            mp_idx = parts.index('mp') if 'mp' in parts else -1
            fpu_idx = parts.index('fpu') if 'fpu' in parts else -1
            fpa_idx = parts.index('fpa') if 'fpa' in parts else -1
            p_idx = parts.index('p') if 'p' in parts else -1
            fi_idx = parts.index('fi') if 'fi' in parts else -1
            
            metadata['frequency'] = int(parts[freq_idx + 1]) if freq_idx != -1 and freq_idx + 1 < len(parts) else 0
            metadata['macro_placement'] = int(parts[mp_idx + 1]) if mp_idx != -1 and mp_idx + 1 < len(parts) else 0
            metadata['utilization'] = int(parts[fpu_idx + 1]) / 100 if fpu_idx != -1 and fpu_idx + 1 < len(parts) else 0
            metadata['aspect_ratio'] = float(parts[fpa_idx + 1]) if fpa_idx != -1 and fpa_idx + 1 < len(parts) else 0
            metadata['power_mesh'] = int(parts[p_idx + 1]) if p_idx != -1 and p_idx + 1 < len(parts) else 0
            metadata['filler_insertion'] = 1 if fi_idx != -1 and fi_idx + 1 < len(parts) and parts[fi_idx + 1] == 'ar' else 0
        
        return metadata
    
    def extract_basic_features(self, design_data):
        """
        Extract basic statistical features from design data.
        
        Args:
            design_data (dict): Loaded design data
            
        Returns:
            dict: Extracted features
        """
        features = {}
        metadata = design_data['metadata']
        
        # Add metadata features
        for key, value in metadata.items():
            features[f"meta_{key}"] = value
        
        # Cell density features
        if 'cell_density' in design_data:
            cell_density = design_data['cell_density']
            features['cell_density_mean'] = np.mean(cell_density)
            features['cell_density_max'] = np.max(cell_density)
            features['cell_density_std'] = np.std(cell_density)
            features['cell_density_p90'] = np.percentile(cell_density, 90)
            
            # Calculate cell density gradient (rate of change)
            if cell_density.shape[0] > 1 and cell_density.shape[1] > 1:
                grad_y, grad_x = np.gradient(cell_density)
                grad_magnitude = np.sqrt(grad_y**2 + grad_x**2)
                features['cell_density_grad_mean'] = np.mean(grad_magnitude)
                features['cell_density_grad_max'] = np.max(grad_magnitude)
        
        # RUDY (routing demand) features
        if 'rudy' in design_data:
            rudy = design_data['rudy']
            features['rudy_mean'] = np.mean(rudy)
            features['rudy_max'] = np.max(rudy)
            features['rudy_std'] = np.std(rudy)
            features['rudy_p90'] = np.percentile(rudy, 90)
            
            # Calculate RUDY hotspots (high routing demand areas)
            hotspot_threshold = np.percentile(rudy, 95)
            hotspots = rudy > hotspot_threshold
            features['rudy_hotspot_count'] = np.sum(hotspots)
            features['rudy_hotspot_ratio'] = np.sum(hotspots) / rudy.size
        
        # Pin RUDY features
        if 'pin_rudy' in design_data:
            pin_rudy = design_data['pin_rudy']
            features['pin_rudy_mean'] = np.mean(pin_rudy)
            features['pin_rudy_max'] = np.max(pin_rudy)
            features['pin_rudy_std'] = np.std(pin_rudy)
        
        # Congestion features
        if 'congestion' in design_data:
            congestion = design_data['congestion']
            features['congestion_mean'] = np.mean(congestion)
            features['congestion_max'] = np.max(congestion)
            features['congestion_std'] = np.std(congestion)
            features['congestion_p90'] = np.percentile(congestion, 90)
            
            # Congestion hotspots
            cong_threshold = np.percentile(congestion, 95)
            cong_hotspots = congestion > cong_threshold
            features['congestion_hotspot_count'] = np.sum(cong_hotspots)
            features['congestion_hotspot_ratio'] = np.sum(cong_hotspots) / congestion.size
        
        # Power features
        if 'instance_power' in design_data:
            power = design_data['instance_power']
            features['power_mean'] = np.mean(power)
            features['power_max'] = np.max(power)
            features['power_std'] = np.std(power)
            features['power_p90'] = np.percentile(power, 90)
            features['power_total'] = np.sum(power)
            
            # Power density and distribution
            nonzero_count = np.sum(power > 0)
            if nonzero_count > 0:
                features['power_density'] = features['power_total'] / nonzero_count
            else:
                features['power_density'] = 0
                
            # Power variation coefficient
            if features['power_mean'] > 0:
                features['power_variation_coef'] = features['power_std'] / features['power_mean']
            else:
                features['power_variation_coef'] = 0
        
        # Macro region features
        if 'macro_region' in design_data:
            macro = design_data['macro_region']
            features['macro_area_ratio'] = np.mean(macro > 0)
            
            # Calculate distance to nearest macro
            if np.sum(macro > 0) > 0 and 'cell_density' in design_data:
                features['avg_distance_to_macro'] = self.calculate_distance_to_feature(macro > 0)
        
        return features
    
    def extract_advanced_features(self, design_data):
        """
        Extract advanced VLSI-specific features.
        
        Args:
            design_data (dict): Loaded design data
            
        Returns:
            dict: Extracted advanced features
        """
        features = {}
        
        # Calculate feature co-occurrence and relationships
        if all(k in design_data for k in ['cell_density', 'rudy', 'instance_power']):
            cell_density = design_data['cell_density']
            rudy = design_data['rudy']
            power = design_data['instance_power']
            
            # Normalize arrays for correlation calculation
            cell_density_norm = (cell_density - np.mean(cell_density)) / (np.std(cell_density) if np.std(cell_density) > 0 else 1)
            rudy_norm = (rudy - np.mean(rudy)) / (np.std(rudy) if np.std(rudy) > 0 else 1)
            power_norm = (power - np.mean(power)) / (np.std(power) if np.std(power) > 0 else 1)
            
            # Calculate spatial correlations
            features['density_rudy_corr'] = np.mean(cell_density_norm * rudy_norm)
            features['density_power_corr'] = np.mean(cell_density_norm * power_norm)
            features['rudy_power_corr'] = np.mean(rudy_norm * power_norm)
            
            # Feature interaction: how power scales with density
            high_density = cell_density > np.percentile(cell_density, 75)
            if np.sum(high_density) > 0:
                features['power_in_high_density'] = np.mean(power[high_density])
                features['power_density_ratio'] = features['power_in_high_density'] / np.mean(power)
            else:
                features['power_in_high_density'] = 0
                features['power_density_ratio'] = 0
                
            # Feature interaction: congestion and power
            if 'congestion' in design_data:
                congestion = design_data['congestion']
                high_congestion = congestion > np.percentile(congestion, 75)
                if np.sum(high_congestion) > 0:
                    features['power_in_high_congestion'] = np.mean(power[high_congestion])
                    features['congestion_power_ratio'] = features['power_in_high_congestion'] / np.mean(power)
                else:
                    features['power_in_high_congestion'] = 0
                    features['congestion_power_ratio'] = 0
        
        # Calculate aspect ratio features if dimensions available
        if 'metadata' in design_data and 'aspect_ratio' in design_data['metadata']:
            aspect_ratio = design_data['metadata']['aspect_ratio']
            features['aspect_ratio'] = aspect_ratio
            
            # Deviation from square (1.0 is square)
            features['aspect_ratio_deviation'] = abs(aspect_ratio - 1.0)
        
        # Calculate transistor density if available
        if 'metadata' in design_data and 'cell_density' in design_data:
            cell_density = design_data['cell_density']
            if 'design' in design_data['metadata']:
                # Estimate total transistor count based on design type
                # This is a placeholder - in real implementation, you'd have actual data
                design_type = design_data['metadata']['design']
                estimated_transistors = 0
                
                if 'RISCY' in design_type:
                    estimated_transistors = 1000000  # Example value
                elif 'zero-riscy' in design_type:
                    estimated_transistors = 500000   # Example value
                
                if estimated_transistors > 0:
                    # Calculate overall density
                    total_cells = np.sum(cell_density)
                    if total_cells > 0:
                        features['transistor_density'] = estimated_transistors / total_cells
                    else:
                        features['transistor_density'] = 0
        
        # Interconnect complexity features
        if 'rudy' in design_data and 'pin_rudy' in design_data:
            rudy = design_data['rudy']
            pin_rudy = design_data['pin_rudy']
            
            # Calculate ratio as interconnect complexity indicator
            if np.mean(pin_rudy) > 0:
                features['interconnect_complexity'] = np.mean(rudy) / np.mean(pin_rudy)
            else:
                features['interconnect_complexity'] = 0
                
            # Calculate spatial variation of interconnect demand
            if np.std(rudy) > 0:
                features['interconnect_variation'] = np.std(rudy) / np.mean(rudy) if np.mean(rudy) > 0 else 0
            else:
                features['interconnect_variation'] = 0
        
        return features
    
    def calculate_distance_to_feature(self, feature_map):
        """
        Calculate average Manhattan distance to a feature.
        
        Args:
            feature_map (numpy.ndarray): Binary map with features marked as True/1
            
        Returns:
            float: Average distance to nearest feature
        """
        if not np.any(feature_map):
            return -1
        
        from scipy.ndimage import distance_transform_cdt
        
        # Calculate distance transform (distance to nearest True value)
        distances = distance_transform_cdt(~feature_map, metric='taxicab')
        
        # Calculate average distance in non-feature areas
        non_feature_mask = ~feature_map
        if np.any(non_feature_mask):
            avg_distance = np.mean(distances[non_feature_mask])
        else:
            avg_distance = 0
            
        return avg_distance
    
    def calculate_spatial_autocorrelation(self, feature_map):
        """
        Calculate a simple measure of spatial autocorrelation.
        
        Args:
            feature_map (numpy.ndarray): 2D array of feature values
            
        Returns:
            float: Spatial autocorrelation measure
        """
        if feature_map.size <= 1:
            return 0
            
        # Calculate mean and standard deviation
        feature_mean = np.mean(feature_map)
        feature_std = np.std(feature_map)
        
        if feature_std == 0:
            return 0
            
        # Normalize the feature map
        normalized_map = (feature_map - feature_mean) / feature_std
        
        # Calculate local spatial autocorrelation
        rows, cols = feature_map.shape
        autocorr_sum = 0
        count = 0
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                center = normalized_map[i, j]
                neighbors = [
                    normalized_map[i-1, j], normalized_map[i+1, j],
                    normalized_map[i, j-1], normalized_map[i, j+1]
                ]
                
                for neighbor in neighbors:
                    autocorr_sum += center * neighbor
                    count += 1
        
        if count > 0:
            return autocorr_sum / count
        else:
            return 0
    
    def process_all_designs(self, design_ids):
        """
        Process all designs and extract features.
        
        Args:
            design_ids (list): List of design identifiers
            
        Returns:
            pandas.DataFrame: Dataframe with extracted features
        """
        all_features = []
        
        for idx, design_id in enumerate(design_ids):
            if idx % 10 == 0:
                print(f"Processing design {idx+1}/{len(design_ids)}: {design_id}")
                
            # Load design data
            design_data = self.load_design_data(design_id)
            
            if not design_data:
                print(f"Warning: Could not load data for {design_id}")
                continue
                
            # Extract features
            basic_features = self.extract_basic_features(design_data)
            advanced_features = self.extract_advanced_features(design_data)
            
            # Combine features
            combined_features = {**basic_features, **advanced_features}
            combined_features['design_id'] = design_id
            
            # Add to list
            all_features.append(combined_features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        return features_df
    
    def analyze_features(self, features_df):
        """
        Analyze features, create visualizations, and perform feature selection.
        
        Args:
            features_df (pandas.DataFrame): DataFrame with extracted features
            
        Returns:
            tuple: (selected_features, feature_importance)
        """
        # Drop design_id for correlation analysis
        analysis_df = features_df.drop(columns=['design_id'])
        
        # Check for constant columns and remove them
        constant_cols = [col for col in analysis_df.columns if analysis_df[col].nunique() <= 1]
        if constant_cols:
            print(f"Removing {len(constant_cols)} constant columns: {constant_cols}")
            analysis_df = analysis_df.drop(columns=constant_cols)
        
        # Check for columns with missing values
        missing_cols = [col for col in analysis_df.columns if analysis_df[col].isnull().any()]
        if missing_cols:
            print(f"Columns with missing values: {missing_cols}")
            analysis_df = analysis_df.dropna(axis=1)
        
        # Correlation analysis
        corr_matrix = analysis_df.corr()
        
        # Save correlation heatmap
        plt.figure(figsize=(20, 16))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, linewidths=0.5, annot=False)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'correlation_matrix.png'))
        plt.close()
        
        # Identify highly correlated features (potential redundancy)
        high_corr_threshold = 0.95
        high_corr_features = set()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > high_corr_threshold:
                    high_corr_features.add(corr_matrix.columns[j])
        
        if high_corr_features:
            print(f"Identified {len(high_corr_features)} highly correlated features that could be removed")
        
        # If we have power_total as a target feature, analyze its relationship with other features
        if 'power_total' in analysis_df.columns:
            target_col = 'power_total'
            X = analysis_df.drop(columns=[target_col])
            y = analysis_df[target_col]
            
            # Calculate correlation with target
            target_corr = X.apply(lambda col: col.corr(y))
            
            # Save target correlation plot
            plt.figure(figsize=(12, len(X.columns) * 0.3))
            target_corr.sort_values().plot(kind='barh')
            plt.title(f'Feature Correlation with {target_col}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'target_correlation.png'))
            plt.close()
            
            # Train a preliminary XGBoost model for feature importance
            try:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model
                model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                model.fit(X_train, y_train)
                
                # Get feature importance
                importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Save feature importance plot
                plt.figure(figsize=(12, len(X.columns) * 0.3))
                importance.plot(kind='barh', x='feature', y='importance')
                plt.title('XGBoost Feature Importance')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_path, 'feature_importance.png'))
                plt.close()
                
                # Select top features (e.g., top 20 or those with importance > threshold)
                top_features = importance.head(min(20, len(importance)))['feature'].tolist()
                
                print(f"Top {len(top_features)} features based on XGBoost importance:")
                for i, feature in enumerate(top_features):
                    print(f"{i+1}. {feature} (Importance: {importance[importance['feature'] == feature]['importance'].values[0]:.4f})")
                
                return top_features, importance
                
            except Exception as e:
                print(f"Error training XGBoost model for feature importance: {e}")
                return [], None
        
        # If no power_total column, return all features
        return list(analysis_df.columns), None
    
    def preprocess_and_save(self, features_df, selected_features=None):
        """
        Preprocess selected features and save to disk.
        
        Args:
            features_df (pandas.DataFrame): DataFrame with extracted features
            selected_features (list): List of selected feature names
            
        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """
        if selected_features is None or not selected_features:
            # Use all numeric features if no selection provided
            numeric_cols = features_df.select_dtypes(include=np.number).columns.tolist()
            selected_features = [col for col in numeric_cols if col != 'design_id']
        
        # Ensure 'design_id' is kept for reference
        if 'design_id' in features_df.columns and 'design_id' not in selected_features:
            final_columns = ['design_id'] + selected_features
        else:
            final_columns = selected_features
            
        # Select features
        processed_df = features_df[final_columns].copy()
        
        # Handle any remaining missing values
        processed_df = processed_df.fillna(0)
        
        # Save processed features
        processed_df.to_csv(os.path.join(self.output_path, 'processed_features.csv'), index=False)
        print(f"Saved processed features to {os.path.join(self.output_path, 'processed_features.csv')}")
        
        # Also save a version without design_id for direct use in ML models
        if 'design_id' in processed_df.columns:
            ml_ready_df = processed_df.drop(columns=['design_id'])
            ml_ready_df.to_csv(os.path.join(self.output_path, 'ml_ready_features.csv'), index=False)
            print(f"Saved ML-ready features to {os.path.join(self.output_path, 'ml_ready_features.csv')}")
        
        # If 'power_total' exists, create train/test splits
        if 'power_total' in processed_df.columns:
            if 'design_id' in processed_df.columns:
                X = processed_df.drop(columns=['design_id', 'power_total'])
                design_ids = processed_df['design_id']
            else:
                X = processed_df.drop(columns=['power_total'])
                design_ids = None
                
            y = processed_df['power_total']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Save splits
            X_train.to_csv(os.path.join(self.output_path, 'X_train.csv'), index=False)
            X_test.to_csv(os.path.join(self.output_path, 'X_test.csv'), index=False)
            y_train.to_csv(os.path.join(self.output_path, 'y_train.csv'), index=False)
            y_test.to_csv(os.path.join(self.output_path, 'y_test.csv'), index=False)
            
            print(f"Created and saved train/test splits with {len(X_train)} training and {len(X_test)} test samples")
            
            # If we have design_ids, also save id splits for reference
            if design_ids is not None:
                design_id_train, design_id_test = train_test_split(design_ids, test_size=0.2, random_state=42)
                pd.DataFrame(design_id_train).to_csv(os.path.join(self.output_path, 'design_id_train.csv'), index=False)
                pd.DataFrame(design_id_test).to_csv(os.path.join(self.output_path, 'design_id_test.csv'), index=False)
        
        return processed_df
    
    def run_pipeline(self, design_ids=None):
        """
        Run the full feature engineering pipeline.
        
        Args:
            design_ids (list): List of design identifiers. If None, will attempt to discover from data_path.
            
        Returns:
            pandas.DataFrame: Processed features DataFrame
        """
        # If no design_ids provided, attempt to discover from data_path
        if design_ids is None:
            try:
                design_ids = [d for d in os.listdir(self.data_path) 
                              if os.path.isdir(os.path.join(self.data_path, d))]
                print(f"Discovered {len(design_ids)} designs")
            except Exception as e:
                print(f"Error discovering designs: {e}")
                return None
        
        if not design_ids:
            print("No designs found or provided. Exiting.")
            return None
        
        print(f"Starting feature engineering pipeline for {len(design_ids)} designs")
        
        # Process all designs
        features_df = self.process_all_designs(design_ids)
        
        if features_df.empty:
            print("No features extracted. Exiting.")
            return None
        
        print(f"Extracted {features_df.shape[1]} features for {features_df.shape[0]} designs")
        
        # Analyze features and get selection
        selected_features, _ = self.analyze_features(features_df)
        
        # Preprocess and save
        processed_df = self.preprocess_and_save(features_df, selected_features)
        
        print("Feature engineering pipeline completed successfully")
        
        return processed_df


def main():
    """Main function to run the feature engineering pipeline."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='VLSI Chip Layout Feature Engineering')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to CircuitNet dataset')
    parser.add_argument('--output_path', type=str, default='./processed_features/',
                        help='Path to save processed features')
    parser.add_argument('--max_designs', type=int, default=None,
                        help='Maximum number of designs to process (for testing)')
    
    args = parser.parse_args()
    
    # Initialize feature engineering
    fe = VLSIFeatureEngineering(args.data_path, args.output_path)
    
    # Discover designs
    try:
        all_designs = [d for d in os.listdir(args.data_path) 
                       if os.path.isdir(os.path.join(args.data_path, d))]
    except Exception as e:
        print(f"Error discovering designs: {e}")
        return
    
    # Limit number of designs if specified
    if args.max_designs is not None and args.max_designs > 0:
        print(f"Limiting to {args.max_designs} designs for processing")
        all_designs = all_designs[:args.max_designs]
    
    if not all_designs:
        print("No designs found in the specified directory. Exiting.")
        return
    
    print(f"Found {len(all_designs)} designs to process")
    
    # Run the feature engineering pipeline
    processed_df = fe.run_pipeline(all_designs)
    
    if processed_df is not None:
        print(f"Successfully processed {len(processed_df)} designs with {len(processed_df.columns)} features")
        # Generate summary report
        summary_path = os.path.join(args.output_path, 'feature_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"VLSI Feature Engineering Summary\n")
            f.write(f"==============================\n\n")
            f.write(f"Processed {len(processed_df)} designs\n")
            f.write(f"Total features: {len(processed_df.columns)}\n\n")
            f.write(f"Feature list:\n")
            for col in processed_df.columns:
                if col != 'design_id':
                    f.write(f"- {col}\n")
        
        print(f"Summary report written to {summary_path}")
        
        # Save dataset stats
        stats_df = processed_df.describe().transpose()
        stats_df.to_csv(os.path.join(args.output_path, 'feature_statistics.csv'))
        print(f"Feature statistics saved to {os.path.join(args.output_path, 'feature_statistics.csv')}")
    else:
        print("Feature engineering process failed or no valid data was produced.")

if __name__ == "__main__":
    main()