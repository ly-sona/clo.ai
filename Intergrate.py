# CLO.AI: Chip Layout Optimization Framework
# Integrated Pipeline for Chip Layout Optimization using ML and Genetic Algorithms

import numpy as np
import pandas as pd
import xgboost as xgb
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import json
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizationError(Exception):
    """Base exception for optimization errors"""
    pass

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def load_dataset(self, input_dirs, output_dirs=None, manifest_path=None):
        """
        Load dataset from input directory
        """
        try:
            # Load chip layouts from JSON
            if os.path.isfile(input_dirs):
                with open(input_dirs, 'r') as f:
                    data = json.load(f)
                
                # Convert to DataFrame
                df = pd.json_normalize(data)
                
                # Extract features and target
                X = self.extract_features(df)
                y = df['power_metric'] if 'power_metric' in df.columns else None
                
                return X, y
            else:
                raise FileNotFoundError(f"Input file not found: {input_dirs}")
                
        except Exception as e:
            raise OptimizationError(f"Error loading dataset: {str(e)}")
    
    def extract_features(self, data):
        """
        Extract statistical features from layout data
        """
        try:
            # Extract statistical features
            features = {}
            
            # Calculate basic statistics for numerical columns
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in numerical_cols:
                features[f'{col}_mean'] = data[col].mean()
                features[f'{col}_std'] = data[col].std()
                features[f'{col}_min'] = data[col].min()
                features[f'{col}_max'] = data[col].max()
                features[f'{col}_skew'] = data[col].skew()
                features[f'{col}_kurtosis'] = data[col].kurtosis()
            
            # Convert to DataFrame
            feature_df = pd.DataFrame([features])
            
            return feature_df
            
        except Exception as e:
            raise OptimizationError(f"Error extracting features: {str(e)}")
    
    def preprocess_data(self, X, y=None):
        """
        Standardize features and split data
        """
        try:
            X_scaled = self.scaler.fit_transform(X)
            
            if y is not None:
                return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            return X_scaled
            
        except Exception as e:
            raise OptimizationError(f"Error preprocessing data: {str(e)}")

class SurrogateModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_estimators=100
        )
    
    def train(self, X_train, y_train):
        """
        Train the XGBoost model
        """
        try:
            self.model.fit(X_train, y_train)
            logger.info("Surrogate model training completed")
        except Exception as e:
            raise OptimizationError(f"Error training model: {str(e)}")
    
    def predict(self, X):
        """
        Make predictions using the trained model
        """
        try:
            return self.model.predict(X)
        except Exception as e:
            raise OptimizationError(f"Error making predictions: {str(e)}")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        """
        try:
            y_pred = self.predict(X_test)
            return {
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
        except Exception as e:
            raise OptimizationError(f"Error evaluating model: {str(e)}")

class GAOptimizer:
    def __init__(self, surrogate_model, n_features):
        self.surrogate_model = surrogate_model
        self.n_features = n_features
        self.setup_ga()
    
    def setup_ga(self):
        """
        Setup DEAP genetic algorithm components
        """
        try:
            # Clear any existing DEAP types
            if 'FitnessMin' in creator.__dict__:
                del creator.FitnessMin
            if 'Individual' in creator.__dict__:
                del creator.Individual
            
            # Create new types
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)
            
            self.toolbox = base.Toolbox()
            
            # Attribute generator
            self.toolbox.register("attr_float", np.random.uniform, -1, 1)
            
            # Structure initializers
            self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                                self.toolbox.attr_float, n=self.n_features)
            self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
            
            # Genetic operators
            self.toolbox.register("evaluate", self.evaluate)
            self.toolbox.register("mate", tools.cxTwoPoint)
            self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
            self.toolbox.register("select", tools.selTournament, tournsize=3)
            
        except Exception as e:
            raise OptimizationError(f"Error setting up GA: {str(e)}")
    
    def evaluate(self, individual):
        """
        Fitness function using surrogate model
        """
        try:
            prediction = self.surrogate_model.predict(np.array(individual).reshape(1, -1))
            return prediction[0],
        except Exception as e:
            raise OptimizationError(f"Error evaluating individual: {str(e)}")
    
    def optimize(self, n_gen=50, pop_size=100):
        """
        Run genetic algorithm optimization
        """
        try:
            # Create initial population
            population = self.toolbox.population(n=pop_size)
            
            # Statistics setup
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("min", np.min)
            stats.register("avg", np.mean)
            
            # Run the GA
            result, logbook = algorithms.eaSimple(population, self.toolbox,
                                                cxpb=0.7, mutpb=0.2,
                                                ngen=n_gen, stats=stats,
                                                verbose=True)
            
            return result, logbook
            
        except Exception as e:
            raise OptimizationError(f"Error during optimization: {str(e)}")

class Visualizer:
    @staticmethod
    def plot_optimization_progress(logbook):
        """
        Plot GA optimization progress
        """
        try:
            gen = logbook.select("gen")
            fit_mins = logbook.select("min")
            fit_avgs = logbook.select("avg")
            
            plt.figure(figsize=(10, 6))
            plt.plot(gen, fit_mins, 'b-', label='Minimum Fitness')
            plt.plot(gen, fit_avgs, 'r-', label='Average Fitness')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('Optimization Progress')
            plt.legend()
            plt.grid(True)
            plt.show()
            
        except Exception as e:
            raise OptimizationError(f"Error plotting results: {str(e)}")

def run_optimization_pipeline(input_data_path, n_generations=50, population_size=100):
    """
    Main optimization pipeline
    """
    try:
        logger.info("Starting optimization pipeline...")
        
        # Initialize components
        processor = DataProcessor()
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        X, y = processor.load_dataset(input_data_path)
        X_train, X_test, y_train, y_test = processor.preprocess_data(X, y)
        
        # Initialize and train surrogate model
        logger.info("Training surrogate model...")
        surrogate = SurrogateModel()
        surrogate.train(X_train, y_train)
        
        # Evaluate model
        metrics = surrogate.evaluate(X_test, y_test)
        logger.info(f"Surrogate Model Performance: {metrics}")
        
        # Initialize and run GA optimization
        logger.info("Starting genetic algorithm optimization...")
        optimizer = GAOptimizer(surrogate, n_features=X.shape[1])
        result, logbook = optimizer.optimize(n_generations, population_size)
        
        # Visualize results
        logger.info("Generating visualization...")
        visualizer = Visualizer()
        visualizer.plot_optimization_progress(logbook)
        
        return result, logbook, surrogate
        
    except Exception as e:
        raise OptimizationError(f"Pipeline execution failed: {str(e)}")

if __name__ == "__main__":
    try:
        # Example usage
        input_path = "path/to/data/chip_layouts.json"
        result, logbook, model = run_optimization_pipeline(input_path)
        
        # Get best solution
        best_individual = result[0]
        best_fitness = best_individual.fitness.values[0]
        
        print(f"Best fitness achieved: {best_fitness}")
        print(f"Best layout configuration: {best_individual}")
        
    except OptimizationError as e:
        logger.error(f"Optimization failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")