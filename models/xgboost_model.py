# xgboost_model.py
import argparse
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from preprocessing import load_dataset

def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost model on CircuitNet data for power prediction."
    )
    parser.add_argument("--input_dirs", required=True,
                        help="Comma-separated list of input .npy feature directories.")
    parser.add_argument("--output_dirs", required=True,
                        help="Comma-separated list of output .npy label directories.")
    parser.add_argument("--manifest", required=False,
                        help="Optional manifest file (JSON or text) for sample paths.")
    parser.add_argument("--preserve_shape", action="store_true",
                        help="Preserve array shape (no flattening).")
    parser.add_argument("--model_output", default="xgb_layout_model.json",
                        help="Path to save the trained XGBoost model.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Power consumption threshold for reporting.")
    args = parser.parse_args()

    # Load data
    X_raw, y = load_dataset(
        args.input_dirs,
        output_dirs=args.output_dirs,
        manifest_path=args.manifest,
        preserve_shape=args.preserve_shape
    )

    features = np.array([sample.reshape(-1) for sample in X_raw]) 

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train model
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred)}")
    print(f"R-squared: {r2_score(y_test, y_pred)}")

    # Save model
    model.save_model(args.model_output)
    print(f"Trained model saved to {args.model_output}")

if __name__ == "__main__":
    main()