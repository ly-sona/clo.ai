# models/xgboost_model.py  –  CSV-driven congestion training
import argparse, numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path

def load_csv(csv_path: str, base_dir: str = "."):
    """
    Returns X (flattened feature tensors) and y (congestion labels).
    CSV must have two columns (no header):
        col0  relative path to feature .npy  (C,H,W or H,W)
        col1  relative path to label   .npy  (scalar or length-1)
    """
    base = Path(base_dir)
    rows = pd.read_csv(csv_path, header=None).values

    X_list, y_list = [], []
    for feat_rel, lab_rel in rows:
        feat = np.load(base / feat_rel, allow_pickle=True)
        lab  = np.load(base / lab_rel,  allow_pickle=True)

        X_list.append(feat.reshape(-1))        # flatten (C×H×W) → 1-D
        y_list.append(float(lab.squeeze()))    # scalar

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train congestion predictor from CSV list")
    p.add_argument("--csv", required=True, help="Path to train_N28.csv")
    p.add_argument("--base_dir", default=".", help="Folder that contains the .npy files")
    p.add_argument("--model_output", default="xgb_congestion_model.json")
    args = p.parse_args()

    X, y = load_csv(args.csv, args.base_dir)
    print(f"Loaded {len(X)} samples, {X.shape[1]} features each")

    # standardise features
    X = StandardScaler().fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    print("MAE:", mean_absolute_error(y_te, y_pred))
    print("R² :", r2_score(y_te, y_pred))

    model.save_model(args.model_output)
    print("Saved →", args.model_output)