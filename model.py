import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import pathlib


MODEL_PATH = pathlib.Path(__file__).parent / "data" / "model.pkl"
SCALER_PATH = pathlib.Path(__file__).parent / "data" / "scaler.pkl"

# Features we actually feed into the model
# filepath and modification_hour are excluded from the model
# (filepath is an identifier, modification_hour is weak on its own)
FEATURE_COLS = [
    "entropy",
    "magic_mismatch",
    "is_executable",
    "file_size",
    "size_zscore",
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the feature matrix for model input.
    Drops rows with missing/error values (-1 sentinel).
    """
    df = df.copy()

    # Drop rows where feature extraction failed
    df = df[df["entropy"] >= 0]
    df = df[df["file_size"] >= 0]

    # Fill any remaining NaNs with 0
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    return df


def train_model(df: pd.DataFrame, contamination: float = 0.01) -> tuple:
    """
    Train an Isolation Forest on the provided DataFrame.
    Scales features first, then fits the model.
    Returns (model, scaler, clean_df).
    """
    df = prepare_features(df)

    X = df[FEATURE_COLS].values

    # Scale features so no single feature dominates by magnitude
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"[model] Training Isolation Forest on {len(df)} files...")
    print(f"[model] Contamination rate: {contamination}")

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    print("[model] Training complete.")
    return model, scaler, df


def score_files(df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    """
    Score all files in the DataFrame.
    Adds two columns:
      - anomaly_score: raw score from model (more negative = more anomalous)
      - is_anomaly: 1 if flagged, 0 if normal
    """
    df = prepare_features(df)
    X = df[FEATURE_COLS].values
    X_scaled = scaler.transform(X)

    df = df.copy()
    df["anomaly_score"] = model.decision_function(X_scaled)
    df["is_anomaly"] = (model.predict(X_scaled) == -1).astype(int)

    return df


def save_model(model, scaler):
    """Persist model and scaler to disk."""
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[model] Saved model to {MODEL_PATH}")
    print(f"[model] Saved scaler to {SCALER_PATH}")


def load_model() -> tuple:
    """Load model and scaler from disk."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print("[model] Loaded model and scaler from disk.")
    return model, scaler