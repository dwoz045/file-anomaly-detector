import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
import pathlib
from sklearn.preprocessing import StandardScaler
from model import FEATURE_COLS, prepare_features


AUTOENCODER_PATH = pathlib.Path(__file__).parent / "data" / "autoencoder.pkl"
AE_SCALER_PATH = pathlib.Path(__file__).parent / "data" / "ae_scaler.pkl"


# ─── Architecture ────────────────────────────────────────────────────────────

class FileAutoencoder(nn.Module):
    """
    Autoencoder that learns to reconstruct normal file feature vectors.
    Files it reconstructs poorly (high loss) are anomalous.

    Architecture:
        Encoder: 5 -> 16 -> 8 -> 3  (compresses to a 3-dimensional latent space)
        Decoder: 3 -> 8 -> 16 -> 5  (reconstructs back to original feature size)

    The bottleneck of 3 dimensions forces the model to learn only the most
    essential structure of normal files. Anomalous files don't fit that
    structure, so reconstruction error is high.
    """
    def __init__(self, input_dim: int = 5):
        super(FileAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def reconstruction_error(self, x):
        """Per-sample mean squared reconstruction error."""
        with torch.no_grad():
            reconstructed = self.forward(x)
            errors = torch.mean((x - reconstructed) ** 2, dim=1)
        return errors


# ─── Training ─────────────────────────────────────────────────────────────────

def train_autoencoder(
    df: pd.DataFrame,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    threshold_percentile: float = 95.0
) -> tuple:
    """
    Train the autoencoder on normal file features.
    Returns (model, scaler, threshold) where threshold is the reconstruction
    error above which a file is considered anomalous.
    """
    df = prepare_features(df)
    X = df[FEATURE_COLS].values.astype(np.float32)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.FloatTensor(X_scaled)

    # Build model and optimiser
    model = FileAutoencoder(input_dim=len(FEATURE_COLS))
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print(f"[autoencoder] Training on {len(df)} files for {epochs} epochs...")

    # Training loop
    model.train()
    for epoch in range(epochs):
        # Shuffle data each epoch
        perm = torch.randperm(X_tensor.size(0))
        X_tensor = X_tensor[perm]

        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i + batch_size]
            optimiser.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / num_batches
            print(f"[autoencoder] Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.6f}")

    # Compute reconstruction errors on training data to set threshold
    model.eval()
    with torch.no_grad():
        errors = model.reconstruction_error(X_tensor).numpy()

    # Anything above this percentile of training error is flagged as anomalous
    threshold = float(np.percentile(errors, threshold_percentile))
    print(f"[autoencoder] Anomaly threshold (p{threshold_percentile}): {threshold:.6f}")
    print("[autoencoder] Training complete.")

    return model, scaler, threshold


# ─── Scoring ──────────────────────────────────────────────────────────────────

def score_files_autoencoder(
    df: pd.DataFrame,
    model: FileAutoencoder,
    scaler: StandardScaler,
    threshold: float
) -> pd.DataFrame:
    """
    Score all files using reconstruction error.
    Adds two columns:
      - ae_reconstruction_error: how poorly the autoencoder reconstructed this file
      - ae_is_anomaly: 1 if reconstruction error exceeds threshold
    """
    df = prepare_features(df).copy()
    X = df[FEATURE_COLS].values.astype(np.float32)
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)

    model.eval()
    errors = model.reconstruction_error(X_tensor).numpy()

    df["ae_reconstruction_error"] = errors
    df["ae_is_anomaly"] = (errors > threshold).astype(int)

    return df


def combined_anomaly_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine Isolation Forest and autoencoder signals into a single
    consensus score.

    Agreement levels:
      - both_flagged:     flagged by both models (highest confidence)
      - if_only:          only Isolation Forest flagged it
      - ae_only:          only autoencoder flagged it
      - clean:            neither model flagged it
    """
    df = df.copy()

    conditions = [
        (df["is_anomaly"] == 1) & (df["ae_is_anomaly"] == 1),
        (df["is_anomaly"] == 1) & (df["ae_is_anomaly"] == 0),
        (df["is_anomaly"] == 0) & (df["ae_is_anomaly"] == 1),
    ]
    labels = ["both_flagged", "if_only", "ae_only"]

    df["consensus"] = np.select(conditions, labels, default="clean")
    return df


# ─── Persistence ──────────────────────────────────────────────────────────────

def save_autoencoder(model: FileAutoencoder, scaler: StandardScaler, threshold: float):
    with open(AUTOENCODER_PATH, "wb") as f:
        pickle.dump((model, scaler, threshold), f)
    print(f"[autoencoder] Saved to {AUTOENCODER_PATH}")


def load_autoencoder() -> tuple:
    with open(AUTOENCODER_PATH, "rb") as f:
        model, scaler, threshold = pickle.load(f)
    print("[autoencoder] Loaded from disk.")
    return model, scaler, threshold