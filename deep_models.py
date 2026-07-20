# deep_models.py
"""
Deep learning models for fraud detection:

- LSTMFraudNet: a sequence model that looks at each account's recent
  transaction history to flag fraud (sequential fraud detection).
- AutoencoderNet: an unsupervised anomaly detector trained only on
  legitimate transactions; transactions it reconstructs poorly are
  flagged as anomalous / likely fraud.

Both training functions return a result dict shaped like the ones in
core.train_traditional_models, so they drop straight into the same
results table / ROC / PR plots used for the traditional models.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from core import safe_auc

TORCH_AVAILABLE = True


# ---------------------------------------------------------------------------
# LSTM — sequential fraud detection
# ---------------------------------------------------------------------------
class _SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.as_tensor(sequences, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class LSTMFraudNet(nn.Module):
    def __init__(self, n_features, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


def build_sequences(data_enhanced: pd.DataFrame, X: pd.DataFrame, seq_len: int = 5):
    """
    Build one fixed-length, zero-left-padded sequence per transaction: the
    account's (nameOrig) last `seq_len` transactions ordered by `step`,
    ending at that transaction. Returns (sequences, original_index_order)
    so callers can align sequences back to a train/test split made on X.
    """
    if "nameOrig" not in data_enhanced.columns or "step" not in data_enhanced.columns:
        raise ValueError("build_sequences requires 'nameOrig' and 'step' columns.")

    feature_cols = list(X.columns)
    n_features = len(feature_cols)

    order = data_enhanced.sort_values(["nameOrig", "step"]).index.to_numpy()
    ordered_feats = X.loc[order, feature_cols].to_numpy(dtype=np.float32)
    ordered_names = data_enhanced.loc[order, "nameOrig"].to_numpy()

    n = len(order)
    sequences = np.zeros((n, seq_len, n_features), dtype=np.float32)

    start = 0
    while start < n:
        end = start + 1
        while end < n and ordered_names[end] == ordered_names[start]:
            end += 1
        group_feats = ordered_feats[start:end]
        for i in range(end - start):
            window = group_feats[max(0, i - seq_len + 1):i + 1]
            sequences[start + i, seq_len - len(window):, :] = window
        start = end

    return sequences, order


def train_lstm_model(data_enhanced, X, y, train_idx, test_idx, cfg):
    seq_len = int(cfg.get("lstm_seq_len", 5))
    epochs = max(1, int(cfg.get("lstm_epochs", 5)))
    hidden_size = int(cfg.get("lstm_hidden_size", 32))
    lr = float(cfg.get("lstm_lr", 1e-3))
    batch_size = int(cfg.get("lstm_batch_size", 256))
    random_state = int(cfg.get("random_state", 42))

    torch.manual_seed(random_state)

    sequences, order_idx = build_sequences(data_enhanced, X, seq_len=seq_len)
    idx_to_pos = {idx: pos for pos, idx in enumerate(order_idx)}

    train_pos = np.array([idx_to_pos[i] for i in train_idx if i in idx_to_pos], dtype=int)
    test_pos = np.array([idx_to_pos[i] for i in test_idx if i in idx_to_pos], dtype=int)

    y_full = y.loc[order_idx].to_numpy(dtype=np.float32) if hasattr(y, "loc") else np.asarray(y, dtype=np.float32)[order_idx]

    X_train_seq, y_train_seq = sequences[train_pos], y_full[train_pos]
    X_test_seq, y_test_seq = sequences[test_pos], y_full[test_pos]

    loader = DataLoader(
        _SequenceDataset(X_train_seq, y_train_seq),
        batch_size=batch_size,
        shuffle=True,
    )

    model = LSTMFraudNet(n_features=sequences.shape[2], hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_pos = max(1, int((y_train_seq == 1).sum()))
    n_neg = max(1, int((y_train_seq == 0).sum()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos, dtype=torch.float32))

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_preds = torch.sigmoid(model(torch.as_tensor(X_train_seq, dtype=torch.float32))).numpy()
        test_preds = torch.sigmoid(model(torch.as_tensor(X_test_seq, dtype=torch.float32))).numpy()

    return {
        "model": model,
        "train_auc": safe_auc(y_train_seq, train_preds),
        "test_auc": safe_auc(y_test_seq, test_preds),
        "test_preds": test_preds,
        "y_test": y_test_seq,
        "feature_names": list(X.columns),
        "X_test_df": X.loc[order_idx[test_pos]].reset_index(drop=True),
    }


# ---------------------------------------------------------------------------
# Autoencoder — unsupervised anomaly-based fraud detection
# ---------------------------------------------------------------------------
class AutoencoderNet(nn.Module):
    def __init__(self, n_features, bottleneck=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 32),
            nn.ReLU(),
            nn.Linear(32, n_features),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder_model(X_train, y_train, X_test, y_test, cfg):
    epochs = max(1, int(cfg.get("ae_epochs", 15)))
    lr = float(cfg.get("ae_lr", 1e-3))
    bottleneck = int(cfg.get("ae_bottleneck", 8))
    batch_size = int(cfg.get("ae_batch_size", 256))
    random_state = int(cfg.get("random_state", 42))

    torch.manual_seed(random_state)

    y_train_arr = y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)
    y_test_arr = y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)

    # Train only on legitimate transactions: the model should learn to
    # reconstruct "normal" behavior well and fraud poorly.
    X_train_legit = X_train[y_train_arr == 0]

    scaler = StandardScaler()
    scaler.fit(X_train_legit)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_legit_scaled = scaler.transform(X_train_legit)

    model = AutoencoderNet(n_features=X_train.shape[1], bottleneck=bottleneck)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.as_tensor(X_train_legit_scaled, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()
    for _ in range(epochs):
        for (xb,) in loader:
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()

    model.eval()

    def reconstruction_error(X_scaled):
        with torch.no_grad():
            xb = torch.as_tensor(X_scaled, dtype=torch.float32)
            recon = model(xb)
            return ((recon - xb) ** 2).mean(dim=1).numpy()

    train_error = reconstruction_error(X_train_scaled)
    test_error = reconstruction_error(X_test_scaled)

    # Min-max normalize reconstruction error (fit on train) into a
    # pseudo-probability so it plugs into the same AUC/ROC/PR machinery
    # used for the supervised models.
    err_min, err_max = train_error.min(), train_error.max()
    denom = max(err_max - err_min, 1e-9)
    train_preds = np.clip((train_error - err_min) / denom, 0, 1)
    test_preds = np.clip((test_error - err_min) / denom, 0, 1)

    return {
        "model": model,
        "scaler": scaler,
        "train_auc": safe_auc(y_train_arr, train_preds),
        "test_auc": safe_auc(y_test_arr, test_preds),
        "test_preds": test_preds,
        "y_test": y_test_arr,
        "feature_names": list(X_train.columns),
        "X_test_df": X_test.copy(),
    }
