# core.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

DATA_PATH = "fraud_data.csv"

def generate_demo_data(n=5000, random_state=42):
    rng = np.random.default_rng(random_state)

    df = pd.DataFrame({
        "step": rng.integers(1, 100, n),
        "amount": rng.exponential(scale=200, size=n),
        "oldbalanceOrg": rng.uniform(0, 5000, n),
        "newbalanceOrig": rng.uniform(0, 5000, n),
        "oldbalanceDest": rng.uniform(0, 5000, n),
        "newbalanceDest": rng.uniform(0, 5000, n),
        "type": rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT"], n),
        "nameOrig": rng.choice([f"C{i}" for i in range(500)], n),
        "nameDest": rng.choice([f"M{i}" for i in range(500)], n),
    })

    # Fraud is rare (realistic)
    log_amount = np.log1p(df["amount"])
    is_cashout = (df["type"] == "CASH_OUT").astype(int)
    is_transfer = (df["type"] == "TRANSFER").astype(int)
    
    # suspicious balance behavior
    orig_delta = np.abs(df["newbalanceOrig"] - df["oldbalanceOrg"])
    dest_delta = np.abs(df["newbalanceDest"] - df["oldbalanceDest"])
    
    risk = (
        1.2 * log_amount +
        1.0 * is_cashout +
        0.8 * is_transfer +
        0.6 * (orig_delta > 1500).astype(int) +
        0.6 * (dest_delta > 1500).astype(int)
    )
    
    # convert risk to probability, then sample labels
    p = 1 / (1 + np.exp(-(risk - 4.0)))   # shift controls base rate
    p = 0.01 + 0.12 * p                   # final probability range ~1% to ~13%
    
    df["isFraud"] = (rng.random(n) < p).astype(int)

    return df



def load_data(path=DATA_PATH):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print("⚠️ fraud_data.csv not found — running DEMO MODE")
        return generate_demo_data()



def safe_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def load_and_explore_data(df: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    data = df.copy()
    if len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    if "step" in data.columns:
        data["step"] = pd.to_numeric(data["step"], errors="coerce").fillna(0)

    if "isFraud" not in data.columns:
        raise ValueError("Column 'isFraud' not found in dataset.")
    return data


def create_behavioral_features(data: pd.DataFrame, enable_rolling: bool = False) -> pd.DataFrame:
    df = data.copy()

    if "step" in df.columns:
        df = df.sort_values("step").reset_index(drop=True)

    # Frequency
    if "nameOrig" in df.columns:
        orig_counts = df.groupby("nameOrig").size()
        df["orig_txn_count"] = df["nameOrig"].map(orig_counts)
        df["orig_txn_freq_7"] = 0.0
        if enable_rolling:
            df["orig_txn_freq_7"] = df.groupby("nameOrig")["step"].transform(
                lambda x: x.rolling(window=7, min_periods=1).count()
            )

    if "nameDest" in df.columns:
        dest_counts = df.groupby("nameDest").size()
        df["dest_txn_count"] = df["nameDest"].map(dest_counts)

    # Velocity
    if "nameOrig" in df.columns and "step" in df.columns:
        df["time_since_last_txn"] = df.groupby("nameOrig")["step"].diff().fillna(0)
        df["avg_time_between_txn"] = 0.0
        if enable_rolling:
            df["avg_time_between_txn"] = df.groupby("nameOrig")["time_since_last_txn"].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )

    # Amount patterns
    if "amount" in df.columns:
        df["avg_amount_7"] = 0.0
        df["std_amount_7"] = 0.0
        df["max_amount_7"] = 0.0

        if "nameOrig" in df.columns and enable_rolling:
            grp = df.groupby("nameOrig")["amount"]
            df["avg_amount_7"] = grp.transform(lambda x: x.rolling(window=7, min_periods=1).mean())
            df["std_amount_7"] = grp.transform(lambda x: x.rolling(window=7, min_periods=1).std())
            df["max_amount_7"] = grp.transform(lambda x: x.rolling(window=7, min_periods=1).max())

        fallback_mean = df["amount"].mean()
        ref_avg = df["avg_amount_7"].replace(0, np.nan).fillna(fallback_mean)
        df["amount_deviation"] = df["amount"] - ref_avg
        df["amount_ratio"] = df["amount"] / (ref_avg + 1e-6)

    # Type patterns
    if "type" in df.columns:
        type_dummies = pd.get_dummies(df["type"], prefix="type", drop_first=True)
        df = pd.concat([df, type_dummies], axis=1)

        if "nameOrig" in df.columns:
            for col in type_dummies.columns:
                df[f"{col}_count"] = df.groupby("nameOrig")[col].transform("sum")

    # Balance features
    if {"oldbalanceOrg", "newbalanceOrig"}.issubset(df.columns):
        df["balance_change_orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
        df["balance_ratio_orig"] = df["newbalanceOrig"] / (df["oldbalanceOrg"] + 1e-6)

    if {"oldbalanceDest", "newbalanceDest"}.issubset(df.columns):
        df["balance_change_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    return df.fillna(0)


def build_ml_matrix(data_enhanced: pd.DataFrame):
    exclude_cols = ["isFraud", "nameOrig", "nameDest"]
    if "type" in data_enhanced.columns:
        exclude_cols.append("type")

    X = data_enhanced.drop(columns=exclude_cols, errors="ignore")
    X = (
        X.apply(pd.to_numeric, errors="coerce")
         .select_dtypes(include=[np.number])
         .fillna(0)
    )
    y = data_enhanced["isFraud"].astype(int).copy()
    return X, y


def train_traditional_models(X_train, y_train, X_test, y_test, cfg):
    models = {}

    if cfg.get("run_logreg", True):
        models["Logistic Regression"] = LogisticRegression(
            max_iter=1000,
            random_state=cfg["random_state"]
        )

    if cfg.get("run_rf", True):
        models["Random Forest"] = RandomForestClassifier(
            n_estimators=200,
            criterion="entropy",
            random_state=cfg["random_state"],
            class_weight="balanced",
        )

    if cfg.get("run_xgb", True):
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBClassifier(
                eval_metric="logloss",
                random_state=cfg["random_state"],
                scale_pos_weight=len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1])),
            )
        else:
            cfg["run_xgb"] = False

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_preds = model.predict_proba(X_train)[:, 1]
        test_preds = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "model": model,
            "train_auc": safe_auc(y_train, train_preds),
            "test_auc": safe_auc(y_test, test_preds),
            "test_preds": test_preds,
            "y_test": y_test.values if hasattr(y_test, "values") else y_test,
            "feature_names": list(X_train.columns),
            "X_test_df": X_test.copy(),
        }

    return results


# ---------- Plot helpers (return figures, don't display) ----------
def fig_confusion_matrix(y_true, y_score, title):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    threshold = 0.5
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {title}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return fig


def fig_roc_pr(results_dict):
    # ROC
    fig_roc, ax = plt.subplots(figsize=(10, 6))
    any_plotted = False
    for name, res in results_dict.items():
        y_true = np.array(res["y_test"])
        y_score = np.array(res["test_preds"])
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        any_plotted = True
    if any_plotted:
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title("ROC Curves (Models)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()

    # PR
    fig_pr, ax2 = plt.subplots(figsize=(10, 6))
    any_plotted = False
    for name, res in results_dict.items():
        y_true = np.array(res["y_test"])
        y_score = np.array(res["test_preds"])
        if len(np.unique(y_true)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax2.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
        any_plotted = True
    if any_plotted:
        ax2.set_title("Precision-Recall Curves (Models)")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.legend()

    return fig_roc, fig_pr


# ----- Extra helpers (no SHAP) -----
def fig_feature_importance(model, feature_names, top_k=20):
    """
    Plot feature importances for tree-based models (RandomForest, XGBoost, etc.).
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not expose feature_importances_.")

    importances = np.asarray(model.feature_importances_)
    idx = np.argsort(importances)[::-1]
    if top_k is not None:
        idx = idx[:top_k]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(idx)), importances[idx][::-1])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(np.array(feature_names)[idx][::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importances (Top)")
    plt.tight_layout()
    return fig


def fig_corr_heatmap(df: pd.DataFrame, max_features=40):
    """
    Correlation heatmap of numeric features.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] > max_features:
        # take top features by variance to keep plot readable
        variances = num_df.var().sort_values(ascending=False)
        cols = list(variances.index[:max_features])
        num_df = num_df[cols]

    corr = num_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (numeric features)")
    plt.tight_layout()
    return fig