# app.py
import os
import sys

# Ensure local core.py is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import pandas as pd
import numpy as np

import core


DEFAULT_CFG = {
    "sample_size": 300000,
    "random_state": 42,
    "test_size": 0.30,
    "run_logreg": True,
    "run_rf": True,
    "run_xgb": True,
}


def run_pipeline(
    sample_size, test_size, random_state,
    enable_rolling, run_logreg, run_rf, run_xgb, threshold
):
    cfg = {
        "sample_size": int(sample_size),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "run_logreg": bool(run_logreg),
        "run_rf": bool(run_rf),
        "run_xgb": bool(run_xgb),
    }

    # 1) load + sample
    raw_df = core.load_data()
    data = core.load_and_explore_data(raw_df, cfg["sample_size"], cfg["random_state"])

    # 2) feature engineering (cap rows)
    MAX_FE_ROWS = 100_000
    fe_data = data.sample(
        n=min(len(data), MAX_FE_ROWS),
        random_state=cfg["random_state"]
    ).reset_index(drop=True)
    data_enhanced = core.create_behavioral_features(fe_data, enable_rolling=enable_rolling)

    # 3) build X/y
    X, y = core.build_ml_matrix(data_enhanced)

    # 4) train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=y
    )

    MAX_EVAL_ROWS = 20_000
    if len(X_test) > MAX_EVAL_ROWS:
        X_test = X_test.sample(
            n=MAX_EVAL_ROWS, 
            random_state=cfg["random_state"]
        )
        y_test = y_test.loc[X_test.index]

    # 5) train traditional models
    model_results = core.train_traditional_models(X_train, y_train, X_test, y_test, cfg)

    # 6) select best
    best_name, best_auc = None, -1
    for name, res in model_results.items():
        auc = res.get("test_auc", np.nan)
        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_name = name

    # 7) results table
    rows = []
    for name, res in model_results.items():
        rows.append(
            {"Model": name, "Train AUC": res["train_auc"], "Test AUC": res["test_auc"]}
        )
    results_df = pd.DataFrame(rows).sort_values("Test AUC", ascending=False).reset_index(drop=True)

    # 8) plots
    roc_fig, pr_fig = core.fig_roc_pr(model_results)

    cm_fig = None
    report_text = ""
    feat_imp_fig = None  # new: feature importance plot

    if best_name is not None:
        y_true = np.array(model_results[best_name]["y_test"])
        y_score = np.array(model_results[best_name]["test_preds"])
        cm_fig = core.fig_confusion_matrix(y_true, y_score, best_name)
        
        y_pred = (y_score >= float(threshold)).astype(int)
        from sklearn.metrics import classification_report
        report_text = classification_report(y_true, y_pred, digits=4)

        # Feature importance for tree models
        tree_candidates = ["Random Forest", "XGBoost"]
        best_tree_name, best_tree_auc = None, -1

        for name in tree_candidates:
            if name in model_results:
                auc = model_results[name].get("test_auc", np.nan)
                if not np.isnan(auc) and auc > best_tree_auc:
                    best_tree_auc = auc
                    best_tree_name = name

        if best_tree_name is not None:
            tree_model = model_results[best_tree_name]["model"]
            feature_names = model_results[best_tree_name]["feature_names"]
            try:
                feat_imp_fig = core.fig_feature_importance(tree_model, feature_names, top_k=20)
            except Exception:
                feat_imp_fig = None
        else:
            feat_imp_fig = None

    summary_md = (
        f"### ✅ Best Model: **{best_name}**  \n"
        f"**Test AUC:** `{best_auc:.4f}`  \n"
        f"**Fraud Threshold:** `{float(threshold):.2f}`"
        if best_name
        else "### ⚠️ No model selected"
    )

    # Return outputs in the same order as Gradio components
    return (
        summary_md,
        results_df,
        roc_fig,
        pr_fig,
        cm_fig,
        report_text,
        feat_imp_fig,
    )


with gr.Blocks(title="Payments Fraud Detection — Gradio") as demo:
    gr.Markdown("# 💳 Payments Fraud Detection — ML (Gradio)")
    gr.Markdown("This Gradio app trains fraud detection models and shows their performance.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Configuration")

            sample_size = gr.Number(
                value=300_000,
                label="Sample size",
                precision=0,
            )
            test_size = gr.Slider(
                0.10, 0.50,
                value=DEFAULT_CFG["test_size"],
                step=0.05,
                label="Test size",
            )
            random_state = gr.Number(
                value=DEFAULT_CFG["random_state"],
                label="Random state",
                precision=0,
            )

            threshold = gr.Slider(
                0.01, 0.99,
                value=0.50,
                step=0.01,
                label="Fraud Decision Threshold",
            )

            enable_rolling = gr.Checkbox(
                value=False,
                label="Enable rolling behavioral features (slow)",
            )

            gr.Markdown("### Traditional ML")
            run_logreg = gr.Checkbox(value=True, label="Logistic Regression")
            run_rf = gr.Checkbox(value=True, label="Random Forest")
            run_xgb = gr.Checkbox(
                value=True,
                label=f"XGBoost (available={core.XGBOOST_AVAILABLE})",
            )

            run_btn = gr.Button("🚀 Run Training Pipeline", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("## 📌 Results")
            summary = gr.Markdown()

            results_table = gr.Dataframe(label="Model Performance (AUC)", wrap=True)

            with gr.Row():
                roc_plot = gr.Plot(label="ROC Curves")
                pr_plot = gr.Plot(label="Precision-Recall Curves")

            with gr.Row():
                cm_plot = gr.Plot(label="Confusion Matrix")
                feat_imp_plot = gr.Plot(label="Feature Importance (Tree Models)")

            report = gr.Textbox(label="Classification Report", lines=16)

    run_btn.click(
        fn=run_pipeline,
        inputs=[
            sample_size, test_size, random_state,
            enable_rolling, run_logreg, run_rf, run_xgb, threshold
        ],
        outputs=[
            summary,
            results_table,
            roc_plot,
            pr_plot,
            cm_plot,
            report,
            feat_imp_plot,
        ],
    )

demo.queue().launch()