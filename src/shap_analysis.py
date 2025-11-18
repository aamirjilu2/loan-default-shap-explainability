# src/shap_analysis.py
import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from load_data import load_csv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.joblib')

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Run src/train_model.py first.")
    return joblib.load(path)

def compute_explainer_and_shap(model, X_background, X_explain):
    """
    Chooses TreeExplainer for tree models; falls back to KernelExplainer for others.
    Returns explainer and shap_values for class 1 (if multi-class/binary returns array).
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
    except Exception:
        # KernelExplainer requires a function that returns probability for the positive class
        # Use a small background sample
        bg = shap.sample(X_background, min(200, len(X_background)))
        f = lambda x: model.predict_proba(x) if hasattr(model, 'predict_proba') else model.predict(x)
        explainer = shap.KernelExplainer(f, bg)
        shap_values = explainer.shap_values(X_explain, nsamples=100)

    # If shap_values is list (multi-class), try pick class 1 (positive)
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values = shap_values[1]

    return explainer, shap_values

def global_and_local_reports(model_path=MODEL_PATH):
    model = load_model(model_path)
    X_train, X_test, y_train, y_test = load_csv()

    # Use a background sample for KernelExplainer if needed
    background = shap.sample(X_train, min(500, len(X_train)))

    print("Computing SHAP explainer and values (may take some time)...")
    explainer, shap_vals = compute_explainer_and_shap(model, background, X_test)

    # Ensure shap_vals is numpy array
    shap_vals = np.array(shap_vals)

    # --- Global summary plot ---
    print("Saving SHAP summary plot...")
    plt.figure(figsize=(9,6))
    shap.summary_plot(shap_vals, X_test, show=False)
    plt.tight_layout()
    summary_path = os.path.join(FIG_DIR, 'summary_plot.png')
    plt.savefig(summary_path, dpi=150)
    plt.close()

    # --- Feature importance (mean absolute) ---
    mean_abs = np.abs(shap_vals).mean(axis=0)
    feat_imp = pd.DataFrame({'feature': X_test.columns, 'mean_abs_shap': mean_abs})
    feat_imp = feat_imp.sort_values('mean_abs_shap', ascending=False)
    feat_imp.to_csv(os.path.join(OUT_DIR, 'feature_importance.csv'), index=False)

    # --- Local explanations for first 5 test instances (or fewer) ---
    n = min(5, len(X_test))
    for i in range(n):
        idx = i
        instance = X_test.iloc[[idx]]
        sv = shap_vals[idx]

        # Waterfall plot (static image)
        try:
            shap.plots.waterfall(shap.Explanation(values=sv, base_values=explainer.expected_value, data=instance.values, feature_names=X_test.columns), show=False)
            plt.tight_layout()
            wf_path = os.path.join(FIG_DIR, f'waterfall_{idx}.png')
            plt.savefig(wf_path, dpi=150)
            plt.close()
        except Exception:
            # fallback: bar plot of top contributors
            contrib = pd.Series(sv, index=X_test.columns).sort_values(ascending=False).head(10)
            plt.figure(figsize=(6,4))
            contrib.plot.bar()
            plt.tight_layout()
            wf_path = os.path.join(FIG_DIR, f'waterfall_{idx}.png')
            plt.savefig(wf_path, dpi=150)
            plt.close()

        # Force plot as HTML (interactive)
        try:
            force = shap.plots.force(shap.Explanation(values=sv, base_values=explainer.expected_value, data=instance.values, feature_names=X_test.columns), matplotlib=False)
            html_path = os.path.join(OUT_DIR, f'force_{idx}.html')
            with open(html_path, 'w') as f:
                f.write(force.html())
        except Exception:
            pass

    print("SHAP analysis finished. Outputs saved to:", OUT_DIR)

if __name__ == '__main__':
    global_and_local_reports()
