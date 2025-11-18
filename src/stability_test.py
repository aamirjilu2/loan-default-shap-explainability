# src/stability_test.py
import os
import shap
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from load_data import load_csv
from shap_analysis import load_model, compute_explainer_and_shap

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

def stability_test(model_path=None, subsample_n=1000):
    model = load_model(model_path) if model_path else load_model()
    X_train, X_test, y_train, y_test = load_csv()
    X_full = pd.concat([X_train, X_test])

    # Use limited samples to control runtime
    full_sample = X_full.sample(min(2000, len(X_full)), random_state=42)
    sub = X_full.sample(min(subsample_n, len(X_full)), random_state=24)

    expl_full, shap_full = compute_explainer_and_shap(model, X_train, full_sample)
    if isinstance(shap_full, list) and len(shap_full) > 1:
        shap_full = shap_full[1]
    shap_full = np.array(shap_full)
    mean_abs_full = np.abs(shap_full).mean(axis=0)

    expl_sub, shap_sub = compute_explainer_and_shap(model, X_train, sub)
    if isinstance(shap_sub, list) and len(shap_sub) > 1:
        shap_sub = shap_sub[1]
    shap_sub = np.array(shap_sub)
    mean_abs_sub = np.abs(shap_sub).mean(axis=0)

    df_full = pd.DataFrame({'feature': full_sample.columns, 'mean_abs_full': mean_abs_full})
    df_sub = pd.DataFrame({'feature': sub.columns, 'mean_abs_sub': mean_abs_sub})

    merged = df_full.merge(df_sub, on='feature')
    rho, pval = spearmanr(merged['mean_abs_full'], merged['mean_abs_sub'])

    k = 10
    top_full = list(df_full.sort_values('mean_abs_full', ascending=False)['feature'].head(k))
    top_sub = list(df_sub.sort_values('mean_abs_sub', ascending=False)['feature'].head(k))
    overlap = len(set(top_full) & set(top_sub))

    report_path = os.path.join(OUT_DIR, 'stability_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Spearman rho: {rho:.4f}, p-value: {pval:.4e}\n")
        f.write(f"Top-{k} overlap: {overlap} / {k}\n\n")
        f.write("Top features (full sample):\n")
        f.write("\n".join(top_full) + "\n\n")
        f.write("Top features (subsample):\n")
        f.write("\n".join(top_sub) + "\n")

    print("Stability test complete. Report saved to:", report_path)

if __name__ == '__main__':
    stability_test()
