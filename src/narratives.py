# src/narratives.py  (fixed)
import os
import pandas as pd
import numpy as np

from load_data import load_csv
from shap_analysis import load_model, compute_explainer_and_shap

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

def select_examples(model, X_test, y_test, n_fp=2, n_tn=2, n_fn=1):
    """
    Return a list of index LABELS from X_test (these are the original index labels).
    We will map labels -> positional indices later when extracting shap values.
    """
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else preds

    df = X_test.copy()
    df['y_true'] = y_test.values
    df['y_pred'] = preds
    df['prob'] = probs

    # Define confusion sets (adjust to your convention)
    # Here: 1 = default (positive class)
    fp = df[(df.y_true==0) & (df.y_pred==1)].sort_values('prob', ascending=False)  # false positives
    fn = df[(df.y_true==1) & (df.y_pred==0)].sort_values('prob', ascending=True)   # false negatives
    tp = df[(df.y_true==1) & (df.y_pred==1)].sort_values('prob', ascending=False)
    tn = df[(df.y_true==0) & (df.y_pred==0)].sort_values('prob', ascending=True)

    selected = []
    selected += list(fp.head(n_fp).index)
    selected += list(tn.head(n_tn).index)
    selected += list(fn.head(n_fn).index)

    # Ensure unique, preserve order, limit to 5
    selected = list(dict.fromkeys(selected))[:(n_fp + n_tn + n_fn)]
    return selected

def generate_narratives():
    model = load_model()
    X_train, X_test, y_train, y_test = load_csv()

    # compute explainer and SHAP values for X_test
    explainer, shap_vals = compute_explainer_and_shap(model, X_train, X_test)
    # if multi-class, pick positive-class shap array
    if isinstance(shap_vals, list) and len(shap_vals) > 1:
        shap_vals = shap_vals[1]
    shap_vals = np.array(shap_vals)

    # selected contains original index LABELS (not positions)
    selected_labels = select_examples(model, X_test, y_test)

    # Map labels -> positional indices in X_test
    pos_list = []
    for lbl in selected_labels:
        try:
            pos = X_test.index.get_loc(lbl)
            pos_list.append(pos)
        except KeyError:
            # If label not found (unlikely), skip it
            print(f"Warning: selected index label {lbl} not found in X_test.index - skipping.")
            continue

    narratives = []
    for pos in pos_list:
        # Use positional indexing for both X_test and shap_vals
        instance = X_test.iloc[pos]
        sv = shap_vals[pos]

        feat_df = pd.DataFrame({
            'feature': X_test.columns,
            'value': instance.values,
            'shap_value': sv
        }).sort_values('shap_value', ascending=False)

        pos_feats = feat_df[feat_df['shap_value'] > 0].head(5)
        neg_feats = feat_df[feat_df['shap_value'] < 0].head(5)

        pred = model.predict(X_test.iloc[[pos]])[0]
        prob = model.predict_proba(X_test.iloc[[pos]])[:,1][0] if hasattr(model, 'predict_proba') else None

        narrative = f"Position {pos} (index label {X_test.index[pos]}) — true={y_test.iloc[pos]} pred={pred} prob={prob}\n"
        narrative += "Top positive SHAP contributors (pushing prediction up):\n"
        for _, r in pos_feats.iterrows():
            narrative += f" - {r['feature']} = {r['value']}: SHAP {r['shap_value']:.4f}\n"
        narrative += "Top negative SHAP contributors (pushing prediction down):\n"
        for _, r in neg_feats.iterrows():
            narrative += f" - {r['feature']} = {r['value']}: SHAP {r['shap_value']:.4f}\n"
        narratives.append(narrative)

    out_path = os.path.join(OUT_DIR, 'narratives.txt')
    with open(out_path, 'w') as f:
        f.write("\n\n".join(narratives))

    print("Narratives written to:", out_path)

if __name__ == '__main__':
    generate_narratives()
