# src/load_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def load_csv(path='../data/dataset.csv',
             target_col='loan_status',
             test_size=0.2,
             random_state=42,
             drop_index_col=True):
    """
    Load dataset, basic preprocessing:
    - imputes missing values (median for numeric, most_frequent for categorical)
    - converts known categorical columns to category dtype
    - one-hot encodes categorical columns (drop_first=True)
    - returns X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(path)

    # If a stray index column exists (like unnamed index), drop it
    if drop_index_col:
        unnamed = [c for c in df.columns if 'unnamed' in c.lower() or c.lower() == 'index']
        if unnamed:
            df = df.drop(columns=unnamed)

    # If target not present, raise helpful error
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset. Columns: {list(df.columns)}")

    # Specify categorical columns likely present in your dataset
    categorical_cols = [
        'person_home_ownership',
        'loan_intent',
        'loan_grade',
        'cb_person_default_on_file'
    ]
    # Keep only categorical_cols that actually exist
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    # Identify numeric columns (exclude target)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]

    # Convert listed categorical columns to 'category' dtype
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    # --- Imputation ---
    # Numeric: median
    if numeric_cols:
        num_imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    # Categorical: most frequent (if any)
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    # Split X/y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # One-hot encode categorical columns found in X
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split with stratify if classification
    try:
        strat = y
    except Exception:
        strat = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    return X_train, X_test, y_train, y_test
