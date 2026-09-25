import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import mlflow

from demo_repo.dataset import load_metadata, load_array
from demo_repo.splits import make_split

def extract_features(array_path):
    """(2,64,64) -> 8x8 block means -> (2,8,8) -> 128 flat features."""
    arr = load_array(array_path)
    # Reshape and average 8x8 blocks
    blocks = arr.reshape(2, 8, 8, 8, 8).mean(axis=(2, 4))
    return blocks.flatten()  # 128 features

def train_and_evaluate(seed=0):
    df = load_metadata()
    train_ids, test_ids = make_split(df, seed=seed)
    
    print("Extracting features...")
    X_train = np.array([extract_features(df.loc[i, "array_path"]) for i in train_ids])
    y_train = df.loc[train_ids, "label"].values
    X_test = np.array([extract_features(df.loc[i, "array_path"]) for i in test_ids])
    
    # Small CPU-only KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='brute', metric='euclidean', n_jobs=1)
    clf.fit(X_train, y_train)
    
    # Predict slice probabilities for class 1
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_df = df.loc[test_ids].copy()
    test_df["prob"] = test_probs
    
    # Patient-level aggregation: arithmetic mean of held-out slice probabilities per patient
    patient_df = test_df.groupby("patient_id").agg(
        true_label=("label", "first"),
        mean_prob=("prob", "mean")
    )
    
    try:
        patient_auc = roc_auc_score(patient_df["true_label"], patient_df["mean_prob"])
    except ValueError:
        patient_auc = None  # Handle undefined AUC explicitly if only one class is present
        
    # Measure Patient Overlap
    train_patients = set(df.loc[train_ids, "patient_id"])
    test_patients = set(df.loc[test_ids, "patient_id"])
    overlap_count = len(train_patients.intersection(test_patients))
    
    print(f"Seed: {seed}")
    print(f"Patient Overlap (N): {overlap_count}")
    print(f"Patient AUC (X): {patient_auc:.4f}" if patient_auc else "Patient AUC: Unavailable")
    
    with mlflow.start_run():
        mlflow.log_param("split_seed", seed)
        mlflow.log_metric("overlap_count", overlap_count)
        if patient_auc is not None:
            mlflow.log_metric("patient_auc", patient_auc)

if __name__ == "__main__":
    train_and_evaluate(seed=0)