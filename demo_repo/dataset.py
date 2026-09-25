import pandas as pd
import numpy as np

def load_metadata(csv_path="data/metadata.csv"):
    """Load metadata with slice_id as the unique DataFrame index."""
    df = pd.read_csv(csv_path)
    return df.set_index("slice_id")

def load_array(npy_path):
    """Load arrays safely using allow_pickle=False."""
    return np.load(npy_path, allow_pickle=False)