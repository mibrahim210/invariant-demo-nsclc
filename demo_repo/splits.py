from sklearn.model_selection import GroupShuffleSplit


def make_split(df, seed=0, test_size=0.2):
    """Split metadata rows into train and test sets; returns index labels for .loc.

    Grouped on patient_id so that all slices of a patient appear in exactly one
    partition, satisfying INV-1 (patient independence).
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_pos, test_pos = next(gss.split(df, groups=df["patient_id"]))
    return df.index[train_pos], df.index[test_pos]
