from sklearn.model_selection import train_test_split


def make_split(df, seed=0, test_size=0.2):
    """
    Buggy baseline: train_test_split over row index labels.
    Stratifies rows by label, but allows patient IDs to cross train/test splits.
    """
    train_ids, test_ids = train_test_split(
        df.index.to_numpy(),
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=df["label"],
    )
    return train_ids, test_ids