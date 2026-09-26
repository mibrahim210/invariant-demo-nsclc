from sklearn.model_selection import train_test_split


def make_split(df, seed=0, test_size=0.2):
    """Split metadata rows into train and test sets; returns index labels for .loc."""
    train_ids, test_ids = train_test_split(
        df.index.to_numpy(),
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=df["label"],
    )
    return train_ids, test_ids