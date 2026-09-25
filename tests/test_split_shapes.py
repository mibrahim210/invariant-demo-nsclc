import pandas as pd
from demo_repo.splits import make_split

def test_split_returns_correct_sizes():
    # 20 patients x 8 slices = 160 rows
    df = pd.DataFrame({
        "patient_id": [f"P{i:02d}" for i in range(20) for _ in range(8)],
        "label": [i % 2 for i in range(20) for _ in range(8)]
    })
    df.index = [f"slice_{i}" for i in range(160)]
    
    train_ids, test_ids = make_split(df, test_size=0.2)
    
    assert len(train_ids) > 0 and len(test_ids) > 0
    assert len(train_ids) + len(test_ids) == len(df)
    assert len(set(train_ids).intersection(set(test_ids))) == 0 # no shared rows