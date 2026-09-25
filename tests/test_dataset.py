import numpy as np
from demo_repo.dataset import load_array

def test_array_properties(tmp_path):
    # Test array shape, dtype, and finite [0,1] values
    test_file = tmp_path / "test_arr.npy"
    arr = np.random.uniform(0, 1, size=(2, 64, 64)).astype(np.float32)
    np.save(test_file, arr, allow_pickle=False)
    
    loaded = load_array(test_file)
    assert loaded.shape == (2, 64, 64)
    assert loaded.dtype == np.float32
    assert np.all(np.isfinite(loaded))
    assert np.min(loaded) >= 0.0 and np.max(loaded) <= 1.0