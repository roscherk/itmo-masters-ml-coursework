import importlib.util
from pathlib import Path

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "experiments" / "task1_synthetic" / "partition.py"
import sys

sys.path.insert(0, str(MODULE.parent))
SPEC = importlib.util.spec_from_file_location("partition", MODULE)
partition = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(partition)


def test_emeans_partition_is_complete_and_deterministic():
    rng = np.random.default_rng(17)
    base = rng.normal(size=(160, 6)).astype(np.float32)
    first = partition.emean_kmeans(base, 4, seed=9, n_train=120,
                                   epochs=2, batch_size=24, verbose=False)
    second = partition.emean_kmeans(base, 4, seed=9, n_train=120,
                                    epochs=2, batch_size=24, verbose=False)

    np.testing.assert_array_equal(first["labels"], second["labels"])
    np.testing.assert_allclose(first["centroids"], second["centroids"])
    assert first["labels"].shape == (len(base),)
    assert first["sizes"].sum() == len(base)
    assert np.isfinite(first["objective"])
