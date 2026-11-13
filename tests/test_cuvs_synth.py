import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faiss")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "task1_synthetic"))

import cuvs_synth as cs  # noqa: E402
from utils import read_fbin  # noqa: E402


def _data(n=4000, d=32, nc=8, rank=6, seed=0, unit=False):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(nc, d)) * 3
    bases = rng.normal(size=(nc, d, rank)) / np.sqrt(rank)
    labels = rng.integers(0, nc, n)
    vectors = centers[labels] + np.einsum(
        "ndr,nr->nd", bases[labels], rng.normal(size=(n, rank))
    )
    vectors += 0.1 * rng.normal(size=vectors.shape)
    if unit:
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors.astype(np.float32)


@pytest.fixture(scope="module")
def fp_dict():
    return cs.fit_fingerprint(_data(), n_clusters=8, pca_components=4, verbose=False)


def _fp(data, seed=42):
    return cs.Fingerprint(
        nclusters=len(data["centroids"]),
        ncols=data["centroids"].shape[1],
        seed=seed,
        cluster_centers=data["centroids"],
        cluster_variances=data["variances_per_dim"],
        cluster_densities=data["densities"],
        pca_components_list=data["pca_components_list"],
        pca_explained_var_list=data["pca_explained_var_list"],
        pca_noise_var=data["pca_noise_var"],
        pca_n_components=data["pca_n_components"],
        norm_quantiles=data["norm_quantiles"],
        norm_unit=data["norm_unit"],
    )


def test_cuvs_noise_quirk(fp_dict):
    assert np.allclose(fp_dict["pca_noise_var"], 1e-6)


def test_roundtrip_npz(fp_dict, tmp_path):
    cs.save_fingerprint(tmp_path / "fp.npz", fp_dict)
    loaded = cs.load_fingerprint(tmp_path / "fp.npz", seed=42)
    reference = _fp(fp_dict)
    np.testing.assert_array_equal(loaded.cluster_centers, reference.cluster_centers)
    np.testing.assert_array_equal(loaded.norm_quantiles, reference.norm_quantiles)
    for actual, expected in zip(loaded.pca_components_list, reference.pca_components_list):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(cs.gen_cluster(3, 100, loaded), cs.gen_cluster(3, 100, reference))


def test_points_per_cluster(fp_dict):
    fp = _fp(fp_dict)
    for total in (8, 1001, 50_000):
        points = cs.get_num_points_per_cluster(total, fp)
        assert points.sum() == total and points.min() >= 1


def test_gen_deterministic_and_unit():
    data = cs.fit_fingerprint(_data(unit=True), 8, 4, verbose=False)
    assert data["norm_unit"] and data["norm_quantiles"] is None
    fp = _fp(data)
    first, second = cs.gen_cluster(2, 500, fp), cs.gen_cluster(2, 500, fp)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, rtol=1e-5)


def test_queries_and_gt(fp_dict, tmp_path):
    fp = _fp(fp_dict)
    n, nq, k = 3000, 50, 20
    sources, rng = cs.sample_query_sources(nq, n, fp)
    picked = cs.generate_to_file(
        fp, n, tmp_path / "base.fbin", query_sources=sources, workers=3, verbose=False
    )
    base = np.array(read_fbin(tmp_path / "base.fbin"))
    np.testing.assert_array_equal(base, cs.generate_synthetic_dataset(fp, n))
    np.testing.assert_array_equal(picked, base[sources])
    queries = cs.add_jitter(picked, rng, normalize=False)
    np.testing.assert_array_equal(queries, cs.generate_queries(nq, n, fp, workers=2))

    distances = ((queries[:, None] - base[None]) ** 2).sum(-1)
    reference = np.argsort(distances, axis=1)[:, :k]
    exact_ids, _ = cs.compute_groundtruth_exact(queries, n, fp, k)
    assert cs.recall(exact_ids, reference, k) == 1.0
    nprobe_ids, _ = cs.compute_groundtruth_nprobe(queries, n, fp, k, nprobes=fp.nclusters)
    assert cs.recall(nprobe_ids, reference, k) == 1.0
    limited_ids, _ = cs.compute_groundtruth_nprobe(queries, n, fp, k, nprobes=1)
    assert 0 < cs.recall(limited_ids, reference, k) <= 1.0
