import re
import numpy as np

FIELDS = ("conditions", "medications", "procedures", "allergies")
MIN_MASK_RANGE = 0.05


def health_terms(ground_truth, train_ids, fields=FIELDS):
    return sorted(
        {
            str(t).strip()
            for eid in train_ids
            for f in fields
            for t in (ground_truth[eid].get(f) or [])
            if str(t).strip()
        },
        key=lambda t: (-len(t), t),
    )


def removal_pattern(terms):
    if not terms:
        raise ValueError("No training-patient health information found")
    return re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(t) for t in terms) + r")(?!\w)",
        re.IGNORECASE,
    )


def remove_health_information(text, pattern):
    removed = pattern.sub("", text)
    return removed, removed != text


def mask_spread(mask):
    mask = np.asarray(mask, dtype=np.float64)
    return {
        "min": float(mask.min()),
        "max": float(mask.max()),
        "mean": float(mask.mean()),
        "std": float(mask.std()),
        "range": float(mask.max() - mask.min()),
        "above_half": int((mask > 0.5).sum()),
        "zero": int((mask <= 0).sum()),
    }


def is_informative(mask, min_range=MIN_MASK_RANGE):
    return mask_spread(mask)["range"] >= min_range


def shape_diagonal(mask, floor=1e-6):
    mask = np.asarray(mask, dtype=np.float64)
    if mask.ndim != 1 or not np.all(np.isfinite(mask)) or np.any(mask < 0):
        raise ValueError("Mask must be a finite, nonnegative vector")
    if floor <= 0 or mask.sum() <= 0:
        raise ValueError("Positive floor and nonzero mask required")
    diagonal = mask + floor
    return diagonal * len(mask) / diagonal.sum()


def identity_diagonal(dim):
    return np.ones(dim, dtype=np.float64)


def mahalanobis_noise(count, diagonal, epsilon, rng):
    diagonal = np.asarray(diagonal, dtype=np.float64)
    if epsilon <= 0 or not np.isfinite(epsilon) or np.any(diagonal <= 0):
        raise ValueError("Positive finite epsilon and shape diagonal required")
    d = len(diagonal)
    directions = rng.normal(size=(count, d))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-300)
    radii = rng.gamma(shape=d, scale=1.0 / epsilon, size=(count, 1))
    return radii * directions * np.sqrt(diagonal)


def top_k_neighbors(database, queries, k):
    sq_db = np.sum(database ** 2, axis=1)
    out = []
    for start in range(0, len(queries), 512):
        q = queries[start:start + 512]
        distances = sq_db[None, :] - 2.0 * q @ database.T
        out.append(np.argpartition(distances, k, axis=1)[:, :k])
    return np.concatenate(out)


def neighbor_overlap(clean_db, noisy_db, queries, k):
    clean = top_k_neighbors(clean_db, queries, k)
    noisy = top_k_neighbors(noisy_db, queries, k)
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(clean, noisy)]))


def self_test():
    pattern = removal_pattern(["Diabetes (disorder)", "Insulin"])
    s, changed = remove_health_information("Assessment: Diabetes (disorder). Insulin.", pattern)
    assert changed and "Diabetes" not in s and "Insulin" not in s
    assert not remove_health_information("No listed diagnosis.", pattern)[1]
    diagonal = shape_diagonal([0.01, 0.1, 0.7, 1.0])
    assert np.isclose(diagonal.sum(), 4) and np.all(diagonal > 0)
    n, eps, d = 100000, 3.0, 4
    noise = mahalanobis_noise(n, diagonal, eps, np.random.default_rng(71))
    radii = np.linalg.norm(noise / np.sqrt(diagonal), axis=1)
    assert abs(radii.mean() - d / eps) < 0.015
    assert abs(radii.var() - d / eps ** 2) < 0.02
    assert np.allclose(noise.var(axis=0), (d + 1) / eps ** 2 * diagonal, rtol=0.04)
    rng = np.random.default_rng(2)
    x, xp, y = rng.normal(size=(3, d))
    norm = lambda v: np.linalg.norm(v / np.sqrt(diagonal))
    assert eps * (norm(y - xp) - norm(y - x)) <= eps * norm(x - xp) + 1e-10
    assert not is_informative(np.full(384, 0.5))
    assert is_informative(np.r_[np.ones(5), np.zeros(379)])
    assert np.allclose(shape_diagonal(np.full(384, 0.5)), identity_diagonal(384))
    db = rng.normal(size=(200, 8))
    assert neighbor_overlap(db, db, db[:20], 5) == 1.0
    print("Core tests passed: phrase removal, trace, radius, covariance, density bound, mask checks, overlap.")


if __name__ == "__main__":
    self_test()
