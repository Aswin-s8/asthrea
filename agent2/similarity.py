"""
similarity.py — Compute a weighted similarity score between developer
repositories and a submitted patch repository.

Weights:
  comment_ratio       → 0.2
  avg_line_length     → 0.2
  indent_style        → 0.2
  avg_function_length → 0.4

The returned score is normalised to [0, 1].
"""

import logging

logger = logging.getLogger(__name__)

# Weights for each feature
WEIGHTS = {
    "comment_ratio": 0.2,
    "avg_line_length": 0.2,
    "indent_style": 0.2,
    "avg_function_length": 0.4,
}

# Maximum expected differences (used for normalisation)
MAX_DIFFS = {
    "comment_ratio": 1.0,        # ratio is 0-1
    "avg_line_length": 120.0,    # reasonable max diff in chars
    "avg_function_length": 100.0,  # reasonable max diff in lines
}


def _feature_similarity(dev_val, patch_val, key: str) -> float:
    """Return a 0-1 similarity for a single feature."""
    if key == "indent_style":
        return 1.0 if dev_val == patch_val else 0.0

    max_diff = MAX_DIFFS.get(key, 1.0)
    diff = abs(float(dev_val) - float(patch_val))
    return max(0.0, 1.0 - diff / max_diff)


def compute_similarity(dev_features_list: list[dict], patch_features: dict) -> float:
    """
    Average the developer's feature dicts, then compute a weighted similarity
    score against the patch repo's features.

    Returns a float in [0, 1].
    """
    if not dev_features_list:
        logger.warning("No developer features provided — returning 0.0")
        return 0.0

    # Average the developer features
    avg_dev: dict = {}
    for key in WEIGHTS:
        if key == "indent_style":
            # Majority vote
            styles = [f.get("indent_style", "spaces") for f in dev_features_list]
            avg_dev["indent_style"] = max(set(styles), key=styles.count)
        else:
            values = [f.get(key, 0.0) for f in dev_features_list]
            avg_dev[key] = sum(values) / len(values)

    logger.info("Averaged developer features: %s", avg_dev)

    # Weighted similarity
    score = 0.0
    for key, weight in WEIGHTS.items():
        sim = _feature_similarity(avg_dev.get(key, 0), patch_features.get(key, 0), key)
        score += weight * sim

    score = round(min(max(score, 0.0), 1.0), 4)
    logger.info("Computed similarity score: %s", score)
    return score
