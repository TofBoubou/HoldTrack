"""Remove tracks with too few reliable observations."""

from collections import Counter

import numpy as np


def retained_observations(
    identities: np.ndarray, reliable: np.ndarray, min_track_length: int = 55
) -> np.ndarray:
    """Keep tracks with strictly more than the configured number of observations."""
    counts = Counter((int(identity) for identity, keep in zip(identities, reliable) if keep))
    return np.array(
        [
            keep and counts[int(identity)] > min_track_length
            for identity, keep in zip(identities, reliable)
        ],
        dtype=bool,
    )
