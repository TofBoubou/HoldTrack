"""Check that the identity chain accepts a clip too short for its classifiers."""

import numpy as np

from holdtrack.identity import links, merge


def test_merge_keeps_the_identities_when_the_classifier_has_no_class():
    rows = np.array([[1, 7, 0, 0, 10, 20], [2, 7, 1, 0, 10, 20], [2, 9, 50, 0, 10, 20]], float)
    crops = np.zeros((3, 10))
    identities, merges, temperature = merge.merge(rows, crops, np.zeros((3, 0)), np.zeros(0))
    assert identities.tolist() == [7, 7, 9] and merges == 0 and temperature is None


def test_second_pass_scores_no_pair():
    model = links.TrainedPair.__new__(links.TrainedPair)
    model.mean, model.std, model.models = np.zeros(21), np.ones(21), [None, None]
    assert model.score(np.zeros((0, 21))).shape == (0,)
