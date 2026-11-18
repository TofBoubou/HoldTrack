"""Test immutable refinement settings."""

import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from holdtrack.config import TrackSteps, preset
from holdtrack.sam.liam import ConstantVelocityKalman
from holdtrack.steps.short_tracks import retained_observations


class ConfigurationTests(unittest.TestCase):
    def test_motion_state_retains_its_typed_filter(self):
        state = ConstantVelocityKalman.from_box((0.0, 0.0, 8.0, 16.0))
        state.predict()
        state.update((1.0, 0.0, 9.0, 16.0))
        self.assertTrue(np.isfinite(state.mean).all())
        self.assertTrue(np.isfinite(state.covariance).all())

    def test_configuration_is_frozen_and_roundtrips(self):
        configuration = TrackSteps()
        with self.assertRaises(FrozenInstanceError):
            configuration.min_track_length = 20
        self.assertEqual(configuration, TrackSteps.from_dict(configuration.to_dict()))

    def test_length_counts_only_reliable_observations(self):
        identities = np.asarray([1, 1, 1, 2, 2])
        reliable = np.asarray([True, True, False, True, True])
        self.assertFalse(retained_observations(identities, reliable, 2).any())
        np.testing.assert_array_equal(retained_observations(identities, reliable, 1), reliable)

    def test_light_has_one_detector_and_one_pass(self):
        configuration = preset("light")
        self.assertEqual(configuration.principal_detector, "rf_2xl")
        self.assertEqual(configuration.voices, ("PROT",))
        self.assertEqual(configuration.auxiliary_detectors, ())
