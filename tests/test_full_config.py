"""Check the configuration of the complete system."""

from holdtrack.config import preset


def test_full_is_the_crossed_system():
    full, large = preset("full"), preset("2xl")
    assert full.principal_detector == "rf_2xl" and full.seed_detector == "rf_large"
    assert full.auxiliary_detectors == ("yolo", "mirror", "dfine", "rf_large")
    assert full.steps == large.steps and full.reliability_head == large.reliability_head
    assert full.voices == ("BASE", "EXCL", "PROT") and full.extra_view_weight == 2
    assert all(preset(name).seed_detector is None for name in ("large", "2xl", "light"))
