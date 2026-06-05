import torch

from eagle.optim.dynamic_depth import DDDConfig, should_stop_drafting


def test_ddd_no_stop_when_score_high():
    config = DDDConfig(checkpoints=(5,), thresholds=(-6.0,))
    scores = torch.tensor([-1.0, -1.5, -2.0])
    assert should_stop_drafting(5, scores, config) is False


def test_ddd_stop_when_score_low():
    config = DDDConfig(checkpoints=(5,), thresholds=(-1.0,))
    scores = torch.tensor([-6.0, -7.0, -8.0])
    assert should_stop_drafting(5, scores, config) is True


def test_ddd_ignores_non_checkpoint_depth():
    config = DDDConfig(checkpoints=(5,), thresholds=(100.0,))
    scores = torch.tensor([-100.0])
    assert should_stop_drafting(4, scores, config) is False


def test_ddd_threshold_mapping():
    config = DDDConfig(checkpoints=(5, 7), thresholds=(-100.0, 100.0))
    scores = torch.tensor([0.0])
    assert should_stop_drafting(5, scores, config) is False
    assert should_stop_drafting(7, scores, config) is True
