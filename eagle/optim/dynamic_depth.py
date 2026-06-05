from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class DDDConfig:
    max_depth: int = 11
    checkpoints: Tuple[int, ...] = (5, 7, 9)
    thresholds: Tuple[float, ...] = (-6.0, -8.0, -10.0)
    metric: str = "beam_logsumexp"

    def __post_init__(self):
        if len(self.checkpoints) != len(self.thresholds):
            raise ValueError("DDD checkpoints and thresholds must have the same length")
        if self.max_depth < 1:
            raise ValueError("DDD max_depth must be positive")
        if self.metric != "beam_logsumexp":
            raise ValueError(f"Unsupported DDD metric: {self.metric}")


def should_stop_drafting(depth: int, beam_logprob_sums: torch.Tensor, config: DDDConfig) -> bool:
    if depth not in config.checkpoints:
        return False
    threshold = config.thresholds[config.checkpoints.index(depth)]
    score = torch.logsumexp(beam_logprob_sums.float(), dim=0).item()
    return score < threshold
