"""FinQA reward for StreamIO GRPO jobs: reward(prompt, completion, target) -> float in [0, 1]."""
from .finqa import finqa_metric, finqa_reward


def reward(prompt: str, completion: str, target: str) -> float:
    """StreamIO's reward signature; the prompt isn't needed to score a FinQA answer."""
    return finqa_reward(completion, target)


__all__ = ["reward", "finqa_reward", "finqa_metric"]
