from typing import Any, Callable, Optional, TypedDict

import numpy as np


class Instance(TypedDict):
    id: str
    tokens: list[str]
    metadata: Optional[dict[str, Any]]

    loss: float
    predicted_labels: np.ndarray  # (|tokens|,)
    gold_labels: np.ndarray  # (|tokens|,)

    conditional_probs: Callable[
        [list[int]], np.ndarray
    ]  # (|tokens|) -> (|tokens|, C) {y_t} -> {p(y_t | y_{-t}, X)}
    marginal_probs: np.ndarray  # (|tokens|, C) p(y_t | X)
    token_feature_vectors: Optional[np.ndarray]  # (|tokens|, E)

    
def copy_instance(instance: Instance, gold_labels: np.ndarray) -> Instance:
    return {k:(v if k != "gold_labels" else gold_labels) for k, v in instance.items()}


class BaseNERScaffolding:
    output_dir: str
    feature_vector_size: int
    num_classes: int
    class_names: list[str]

    def get_outputs(self, split: str, with_feature_vectors: bool) -> list[Instance]:
        pass
