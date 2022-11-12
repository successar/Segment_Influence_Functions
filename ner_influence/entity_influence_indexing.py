from typing import Iterable, NewType, Optional, Union
from more_itertools import chunked

import faiss
import numpy as np

from ner_influence.scaffolding import Instance, BaseNERScaffolding

TokenNeighbor = NewType("TokenNeighbor", tuple[str, int, float])
SearchResult = NewType("SearchResult", tuple[list[TokenNeighbor], list[TokenNeighbor]])


def get_token_gradients(
    output_dict: Instance, true_labels: Optional[list[int]] = None, normalize: bool = True
) -> np.ndarray:
    if true_labels is None:
        true_labels = output_dict["gold_labels"]

    token_embeddings = output_dict["token_feature_vectors"]
    prediction_probs = output_dict["conditional_probs"](labels=true_labels)
    num_tokens = token_embeddings.shape[0]
    num_classes = prediction_probs.shape[1]

    error = prediction_probs - np.eye(num_classes)[true_labels]  # (T, C)
    gradient_F = (token_embeddings[:, :, None] * error[:, None, :]).reshape(num_tokens, -1)  # (|T|, C*E)

    true_labels = np.eye(num_classes)[np.array(true_labels)]
    gradient_T = np.zeros((num_tokens, num_classes, num_classes))
    gradient_T[:-1] += error[:-1, :, None] * true_labels[1:, None, :]  # T[y_t, y_t+1]
    gradient_T[1:] += true_labels[:-1, :, None] * error[1:, None, :]
    gradient_T = gradient_T.reshape(num_tokens, -1)  # (|T|, C*C)

    gradient = np.concatenate([gradient_F, gradient_T], axis=1)  # (|T|, C*E + C*C)
    gradient = np.ascontiguousarray(gradient, dtype=np.float32)

    if normalize:
        faiss.normalize_L2(gradient)

    return gradient


class EntityIndexer:
    train_outputs: dict[str, Instance]
    test_outputs: dict[str, Instance]
    scaffolding: BaseNERScaffolding

    def __init__(
        self,
        scaffolding: BaseNERScaffolding,
        normalize: bool
    ):
        self.scaffolding = scaffolding
        self._normalize = normalize
        self._output_dir = f"{self.scaffolding.output_dir}/IF_normalize={self._normalize}"

        self._get_vectors = get_token_gradients

        self._vector_size = self.scaffolding.feature_vector_size * self.scaffolding.num_classes
        self._vector_size += self.scaffolding.num_classes * self.scaffolding.num_classes
        

    def create_index(self, split: str):
        self.index = faiss.IndexFlatIP(self._vector_size)
        self._ids = []
        train_outputs = self.scaffolding.get_outputs(split, with_feature_vectors=True)

        for instance in train_outputs:
            instance_id = instance["id"]
            influence = self._get_vectors(
                instance, normalize=self._normalize, true_labels=instance["gold_labels"]
            )
            assert influence.shape[0] == len(instance["tokens"]), breakpoint()
            self.index.add(influence)

            self._ids += [(instance_id, i) for i in range(influence.shape[0])]

    def generate_influence_vectors(self, split: str, label_set: Union[str, dict]):
        test_outputs = self.scaffolding.get_outputs(split, with_feature_vectors=True)
        self.test_outputs = {}
        for instance in test_outputs:
            if label_set in ["gold", "predicted"]:
                true_labels = instance[f"{label_set}_labels"]
            else:
                true_labels = label_set[instance["id"]]

            self.test_outputs[instance["id"]] = {
                **instance,
                "influence_vectors": self._get_vectors(
                    instance, true_labels=true_labels, normalize=self._normalize
                ),
            }

    def influence_vector_norms(self, split: str, aggregate=None) -> Iterable[tuple[str, float]]:
        test_outputs = self.scaffolding.get_outputs(split, with_feature_vectors=True)
        for instance in test_outputs:
            idx = instance["id"]
            norm = np.linalg.norm(
                self._get_vectors(instance, true_labels=instance["gold_labels"], normalize=False),
                axis=-1,
            )
            yield idx, norm if aggregate is None else aggregate(norm)

    def search(self, test_idx: str, test_token_idx: int, k: int = 5) -> SearchResult:
        return list(self.batched_search([(test_idx, test_token_idx)], k=k, batch_size=1))[0]

    def batched_search(
        self, examples: Iterable[tuple[str, int]], k: int = 5, batch_size: int = 20
    ) -> Iterable[SearchResult]:
        """
        Need batched search because Faiss more efficient when searching multiple queries at a time
        """
        for batch in chunked(examples, batch_size):
            vectors = np.array(
                [self.test_outputs[idx]["influence_vectors"][token_idx] for idx, token_idx in batch]
            )

            D_support, I_support = self.index.search(vectors, k=k)
            D_oppose, I_oppose = self.index.search(-vectors, k=k)

            supporters = [
                [
                    TokenNeighbor((*self._ids[neighbor], float(distance)))
                    for neighbor, distance in zip(instance, distances)
                ]
                for instance, distances in zip(I_support, D_support)
            ]
            opposers = [
                [
                    TokenNeighbor((*self._ids[neighbor], float(distance)))
                    for neighbor, distance in zip(instance, distances)
                ]
                for instance, distances in zip(I_oppose, D_oppose)
            ]

            for supps, opps in zip(supporters, opposers):
                yield (supps, opps)
