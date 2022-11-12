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
    num_classes = prediction_probs.shape[1]

    error = prediction_probs - np.eye(num_classes)[true_labels]  # (T, C)

    token_embeddings = np.ascontiguousarray(token_embeddings, dtype=np.float32)
    error = np.ascontiguousarray(error, dtype=np.float32)

    if normalize:
        faiss.normalize_L2(error)
        faiss.normalize_L2(token_embeddings)

    return token_embeddings, error


class NumpyEntityIndexer:
    train_outputs: dict[str, Instance]
    test_outputs: dict[str, Instance]
    scaffolding: BaseNERScaffolding

    def __init__(self, scaffolding: BaseNERScaffolding, normalize: bool):
        self.scaffolding = scaffolding
        self._normalize = normalize
        self._output_dir = f"{self.scaffolding.output_dir}/IF_normalize={self._normalize}"

        self._get_vectors = get_token_gradients

        self._vector_size = self.scaffolding.feature_vector_size * self.scaffolding.num_classes
        self._vector_size += self.scaffolding.num_classes * self.scaffolding.num_classes

    def create_index(self, split: str):
        train_outputs = list(self.scaffolding.get_outputs(split, with_feature_vectors=True))
        num_tokens = sum([len(x["tokens"]) for x in train_outputs])
        self.feature_index = np.zeros((num_tokens, self.scaffolding.feature_vector_size), dtype=np.float32)
        self.error_index = np.zeros((num_tokens, self.scaffolding.num_classes), dtype=np.float32)
        self._ids = []

        n_total = 0
        for instance in train_outputs:
            instance_id = instance["id"]
            token_embeddings, error = self._get_vectors(
                instance, normalize=self._normalize, true_labels=instance["gold_labels"]
            )
            n_tokens = token_embeddings.shape[0]
            assert n_tokens == error.shape[0]
            self.feature_index[n_total : n_total + n_tokens, :] = token_embeddings
            self.error_index[n_total : n_total + n_tokens, :] = error

            self._ids += [(instance_id, i) for i in range(n_tokens)]
            n_total += n_tokens

    def generate_influence_vectors(self, split: str, label_set: Union[str, dict]):
        test_outputs = self.scaffolding.get_outputs(split, with_feature_vectors=True)
        self.test_outputs = {}
        for instance in test_outputs:
            if label_set in ["gold", "predicted"]:
                true_labels = instance[f"{label_set}_labels"]
            else:
                true_labels = label_set[instance["id"]]

            token_embeddings, error = self._get_vectors(
                instance, normalize=self._normalize, true_labels=true_labels
            )
            self.test_outputs[instance["id"]] = {**instance, "feature": token_embeddings, "error": error}

    def search(self, test_idx: str, test_token_idx: int, k: int = 5) -> SearchResult:
        return list(self.batched_search([(test_idx, test_token_idx)], k=k, batch_size=1))[0]

    def batched_search(
        self, examples: Iterable[tuple[str, int]], k: int = 5, batch_size: int = 20
    ) -> Iterable[SearchResult]:
        """
        Need batched search because Faiss more efficient when searching multiple queries at a time
        """
        for batch in chunked(examples, batch_size):
            feature_vector = np.array(
                [self.test_outputs[idx]["feature"][token_idx] for idx, token_idx in batch]
            )
            error_vector = np.array([self.test_outputs[idx]["error"][token_idx] for idx, token_idx in batch])

            D = (self.feature_index @ feature_vector.T) * (self.error_index @ error_vector.T)
            D = D.T
            I_support = np.argpartition(D, range(-1, -k - 1, -1), axis=-1)[:, -k:][:, ::-1]
            D_support = np.take_along_axis(D, I_support, axis=-1)

            I_oppose = np.argpartition(D, range(k), axis=-1)[:, :k]
            D_oppose = -np.take_along_axis(D, I_oppose, axis=-1)

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
