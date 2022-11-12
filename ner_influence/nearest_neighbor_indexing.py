from typing import Iterable, NewType
from more_itertools import chunked

import faiss
import numpy as np

from ner_influence.scaffolding import Instance, BaseNERScaffolding

TokenNeighbor = NewType("TokenNeighbor", tuple[str, int, float])
SearchResult = NewType("SearchResult", list[list[TokenNeighbor]])


def get_feature_vectors(output_dict: Instance, normalize: bool = True) -> np.ndarray:
    token_embeddings = output_dict["token_feature_vectors"]
    influence = np.ascontiguousarray(token_embeddings, dtype=np.float32)

    if normalize:
        faiss.normalize_L2(influence)

    return influence


class NNIndexer:
    train_outputs: dict[str, Instance]
    test_outputs: dict[str, Instance]
    scaffolding: BaseNERScaffolding

    def __init__(
        self,
        scaffolding: BaseNERScaffolding,
        normalize: bool,
    ):
        self.scaffolding = scaffolding
        self._normalize = normalize
        self._output_dir = f"{self.scaffolding.output_dir}/knn_normalize={self._normalize}"

        self._get_vectors = get_feature_vectors
        self._vector_size = self.scaffolding.feature_vector_size

        self._indices = [None]*scaffolding.num_classes
        

    def create_index(self, split: str):
        self._indices = [faiss.IndexFlatIP(self._vector_size) for _ in range(self.scaffolding.num_classes)]
        self._ids = [[] for _ in range(self.scaffolding.num_classes)]
        train_outputs = self.scaffolding.get_outputs(split, with_feature_vectors=True)

        for instance in train_outputs:
            instance_id = instance["id"]
            influence = self._get_vectors(instance, normalize=self._normalize)
            labels = instance["gold_labels"]
            assert influence.shape[0] == len(instance["tokens"]), breakpoint()

            for i, label in enumerate(labels):
                self._indices[label].add(influence[i][None, :])
                self._ids[label].append((instance_id, i))

    def generate_influence_vectors(self, split: str):
        test_outputs = self.scaffolding.get_outputs(split, with_feature_vectors=True)
        self.test_outputs = {}
        for instance in test_outputs:
            self.test_outputs[instance["id"]] = {
                **instance,
                "influence_vectors": self._get_vectors(
                    instance, normalize=self._normalize
                ),
            }

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

            outputs = [[] for _ in range(len(batch))]

            for i in range(self.scaffolding.num_classes):
                if len(self._ids[i]) == 0:
                    outputs = outputs = [outputs[i] + [None] for i in range(len(batch))]
                else:
                    D, I = self._indices[i].search(vectors, k=k)
                    neighbors = [
                        [
                            TokenNeighbor((*self._ids[i][neighbor], float(distance)))
                            for neighbor, distance in zip(instance, distances)
                        ]
                        for instance, distances in zip(I, D)
                    ]
                    assert len(neighbors) == len(batch)
                    outputs = [outputs[i] + [neighbors[i]] for i in range(len(batch))]


            for n in outputs:
                assert len(n) == self.scaffolding.num_classes
                yield n
