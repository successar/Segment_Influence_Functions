import json
import os
from typing import Iterable, NewType, Optional, Union

import faiss
import numpy as np
from more_itertools import chunked

from ner_influence.scaffolding import BaseNERScaffolding, Instance

InstanceNeighbor = NewType("InstanceNeighbor", tuple[str, float])
InstanceSearchResult = NewType("InstanceSearchResult", tuple[list[InstanceNeighbor], list[InstanceNeighbor]])


def get_instance_gradient(
    instance: Instance, true_labels: Optional[list[int]] = None, normalize: bool = False
) -> np.ndarray:
    if true_labels is None:
        true_labels = instance["gold_labels"]

    token_embeddings = instance["token_feature_vectors"]
    prediction_probs = instance["marginal_probs"]

    num_tokens = token_embeddings.shape[0]
    num_classes = prediction_probs.shape[1]

    assert len(true_labels) == num_tokens

    error = prediction_probs - np.eye(num_classes)[true_labels]  # (T, C)
    gradient = (token_embeddings[:, :, None] * error[:, None, :]).reshape(num_tokens, -1)  # (|T|, C*E)

    gradient = gradient.sum(0)[None, :]
    gradient = np.ascontiguousarray(gradient, dtype=np.float32)

    if normalize:
        faiss.normalize_L2(gradient)

    return gradient


def get_instance_feature_embedding(output_dict: Instance, normalize: bool = True, **kwargs) -> np.ndarray:
    token_embeddings = output_dict["token_feature_vectors"].mean(0)[None, :]
    token_embeddings = np.ascontiguousarray(token_embeddings, dtype=np.float32)

    if normalize:
        faiss.normalize_L2(token_embeddings)

    return token_embeddings


class InstanceIndexer:
    train_outputs: dict[str, Instance]
    test_outputs: dict[str, Instance]
    scaffolding: BaseNERScaffolding

    def __init__(
        self, scaffolding: BaseNERScaffolding, normalize: bool, feature_similarity_only: bool = False
    ):
        self.scaffolding = scaffolding
        self._normalize = normalize
        self._feature_similarity_only = feature_similarity_only

        if self._feature_similarity_only:
            self._get_vectors = get_instance_feature_embedding
            self._vector_size = self.scaffolding.feature_vector_size
            name = "KNN"
        else:
            self._get_vectors = get_instance_gradient
            self._vector_size = self.scaffolding.feature_vector_size * self.scaffolding.num_classes
            name = "Influence"

        self._output_dir = f"{self.scaffolding.output_dir}/Instance_{name}_normalize={self._normalize}"

    def create_index(self, split: str):
        ids = []
        index = faiss.IndexFlatIP(self._vector_size)
        train_outputs = self.scaffolding.get_outputs(split, with_feature_vectors=True)

        for instance in train_outputs:
            instance_id = instance["id"]
            influence = self._get_vectors(
                instance, normalize=self._normalize, true_labels=instance["gold_labels"]
            )
            # print(influence.shape)
            index.add(influence)
            ids.append(instance_id)

        self.index = index
        self._utils = {"ids": ids}

    def save_train_outputs(self, split: str):
        self.train_outputs = {
            x["id"]: x for x in self.scaffolding.get_outputs(split, with_feature_vectors=False)
        }

    def load_index(self, split: str):
        self.index = faiss.read_index(f"{self._output_dir}/{split}_influence.index")
        self._utils = json.load(open(f"{self._output_dir}/{split}_index_mappers.json"))

    def create_index_and_save(self, split: str):
        self.create_index(split)
        os.makedirs(self._output_dir, exist_ok=True)

        faiss.write_index(self.index, f"{self._output_dir}/{split}_influence.index")
        with open(f"{self._output_dir}/{split}_index_mappers.json", "w") as f:
            json.dump(self._utils, f)

    def generate_influence_vectors(self, split: str, label_set: Union[str, dict]):
        test_outputs = self.scaffolding.get_outputs(split, with_feature_vectors=True)
        self.test_outputs = {}
        for n, instance in enumerate(test_outputs):
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
        print(f"Done {n}")

    def get_id(self, index) -> str:
        return self._utils["ids"][index]

    def search(self, test_idx: str, k: int = 5) -> InstanceSearchResult:
        return list(self.batched_search([test_idx], k=k, batch_size=1))[0]

    def batched_search(
        self, examples: Iterable[str], k: int = 5, batch_size: int = 20
    ) -> Iterable[InstanceSearchResult]:
        """
        Need batched search because Faiss more efficient when searching multiple queries at a time
        """
        for batch in chunked(examples, batch_size):
            vectors = np.concatenate([self.test_outputs[idx]["influence_vectors"] for idx in batch], axis=0)
            D_support, I_support = self.index.search(vectors, k=k)
            D_oppose, I_oppose = self.index.search(-vectors, k=k)

            supporters = [
                [
                    InstanceNeighbor((self.get_id(neighbor), float(distance)))
                    for neighbor, distance in zip(instance, distances)
                ]
                for instance, distances in zip(I_support, D_support)
            ]
            opposers = [
                [
                    InstanceNeighbor((self.get_id(neighbor), float(distance)))
                    for neighbor, distance in zip(instance, distances)
                ]
                for instance, distances in zip(I_oppose, D_oppose)
            ]

            for supps, opps in zip(supporters, opposers):
                yield (supps, opps)

    def batched_calculate_influence_and_search(
        self, examples: Iterable[tuple[str, list[int]]], k: int = 5, batch_size: int = 20
    ) -> Iterable[InstanceSearchResult]:
        """
        Need batched search because Faiss more efficient when searching multiple queries at a time
        """
        for batch in chunked(examples, batch_size):
            vectors = []
            for idx, labels in batch:
                influence_vec = self._get_vectors(
                    self.test_outputs[idx], normalize=self._normalize, true_labels=labels
                )
                vectors.append(influence_vec)

            vectors = np.concatenate(vectors, axis=0)
            D_support, I_support = self.index.search(vectors, k=k)
            D_oppose, I_oppose = self.index.search(-vectors, k=k)

            supporters = [
                [
                    InstanceNeighbor((self.get_id(neighbor), float(distance)))
                    for neighbor, distance in zip(instance, distances)
                ]
                for instance, distances in zip(I_support, D_support)
            ]
            opposers = [
                [
                    InstanceNeighbor((self.get_id(neighbor), float(distance)))
                    for neighbor, distance in zip(instance, distances)
                ]
                for instance, distances in zip(I_oppose, D_oppose)
            ]

            for supps, opps in zip(supporters, opposers):
                yield (supps, opps)
