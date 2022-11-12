from typing import Any, Optional, Union
from sklearn.feature_extraction import DictVectorizer

import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from ner_influence.modelling.datamodule import NERDataModule, Sentence
import spacy

from bisect import bisect_right


def find_le(a, x):
    i = bisect_right(a, x)
    if i:
        return i - 1
    raise ValueError


class LinearNERDataModule(NERDataModule):
    def __init__(
        self,
        splits: dict[str, Union[str, list[str]]],
        label_list: Optional[list[str]] = None,
        batch_size: int = 8,
        loo_mode: bool = False,
        remap_labels: Optional[dict[str, dict[str, str]]] = None,
    ):
        self._splits = splits
        self._batch_size = batch_size

        self._label_list = label_list
        self._remap_labels = remap_labels or {}

        # Need to be set later
        self._dataset: dict[str, list[Sentence]] = {}
        self._train_dataset = None
        self._validation_splits = None

        self.nlp = spacy.load("en_core_web_md")
        self.vectorizer = DictVectorizer(sparse=False)
        self.unk_vector = self.nlp.vocab.vectors.data.mean(0)
        self.loo_mode = loo_mode

    def setup(self, stage=None):
        super().setup(stage)
        self.vectorizer.fit(
            [f for data in self._dataset.values() for sentence in data for f in self.word2features(sentence).features]
        )

        self.feature_size = len(self.vectorizer.feature_names_)
        self.embedding_size = self.nlp.vocab.vectors.shape[1]

    def tokenize(self, sentence: Sentence) -> Sentence:
        sentence = sentence.deepcopy_without_tensors()
        old_tokens = sentence.tokens
        x = 0
        old_starts = [0] + [x := x + len(t) + 1 for t in old_tokens][:-1]
        old_labels = sentence.labels

        tokens = " ".join(sentence.tokens)
        doc = self.nlp(tokens)
        sentence.tokens = [t.text for t in doc]
        sentence.doc = doc

        old_labels_refined = []
        for t in doc:
            old_token_idx = find_le(old_starts, t.idx)
            old_label = old_labels[old_token_idx]
            if t.idx != old_starts[old_token_idx]:
                if old_label.startswith("B-"):
                    old_label = "I-" + old_label[2:]

            old_labels_refined.append(old_label)

        sentence.labels = old_labels_refined

        assert len(sentence.labels) == len(sentence.tokens)

        return sentence

    def collate_fn(self, batch: list[Sentence]) -> dict[str, Any]:
        features = [torch.tensor(self.vectorizer.transform(sentence.features)) for sentence in batch]
        token_vectors = [
            torch.tensor(np.array([token.vector if token.has_vector else self.unk_vector for token in sentence.doc]))
            for sentence in batch
        ]

        features = pad_sequence(features, batch_first=True, padding_value=0)
        token_vectors = pad_sequence(token_vectors, batch_first=True, padding_value=0)
        label_indexed = [torch.tensor([self._label_list.index(x) for x in item.labels]) for item in batch]
        label_indexed = pad_sequence(label_indexed, batch_first=True, padding_value=-100)

        bias_ones = torch.ones(features.shape[0], features.shape[1], 1)
        
        batch_collated = {
            "features": torch.cat([features, token_vectors, bias_ones], dim=2).float(),
            "labels": label_indexed,
            "sentence": batch,
        }

        if self.loo_mode:
            idxs = [item.id for item in batch]
            batch_masked_tokens = [self.masked_tokens[idx] for idx in idxs]
            batch_mask = torch.zeros_like(label_indexed, dtype=torch.bool)
            for i, tokens in enumerate(batch_masked_tokens):
                batch_mask[i, tokens] = True

            batch_collated["tokens_to_remove"] = batch_mask

        return batch_collated

    @staticmethod
    def word2features(sentence: Sentence):
        sentence.features = []
        for i in range(len(sentence.doc)):
            word = sentence.doc[i].text

            features = {
                "word.isupper()": word.isupper(),
                "word.istitle()": word.istitle(),
                "word.isdigit()": word.isdigit(),
                "postag": sentence.doc[i].pos_,
                "is_stop": sentence.doc[i].is_stop,
            }
            if i > 0:
                word1 = sentence.doc[i - 1].text
                features.update(
                    {
                        "-1:word.istitle()": word1.istitle(),
                        "-1:word.isupper()": word1.isupper(),
                        "-1:postag": sentence.doc[i - 1].pos_,
                        "-1:is_stop": sentence.doc[i - 1].is_stop,
                    }
                )
            else:
                features["BOS"] = True

            if i < len(sentence.doc) - 1:
                word1 = sentence.doc[i + 1].text
                features.update(
                    {
                        "+1:word.istitle()": word1.istitle(),
                        "+1:word.isupper()": word1.isupper(),
                        "+1:postag": sentence.doc[i - 1].pos_,
                        "+1:is_stop": sentence.doc[i + 1].is_stop,
                    }
                )
            else:
                features["EOS"] = True

            sentence.features.append(features)
        
        return sentence


class SyntheticNERDataModule(NERDataModule):
    def __init__(
        self, 
        num_labels: int = 5,
        num_features: int = 26,
        batch_size: int = 8,
        loo_mode: bool = False
    ):
        self.loo_mode = loo_mode
        self.num_labels = num_labels
        self.num_features = num_features
        self._batch_size = batch_size
        self._label_list = list(range(num_labels))

        self.feature_size = num_features
        self.embedding_size = 0

    def setup(self):
        self.T = np.zeros((self.num_labels, self.num_labels))
        for i in range(self.num_labels):
            next_s = np.random.choice(self.num_labels, size=(2, ), replace=False)
            next_p = np.random.dirichlet(np.ones(2), size=1)[0]
            self.T[i, next_s] = next_p

        E = np.zeros((self.num_labels, self.num_features))
        for i in range(self.num_labels):
            next_s = np.random.choice(self.num_features, size=(5, ), replace=False)
            next_p = np.random.dirichlet(np.ones(5), size=1)[0]
            E[i, next_s] = next_p

        self.E = E

        self._dataset = {}

        self._dataset["sampled_train"] = []
        for i in range(1000):
            features, labels = self.generate_sample(self.E, self.T)
            self._dataset["sampled_train"].append(Sentence(id=i, tokens=features, labels=labels))

        self._dataset["sampled_validation"] = []
        for i in range(200):
            features, labels = self.generate_sample(self.E, self.T)
            self._dataset["sampled_validation"].append(Sentence(id=i, tokens=features, labels=labels))

    def generate_sample(self, E, T):
        n, v = E.shape
        r = lambda x : np.random.choice(x, size=1)[0] 
        sample_states = [r(n)]
        for i in range(20):
            sample_states.append(np.random.choice(n, size=1, p=T[sample_states[-1]])[0])

        sample_obs = [r(v) for _ in range(len(sample_states))]
        for i in range(1, len(sample_states)):
            sample_obs[i] = np.random.choice(v, size=1, p=E[sample_states[i-1]])[0]

        sample_obs = np.eye(v)[sample_obs]

        return torch.tensor(sample_obs), torch.tensor(sample_states)

    def collate_fn(self, batch: list[Sentence]) -> dict[str, Any]:
        features = [sentence.tokens for sentence in batch]
        features = pad_sequence(features, batch_first=True, padding_value=0)
        label_indexed = [item.labels for item in batch]
        label_indexed = pad_sequence(label_indexed, batch_first=True, padding_value=-100)

        bias_ones = torch.ones(features.shape[0], features.shape[1], 1)
        
        batch_collated = {
            "features": torch.cat([features, bias_ones], dim=2).float(),
            "labels": label_indexed,
            "sentence": batch,
        }

        if self.loo_mode:
            idxs = [item.id for item in batch]
            batch_masked_tokens = [self.masked_tokens[idx] for idx in idxs]
            batch_mask = torch.zeros_like(label_indexed, dtype=torch.bool)
            for i, tokens in enumerate(batch_masked_tokens):
                batch_mask[i, tokens] = True

            batch_collated["tokens_to_remove"] = batch_mask

        return batch_collated