import json
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import pytorch_lightning as pl
import torch
from conf import base_dir
from IPython.core.display import HTML, display
from matplotlib import cm as colormaps
from matplotlib import colors
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Sentence:
    id: str
    tokens: list[str]
    labels: Union[list[str], list[list[str]]]
    input_ids: Optional[torch.Tensor] = None
    wp_to_word_ids: Optional[torch.Tensor] = None
    metadata: Optional[dict[str, Any]] = None

    def deepcopy_without_tensors(self) -> "Sentence":
        return Sentence(
            id=deepcopy(self.id),
            tokens=deepcopy(self.tokens),
            labels=deepcopy(self.labels),
            metadata=deepcopy(self.metadata),
        )

    def deepcopy_with_tensors(self) -> "Sentence":
        sentence = self.deepcopy_without_tensors()
        sentence.input_ids = self.input_ids
        sentence.wp_to_word_ids = self.wp_to_word_ids
        return sentence


class NERDataModule(pl.LightningDataModule):
    def __init__(
        self,
        splits: dict[str, Union[str, list[str]]],
        label_list: Optional[list[str]] = None,
        batch_size: int = 8,
        transformer: str = "distilbert-base-cased",
        remap_labels: Optional[dict[str, dict[str, str]]] = None,
    ):
        super().__init__()
        self._splits = splits
        self._tokenizer = AutoTokenizer.from_pretrained(transformer)
        self._transformer = transformer
        self._batch_size = batch_size

        self._label_list = label_list
        self._remap_labels = remap_labels or {}

        # Need to be set later
        self._dataset: dict[str, list[Sentence]] = {}
        self._train_dataset = None
        self._validation_splits = None

    def __getitem__(self, idx):
        return self._dataset[idx]

    def __setitem__(self, key, value):
        self._dataset[key] = value

    def setup(self, stage=None):
        label_set = set()

        for split, filepath in self._splits.items():
            split_remap_labels = self._remap_labels.get(split, self._remap_labels.get("_", {}))
            if isinstance(filepath, str):
                filepath = os.path.join(base_dir, os.path.normpath(filepath))
                data = [json.loads(x) for x in open(filepath)]
            elif isinstance(filepath, (list, tuple)):
                data = [
                    json.loads(x) for f in filepath for x in open(os.path.join(base_dir, os.path.normpath(f)))
                ]

            tokenized_sentences = []
            bad_sentences = 0

            for instance in tqdm(data):
                sentence = self.dict_to_sentence(instance, split_remap_labels)
                label_set = label_set.union(set(sentence.labels))
                sentence = self.tokenize(sentence)
                if sentence is not None:
                    tokenized_sentences.append(sentence)
                else:
                    bad_sentences += 1

            print(f"{bad_sentences} bad sentences in {split}; Possible unicode issues if > 0")
            self._dataset[split] = tokenized_sentences

        if self._label_list is None:
            label_list = list(label_set)
            i = label_list.index("O")
            label_list[0], label_list[i] = label_list[i], label_list[0]
            self._label_list = label_list[:1] + sorted(label_list[1:])

    @staticmethod
    def dict_to_sentence(instance: dict, remap_labels: Optional[dict] = None) -> Sentence:
        labels = instance["ner_tags"] if "ner_tags" in instance else instance["labels"]
        return Sentence(
            id=instance["id"],
            tokens=instance["tokens"],
            labels=[
                remap_labels.get(label, label) if remap_labels is not None else label for label in labels
            ],
            metadata=instance.get("metadata", None),
        )

    def switch_label_type(self, labels):
        if isinstance(labels[0], str):
            return [self._label_list.index(x) for x in labels]
        return [self._label_list[x] for x in labels]

    def apply_transform(
        self,
        sentences: list[Sentence],
        transform: Callable[[Sentence], Optional[Sentence]],
        retokenize: bool = False,
    ) -> list[Sentence]:
        transformed_sentences = []
        for sentence in tqdm(sentences):
            sentence = transform(sentence)
            if sentence is not None:
                sentence = self.tokenize(sentence) if retokenize else sentence
                assert sentence is not None, "Transform makes sentence no longer tokenizable"
                transformed_sentences.append(sentence)

        return transformed_sentences

    def set_train_split(self, train_split: str):
        self._train_split = train_split
        self._train_dataset = self._dataset[train_split]

    def set_validation_splits(self, validation_splits: list[str]):
        self._validation_splits = validation_splits

    def tokenize(self, sentence: Sentence) -> Optional[Sentence]:
        assert len(sentence.tokens) == len(sentence.labels)  # WARNING: Does not work with multilabel sets
        try:
            indexed_tokens = self._tokenizer(
                [sentence.tokens],
                is_split_into_words=True,
                return_tensors="pt",
                padding=True,
            )
            wordpiece_scatter = [0] * indexed_tokens.input_ids[0].shape[0]
            for j in range(len(sentence.tokens)):
                span = indexed_tokens.word_to_tokens(0, j, 0)
                for k in range(span.start, span.end):
                    wordpiece_scatter[k] = j + 1

            wordpiece_scatter = torch.tensor(wordpiece_scatter)
            sentence.input_ids = indexed_tokens.input_ids[0]
            sentence.wp_to_word_ids = wordpiece_scatter

            return sentence
        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            return None

    def collate_fn(self, batch: list[Sentence]) -> dict[str, Any]:
        input_ids = [item.input_ids for item in batch]
        pad_token = self._tokenizer.vocab[self._tokenizer.pad_token]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token)

        attention_mask = (input_ids != self._tokenizer.vocab[self._tokenizer.pad_token]).long()

        wp_to_word_ids = [item.wp_to_word_ids for item in batch]
        wp_to_word_ids = pad_sequence(wp_to_word_ids, batch_first=True, padding_value=0)

        if isinstance(batch[0].labels[0], str):
            label_indexed = [torch.tensor([self._label_list.index(x) for x in item.labels]) for item in batch]
            label_indexed = pad_sequence(label_indexed, batch_first=True, padding_value=-100)
        else:
            N = max([len(item.labels) for item in batch])
            V = max([len(item.labels[0]) for item in batch])
            pad = lambda x: torch.nn.functional.pad(x, (0, V - x.shape[1], 0, N - x.shape[0]), value=-100)
            label_indexed = [
                pad(torch.tensor([[self._label_list.index(x) for x in ll] for ll in item.labels])).unsqueeze(
                    0
                )
                for item in batch
            ]

            label_indexed = torch.cat(label_indexed, dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "wp_to_word_ids": wp_to_word_ids,
            "labels": label_indexed,
            "sentence": batch,
        }

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            collate_fn=self.collate_fn,
            num_workers=0,
            shuffle=True,
        )

    def val_dataloader(self):
        return [
            torch.utils.data.DataLoader(
                self._dataset[split],
                batch_size=self._batch_size,
                collate_fn=self.collate_fn,
                num_workers=0,
                shuffle=False,
            )
            for split in self._validation_splits
        ]

    def dataloader_from_split(self, split: str, batch_size: int = None):
        return self.dataloader_from_dataset(self._dataset[split], batch_size)

    def dataloader_from_sentences(self, sentences: list[Sentence], batch_size: int = None):
        sentences = [self.tokenize(sentence) for sentence in sentences]
        return self.dataloader_from_dataset(sentences, batch_size)

    def dataloader_from_dataset(self, dataset: torch.utils.data.Dataset, batch_size: int = None):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size or self._batch_size,
            collate_fn=self.collate_fn,
            num_workers=0,
            shuffle=False,
        )

    def _visualize(self, tokens, labels):
        html = []
        cmap = colormaps.get_cmap("Pastel2")
        get_color = lambda l: colors.rgb2hex(cmap(l - 1)) if l > 0 else "#FFFFFF"

        for i, (t, l) in enumerate(zip(tokens, labels)):
            html.append(f'<span style="background-color:{get_color(l)};">{t}</span> ')

        return "".join(html)

    def visualize_simple(self, example: Sentence, ipython: bool = True) -> Optional[str]:
        html = self._visualize(example.tokens, [self._label_list.index(x) for x in example.labels])
        return "".join(html) if not ipython else display(HTML("".join(html)))

    def visualize_bio(self, example: Sentence, ipython: bool = True) -> Optional[str]:
        html = []
        classes = list(dict.fromkeys([x.replace("B-", "").replace("I-", "") for x in self._label_list]))
        assert "O" == classes[0]
        labels_list = example.labels if isinstance(example.labels[0], list) else [example.labels]

        html.append(self._visualize(classes, list(range(len(classes)))))
        html.append("<br>")
        for labels in labels_list:
            labels = [classes.index(l.split("-")[1] if l != "O" else "O") for l in labels]
            html.append(self._visualize(example.tokens, labels))
            html.append("<br>")

        return "".join(html) if not ipython else display(HTML("".join(html)))

    def save_sentence_to_file(self, sentences: list[Sentence], filename: str):
        filename = os.path.join(base_dir, os.path.normpath(filename))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": sentence.id,
                                "tokens": sentence.tokens,
                                "ner_tags": sentence.labels,
                                "metadata": sentence.metadata,
                            }
                        )
                        for sentence in sentences
                    ]
                )
            )

    @staticmethod
    def combine_to_docs(sentences: list[Sentence], key, order, preserve_sentences: bool = False) -> dict[str, Sentence]:
        documents = {}
        for sentence in sentences:
            skey = key(sentence)
            sorder = order(sentence)
            if skey not in documents:
                documents[skey] = []

            documents[skey].append((sentence, sorder))

        for k in documents:
            documents[k] = tuple(zip(*sorted(documents[k], key=lambda x: x[1])))[0]
            if not preserve_sentences:
                documents[k] = Sentence(
                    id=k,
                    tokens=[t for sent in documents[k] for t in sent.tokens],
                    labels=[t for sent in documents[k] for t in sent.labels],
                    metadata=None,
                )

        return documents
