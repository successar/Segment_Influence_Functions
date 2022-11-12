import warnings
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from ner_influence.linear_modelling.conditional_random_field import (
    ConditionalRandomField, allowed_transitions)

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)  # Because pytorch lightning has crappy batch size finder

warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers`.*")

class LinearNERModel(pl.LightningModule):
    def __init__(self, label_list: list[str], feature_size: int, embedding_size: int):
        super().__init__()
        self._num_labels = len(label_list)

        self._feature_to_logit = torch.nn.Linear(
            feature_size + embedding_size + 1, self._num_labels, bias=False
        )
        self.feature_vector_size = feature_size + embedding_size + 1

        if isinstance(label_list, str):
            allowed_transitions_bio = allowed_transitions("BIO", dict(zip(list(range(self._num_labels)), label_list)))
        else:
            allowed_transitions_bio = None

        self._crf = ConditionalRandomField(
            self._num_labels,
            constraints=allowed_transitions_bio,
            include_start_end_transitions=False,
        )

        self.label_list = label_list

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        logits = self._feature_to_logit(batch["features"].type(self._feature_to_logit.weight.dtype))
        if "labels" in batch:
            return {"logits": logits, "loss": self.get_loss(logits, batch["labels"])}

        return {"logits": logits}

    def get_loss(self, logits, labels):
        mask = labels != -100
        labels = torch.where(mask, labels, 0)
        loss = -self._crf.forward(logits, labels, mask)
        return loss

    def get_conditional_ll(self, logits, labels):
        mask = labels != -100
        labels = torch.where(mask, labels, 0)
        conditional_ll = self._crf.token_conditional_likelihood(logits, labels)
        return conditional_ll

    def training_step(self, batch, batch_index):
        outputs = self.forward(batch)
        loss = self.get_loss(outputs["logits"], batch["labels"])
        if "tokens_to_remove" in batch:
            conditional_ll = self.get_conditional_ll(outputs["logits"], batch["labels"])
            conditional_loss = torch.nn.NLLLoss(reduction="none")(
                conditional_ll.transpose(1, 2), batch["labels"]
            )
            removal_ll = torch.masked_select(conditional_loss, batch["tokens_to_remove"])
            loss = loss - removal_ll.sum()
        return loss

    def predict_step_for_gradient(self, batch):
        outputs = self.forward(batch)
        conditional_ll = self.get_conditional_ll(outputs["logits"], batch["labels"])
        conditional_probs = torch.exp(conditional_ll)
        mask = batch["labels"] != -100
        predicted_labels = self._crf.viterbi_tags(outputs["logits"].detach(), mask)
        predicted_labels, scores = tuple(zip(*predicted_labels))
        gold_labels = batch["labels"]

        conditional_loss = torch.nn.NLLLoss(reduction="none")(conditional_ll.transpose(1, 2), gold_labels)

        instance_predictions = []

        for i in range(len(gold_labels)):
            selector = gold_labels[i] != -100
            instance_predictions.append(
                {
                    "features": batch["features"][i][selector].cpu().data.numpy(),
                    "gold_labels": gold_labels[i][selector].cpu().data.numpy(),
                    "predicted_labels": np.array(predicted_labels[i]),
                    "conditional_probs": conditional_probs[i][selector].cpu().data.numpy(),
                    "conditional_loss": conditional_loss[i][selector].cpu().data.numpy(),
                }
            )

        return instance_predictions
