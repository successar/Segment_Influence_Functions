from typing import Any, Optional, TypedDict

import numpy as np
import pytorch_lightning as pl
import torch
from ner_influence.modelling.conditional_random_field import ConditionalRandomField, allowed_transitions
from scipy.special import softmax
from transformers import AutoModel, T5EncoderModel
from ner_influence.modelling.datamodule import Sentence

import warnings

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)  # Because pytorch lightning has crappy batch size finder

warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers`.*")


class Prediction(TypedDict, total=False):
    loss: float
    sentence: Sentence
    gold_labels: list[int]  # (T,)
    predicted_labels: list[int]  # (T,)
    logits: np.ndarray  # (T, C)
    token_embeddings: np.ndarray  # (T, E)
    marginal_logits: Optional[np.ndarray]


class NERModel(pl.LightningModule):
    def __init__(self, label_list: list[str], transformer: str, use_crf: bool):
        super().__init__()
        self._num_labels = len(label_list)
        self._bert = (
            AutoModel.from_pretrained(transformer)
            if "bigbird" not in transformer
            else AutoModel.from_pretrained(transformer, attention_type="original_full")
        )
        self._bert.requires_grad = True
        self._dropout = torch.nn.Dropout(p=0.3)
        self._classifier = torch.nn.Linear(self._bert.config.hidden_size, self._num_labels)
        self.feature_vector_size = self._bert.config.hidden_size + 1

        self._use_crf = use_crf
        if use_crf:
            self._crf = ConditionalRandomField(
                self._num_labels,
                constraints=allowed_transitions("BIO", dict(zip(list(range(self._num_labels)), label_list))),
                include_start_end_transitions=False,
            )

        self.return_embeddings = False
        self.label_list = label_list

        self.save_hyperparameters("label_list", "transformer", "use_crf")

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        outputs = self._bert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
        )
        wp_embeddings = outputs.last_hidden_state
        B, Lp, E = wp_embeddings.shape
        L = torch.max(batch["wp_to_word_ids"]).item() + 1
        wp_to_word_ids = batch["wp_to_word_ids"].unsqueeze(-1).expand(-1, -1, E)
        word_embeddings = (
            torch.zeros((B, L, E)).to(wp_embeddings.device).scatter_add(1, wp_to_word_ids, wp_embeddings)
        )[:, 1:]
        word_lengths = (
            torch.zeros((B, L))
            .to(wp_embeddings.device)
            .scatter_add(1, wp_to_word_ids[:, :, 0], torch.ones((B, Lp)).to(wp_embeddings.device))
        )[:, 1:]

        word_embeddings = word_embeddings / (word_lengths.unsqueeze(-1).clamp(1.0))

        return {
            "embeddings": word_embeddings,
            "logits": self._classifier(self._dropout(word_embeddings)),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5, amsgrad=True)
        return {"optimizer": optimizer}

    def get_loss(self, logits, labels):
        if self._use_crf:
            mask = labels != -100
            labels = torch.where(mask, labels, 0)
            if len(labels.shape) == 3:
                loss = -self._crf.multi_tag_forward(logits, labels, mask)
            else:
                loss = -self._crf.forward(logits, labels, mask)
        else:
            if len(labels.shape) == 3:
                logits = logits.transpose(1, 2)
                mask = labels != -100

                labels = labels.transpose(0, 1)  # (variant, B, L)
                mask = mask.transpose(0, 1)  # (variant, B, L)
                loss = []
                for v in range(mask.shape[0]):
                    v_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
                        logits.transpose(1, 2), labels[v]
                    )
                    loss.append(v_loss.sum(-1).unsqueeze(0))

                loss = torch.cat(loss, dim=0)
                loss = loss.masked_fill((~mask).all(-1), float("-inf"))
                loss = torch.logsumexp(loss, dim=0)
            else:
                loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(logits.transpose(1, 2), labels)
        return loss

    def training_step(self, batch, batch_index):
        outputs = self.forward(batch)
        loss = self.get_loss(outputs["logits"], batch["labels"])

        self.log("train_loss", loss.detach(), prog_bar=True, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int = 0, dataloader_index: int = None):
        outputs = self.forward(batch)
        loss = self.get_loss(outputs["logits"], batch["labels"])

        self.log(f"val_loss", loss.detach(), prog_bar=True)
        return {f"val_loss": loss.detach()}

    def predict_step(self, batch, batch_idx: int = 0, dataloader_index: int = None) -> list[Prediction]:
        outputs = self.forward(batch)
        gold_labels = batch["labels"]
        sentences = batch["sentence"]
        logits = outputs["logits"]
        mask = batch["labels"] != -100

        if self._use_crf:
            predicted_labels = self._crf.viterbi_tags(outputs["logits"].detach(), mask)
            predicted_labels, scores = tuple(zip(*predicted_labels))
            marginal_logits = self._crf.marginal_likelihood(outputs["logits"], mask)
            mask = gold_labels != -100
            loss = -self._crf.forward(
                outputs["logits"], torch.where(mask, gold_labels, 0), mask, reduction=False
            )
        else:
            predicted_labels = outputs["logits"].argmax(-1)
            predicted_labels = [
                predicted_labels[i][gold_labels[i] != -100].cpu().data.numpy()
                for i in range(predicted_labels.shape[0])
            ]
            marginal_logits = logits
            loss = (
                torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
                    logits.transpose(1, 2), gold_labels
                )
                * mask.float()
            ).sum(-1)

        instance_predictions = []

        for i in range(len(gold_labels)):
            instance_logits = logits[i][gold_labels[i] != -100]  # (T, C)
            instance_logits = instance_logits.cpu().data.numpy()

            instance_marginal_logits = marginal_logits[i][gold_labels[i] != -100]  # (T, C)
            instance_marginal_logits = instance_marginal_logits.cpu().data.numpy()

            instance_gold_label = gold_labels[i][gold_labels[i] != -100]  # (T, )
            instance_gold_label = instance_gold_label.cpu().data.numpy()

            embeddings = outputs["embeddings"][i][gold_labels[i] != -100].cpu().data.numpy()
            embeddings = np.hstack([embeddings, np.ones((embeddings.shape[0], 1))])

            instance_predictions.append(
                Prediction(
                    loss=loss[i].item(),
                    logits=instance_logits,
                    predicted_labels=list(predicted_labels[i]),
                    gold_labels=list(instance_gold_label),
                    sentence=sentences[i],
                    token_embeddings=embeddings,
                    marginal_logits=instance_marginal_logits,
                )
            )

        return instance_predictions

    def move_to_device(self, batch):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)

        return batch

    def move_and_predict_step(self, batch) -> list[Prediction]:
        self.eval()
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)

        predictions = self.predict_step(batch)
        return predictions

    def conditional_probs(
        self,
    ):
        if self._use_crf:
            transitions = self._crf.transitions.cpu().data.numpy()

            def crf_conditional_probs(logits, labels):
                logits = np.copy(logits)
                logits[:-1] += transitions[:, labels[1:]].T
                logits[1:] += transitions[labels[:-1], :]
                return softmax(logits, axis=-1)

            return crf_conditional_probs
        else:
            return lambda logits, labels: softmax(logits, axis=-1)

    def conditional_probs_token(
        self,
    ):
        if self._use_crf:
            transitions = self._crf.transitions.cpu().data.numpy()

            def crf_conditional_probs(logits, prev_label, next_label):
                logits = np.copy(logits)
                if prev_label is not None:
                    logits += transitions[prev_label, :]
                if next_label is not None:
                    logits += transitions[:, next_label]
                return softmax(logits)

            return crf_conditional_probs
        else:
            return lambda logits, prev_label, next_label: softmax(logits)
