import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from ner_influence.linear_modelling.datamodule import LinearNERDataModule
from ner_influence.linear_modelling.metrics import NERMetrics
from ner_influence.linear_modelling.model import LinearNERModel
from ner_influence.linear_modelling.pytorch_lbfgs import LBFGS
from ner_influence.scaffolding import BaseNERScaffolding
from torch.nn.utils import _stateless
from tqdm import tqdm

import sys
from collections import defaultdict


class LinearNERScaffolding(BaseNERScaffolding):
    def __init__(self, data: LinearNERDataModule, seed: int = 2022, reg_strength: float = 1.0):
        self.data = data
        pl.seed_everything(seed)
        self.model = LinearNERModel(
            label_list=data._label_list,
            feature_size=data.feature_size,
            embedding_size=data.embedding_size,
        )
        self.reg_strength = reg_strength

    def sample_data(self, train_size, val_size):
        rs = np.random.RandomState(seed=2022)
        small_train = [x for x in self.data["train"] if len(x.tokens) < 25]
        sampled_train = rs.choice(small_train, size=train_size, replace=False)
        self.data["sampled_train"] = sampled_train

        small_validation = [x for x in self.data["validation"] if len(x.tokens) < 25]
        self.data["sampled_validation"] = rs.choice(small_validation, size=val_size, replace=False)

        self.data.set_train_split("sampled_train")
        self.data.set_validation_splits(["sampled_validation"])

    def print_loss_and_grad(self):
        N = len(self.data["sampled_train"])
        self.model.zero_grad(set_to_none=True)
        loss_val = 0.0
        for batch in self.data.dataloader_from_split("sampled_train"):
            loss = self.model.training_step(batch, 0) / N
            loss.backward()
            loss_val += loss.item()

        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad += p.data * self.reg_strength / N
                    loss_val += (p.data ** 2).sum() * self.reg_strength / (2 * N)

        grads = {
            k: v.grad.norm().item()
            for k, v in dict(self.model.named_parameters()).items()
            if v.grad is not None
        }
        print(f"{loss_val} {grads}")

    def closure(self):
        N = len(self.data["sampled_train"])
        self.model.zero_grad(set_to_none=True)
        loss_val = 0.0
        for batch in self.data.dataloader_from_split("sampled_train"):
            loss = self.model.training_step(batch, 0) / N
            loss.backward()
            loss_val += loss.item()

        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad += p.data * self.reg_strength / N
                    loss_val += (p.data ** 2).sum() * self.reg_strength / (2 * N)

        return loss_val

    def train(self, tol=1e-6):
        optim = LBFGS(
            self.model.parameters(),
            lr=0.1,
            max_iter=1000,
            history_size=100,
            line_search_fn=None,
            tolerance_grad=tol,
        )
        optim.step(self.closure)
        self.print_loss_and_grad()

    def evaluate(self):
        outputs = []
        for batch in self.data.dataloader_from_split("sampled_validation"):
            outputs.append(self.model.predict_step(batch))
        outputs = [item for batch in outputs for item in batch]

        return pd.DataFrame(NERMetrics.exact_match_metrics(outputs, self.data._label_list))

    def hessian(self):
        self.model.double()
        overall_hessian = None

        for batch in tqdm(self.data.dataloader_from_split("sampled_train", batch_size=200)):
            self.model.zero_grad()

            def loss(*params):
                outputs = _stateless.functional_call(self.model, {n: p for n, p in zip(names, params)}, batch)
                return outputs["loss"] / len(self.data["sampled_train"])

            names = list(n for n, p in self.model.named_parameters() if p.requires_grad)
            params = tuple([p for _, p in self.model.named_parameters() if p.requires_grad])
            hess = torch.autograd.functional.hessian(loss, params)

            if overall_hessian is None:
                overall_hessian = [[x.detach().cpu().data.numpy() for x in y] for y in hess]
            else:
                for i in range(len(params)):
                    for j in range(len(params)):
                        overall_hessian[i][j] += hess[i][j].detach().cpu().data.numpy()

        total_params = sum(p.numel() for p in params)
        total_hessian = np.zeros((total_params, total_params))
        i_offset = 0
        for i in range(len(params)):
            j_offset = 0
            for j in range(len(params)):
                total_hessian[
                    i_offset : i_offset + params[i].numel(), j_offset : j_offset + params[j].numel()
                ] = overall_hessian[i][j].reshape(params[i].numel(), params[j].numel())

                j_offset += params[j].numel()

            i_offset += params[i].numel()

        self.total_hessian = total_hessian + np.eye(total_params) * (
            self.reg_strength / len(self.data["sampled_train"])
        )
        self.total_hessian_inv = np.linalg.inv(self.total_hessian)

        assert np.allclose(self.total_hessian, self.total_hessian.T)
        assert np.allclose(self.total_hessian_inv, self.total_hessian_inv.T)

    def gradient(self, prediction, separate_gradients=False):
        gold_labels = np.eye(len(self.data._label_list))[prediction["gold_labels"]]
        predicted_probs = prediction["conditional_probs"]
        assert gold_labels.shape == predicted_probs.shape

        error = gold_labels - predicted_probs
        gradient_W = error[:, :, None] * prediction["features"][:, None, :]

        gradient_T = np.zeros((gold_labels.shape[0], len(self.data._label_list), len(self.data._label_list)))
        gradient_T[:-1] += error[:-1, :, None] * gold_labels[1:, None, :]  # T[y_t, y_t+1]
        gradient_T[1:] += gold_labels[:-1, :, None] * error[1:, None, :]  # T[y_t-1, y_t]

        L = gradient_W.shape[0]
        gradient_full = np.concatenate([gradient_W.reshape(L, -1), gradient_T.reshape(L, -1)], axis=-1)

        if separate_gradients:
            return gradient_W, gradient_T
        else:
            return gradient_full

    def generate_all_predictions(self, split: str):
        outputs = []
        for batch in self.data.dataloader_from_split(split):
            outputs.append(self.model.predict_step_for_gradient(batch))
        outputs = [item for batch in outputs for item in batch]
        return outputs

    def compute_influence(self, val_idx, val_token_idx, k=10):
        token_grad = self.val_gradients[val_idx][val_token_idx]
        token_influence = [
            (token_grad @ self.total_hessian_inv @ g.T) / len(self.train_gradients)
            for g in self.train_gradients
        ]
        token_sorted = sorted(
            [(i, j) for i, g in enumerate(token_influence) for j in range(len(g))],
            key=lambda a: np.abs(token_influence[a[0]][a[1]]),
        )
        return token_sorted[-k:]

    def compute_specific_influence(self, val_idx, val_token_idx, train_idx, train_token_idx):
        influence = (
            self.val_gradients[val_idx][val_token_idx]
            @ self.total_hessian_inv
            @ self.train_gradients[train_idx][train_token_idx]
        ) / len(self.train_gradients)

        return influence

    def retrain(self, train_idx, train_token_idx, val_idx, val_token_idx):
        retrain_scaff = LinearNERScaffolding(self.data, seed=2022, reg_strength=1.0)
        self.data.loo_mode = True
        self.data.masked_tokens = defaultdict(list)
        train_id = self.data["sampled_train"][train_idx].id
        self.data.masked_tokens[train_id] = [train_token_idx]
        print(self.data.masked_tokens)
        sys.stdout.flush()
        retrain_scaff.train(5e-6)
        self.data.loo_mode = False
        sys.stdout.flush()

        self.data["tmp_validation"] = [self.data["sampled_validation"][val_idx]]
        pred = retrain_scaff.generate_all_predictions("tmp_validation")
        assert len(pred) == 1
        new_loss = pred[0]["conditional_loss"][val_token_idx]
        old_loss = self.val_predictions[val_idx]["conditional_loss"][val_token_idx]
        influence = (
            self.val_gradients[val_idx][val_token_idx]
            @ self.total_hessian_inv
            @ self.train_gradients[train_idx][train_token_idx]
        ) / len(self.train_gradients)

        print(new_loss, old_loss, new_loss - old_loss, influence)
        sys.stdout.flush()

        del self.data.masked_tokens

        return (train_idx, train_token_idx, val_idx, val_token_idx, new_loss, old_loss, new_loss - old_loss, influence)

    def open_file(self, filename):
        self.output_file = open(filename, "w")

    def store(self, values):
        values = "\t".join([str(x) for x in values])
        self.output_file.write(values + "\n")
        self.output_file.flush()

    def close_file(self):
        self.output_file.close()
