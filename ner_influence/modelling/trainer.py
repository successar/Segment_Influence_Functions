import json
import os
import shutil
from typing import Optional

from ner_influence.modelling.utilities import NumpyEncoder
import pandas as pd
import pytorch_lightning as pl
import torch
from ner_influence.modelling.datamodule import NERDataModule
from ner_influence.modelling.metrics import NERMetrics
from ner_influence.modelling.model import NERModel
from pytorch_lightning.callbacks import ModelCheckpoint


def train_ner_model(
    data: NERDataModule,
    model_path: str,
    use_crf: bool,
    val_metric: str = "val_loss",
    seed: int = "42",
):
    if "BSIZE" in os.environ:
        batch_size = int(os.environ["BSIZE"])
        data._batch_size = batch_size

    pl.seed_everything(seed)
    model = NERModel(
        transformer=data._transformer,
        label_list=data._label_list,
        use_crf=use_crf,
    )

    checkpoint_callback_0 = ModelCheckpoint(monitor=val_metric, mode="min", filename=f"best")

    trainer = pl.Trainer(
        gpus=[0],
        min_epochs=5,
        max_epochs=15,
        default_root_dir=model_path,
        gradient_clip_val=10.0,
        callbacks=[checkpoint_callback_0],
    )

    trainer.fit(model, data)
    shutil.copyfile(checkpoint_callback_0.best_model_path, f"{model_path}/best.ckpt")
    shutil.rmtree(os.path.dirname(checkpoint_callback_0.best_model_path))

    return torch.load(f"{model_path}/best.ckpt")


def evaluate_ner_model(
    data: NERDataModule,
    model_path: str,
    split: str,
    pandas: bool = True,
    melt: bool = False,
    metrics: str = "exact",
    save: bool = False,
):
    model = NERModel.load_from_checkpoint(f"{model_path}/best.ckpt")

    trainer = pl.Trainer(
        gpus=[0],
        default_root_dir=f"{model_path}/testing",
    )

    with torch.no_grad():
        model.return_embeddings = False
        outputs = trainer.predict(model, data.dataloader_from_split(split), return_predictions=True)
        outputs = [item for batch in outputs for item in batch]
        if metrics == "exact":
            metrics = NERMetrics.exact_match_metrics(outputs, data._label_list)
        elif metrics == "token":
            metrics = NERMetrics.token_metrics(outputs, data._label_list)
        else:
            raise NotImplementedError

    shutil.rmtree(f"{model_path}/testing")
    if save:
        json.dump(metrics, open(f"{model_path}/metrics_{split}.jsonl", "w"), cls=NumpyEncoder)

    if pandas:
        metrics = pd.DataFrame(metrics)
        if melt:
            metrics = (
                metrics.T.melt(ignore_index=False)
                .reset_index()
                .rename(columns={"index": "type", "variable": "metric"})
            )

    return metrics


def melt_metrics(metrics):
    return (
        pd.DataFrame(metrics)
        .T.melt(ignore_index=False)
        .reset_index()
        .rename(columns={"index": "type", "variable": "metric"})
    )


def predict_ner_model(data: NERDataModule, model_path: str, split: str):
    model = NERModel.load_from_checkpoint(f"{model_path}/best.ckpt")

    trainer = pl.Trainer(
        gpus=[0],
        default_root_dir=f"{model_path}/testing",
    )

    with torch.no_grad():
        outputs = trainer.predict(model, data.dataloader_from_split(split), return_predictions=True)
        outputs = [item for batch in outputs for item in batch]
        return outputs
