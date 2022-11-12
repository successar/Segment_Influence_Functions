import json
import os
import shutil

from sklearn.utils import check_array

from ner_influence.modelling.utilities import NumpyEncoder
import pandas as pd
import pytorch_lightning as pl
import torch
from ner_influence.linear_modelling.datamodule import LinearNERDataModule
from ner_influence.modelling.metrics import NERMetrics
from ner_influence.linear_modelling.model import LinearNERModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import RichProgressBar



def train_ner_model(
    data: LinearNERDataModule,
    model_path: str,
    use_embeddings: bool,
    val_metric: str = "val_loss",
    seed: int = "42",
):
    if "BSIZE" in os.environ:
        batch_size = int(os.environ["BSIZE"])
        data._batch_size = batch_size

    pl.seed_everything(seed)
    model = LinearNERModel(
        label_list=data._label_list,
        feature_size=data.feature_size,
        embedding_size=data.embedding_size,
        use_embeddings=use_embeddings
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        min_epochs=5,
        max_epochs=60,
        default_root_dir=model_path,
        gradient_clip_val=10.0,
        enable_checkpointing=False,
        weights_save_path=f"{model_path}",
        enable_progress_bar=False,
        logger=False
    )

    trainer.fit(model, data)

    return model


def evaluate_ner_model(
    data: LinearNERDataModule,
    model_path: str,
    split: str,
    pandas: bool = True,
    melt: bool = False,
    metrics: str = "exact",
    save: bool = False,
):
    model = LinearNERModel.load_from_checkpoint(f"{model_path}/best.ckpt")

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


def predict_ner_model(data: LinearNERDataModule, model_path: str, split: str):
    model = LinearNERModel.load_from_checkpoint(f"{model_path}/best.ckpt")

    trainer = pl.Trainer(
        gpus=[0],
        default_root_dir=f"{model_path}/testing",
    )

    with torch.no_grad():
        outputs = trainer.predict(model, data.dataloader_from_split(split), return_predictions=True)
        outputs = [item for batch in outputs for item in batch]
        return outputs
