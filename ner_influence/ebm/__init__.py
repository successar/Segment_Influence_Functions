from ner_influence.modelling.datamodule import NERDataModule
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformer: str = "distilbert-base-uncased"


def load_datamodule(transformer=transformer):
    splits = {
        "train": "data/ebm_nlp/train.jsonl",
        "validation": "data/ebm_nlp/dev.jsonl",
        "test": "data/ebm_nlp/test.jsonl",
    }

    data = NERDataModule(
        splits=splits, label_list=["O", "POP", "INT", "OUT"], transformer=transformer, remap_labels={}
    )
    data.setup()
    data.set_validation_splits(["validation"])
    return data
