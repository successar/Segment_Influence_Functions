from ner_influence.modelling.datamodule import NERDataModule
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ontonotes_map = {
    "CARDINAL": "O",
    "DATE": "O",
    "EVENT": "O",
    "FAC": "ORG",
    "GPE": "LOC",
    "LANGUAGE": "O",
    "LAW": "O",
    "MONEY": "O",
    "NORP": "O",
    "ORDINAL": "O",
    "PERCENT": "O",
    "PERSON": "PER",
    "PRODUCT": "O",
    "QUANTITY": "O",
    "TIME": "O",
    "WORK_OF_ART": "O",
}

ontonotes_map_new = {
    f"B-{k}": v if v == "O" else f"B-{v}" for k, v in ontonotes_map.items()
}
ontonotes_map_new = {
    **{f"I-{k}": v if v == "O" else f"I-{v}" for k, v in ontonotes_map.items()},
    **ontonotes_map_new,
}

conll_map = {"B-MISC": "O", "I-MISC": "O"}

remaps = {
    "conll" : conll_map,
    "ontonotes": ontonotes_map_new, 
    "twitter": conll_map,
    "docred": conll_map,
    "conll-corrected": conll_map,
}


def load_datamodule(
    sets: set[str] = None,
    extra: dict[str, str] = None,
    transformer: str = "distilbert-base-cased",
):
    splits = {
        "conll_train": "data/conll/train.jsonl",
        "conll_validation": "data/conll/validation.jsonl",
        "conll_test": "data/conll/test.jsonl",
        "conll-corrected_train": "data/conll_corrected/train_corrected.jsonl",
        "conll-corrected_validation": "data/conll_corrected/validation_corrected.jsonl",
        "conll-corrected_test": "data/conll_corrected/test_corrected.jsonl",
        "twitter_train": "data/twitter/train.jsonl",
        "twitter_validation": "data/twitter/validation.jsonl",
        "twitter_test": "data/twitter/test.jsonl",
        "docred_train": "data/docred/train_annotated.jsonl",
        "docred_validation": "data/docred/dev.jsonl",
        "docred_test": "data/docred/test.jsonl",
        "ontonotes_train": ["data/ontonotes/bc.train.jsonl", "data/ontonotes/bn.train.jsonl", "data/ontonotes/nw.train.jsonl"],
        "ontonotes_validation": ["data/ontonotes/bc.validation.jsonl", "data/ontonotes/bn.validation.jsonl", "data/ontonotes/nw.validation.jsonl"],
        "ontonotes_test": ["data/ontonotes/bc.test.jsonl", "data/ontonotes/bn.test.jsonl", "data/ontonotes/nw.test.jsonl"]
    }
    if sets is not None:
        splits = {k: v for k, v in splits.items() if k.split("_")[0] in sets}

    if extra is not None:
        splits = {**splits, **extra}

    validation_splits = [
        "conll_validation",
        "twitter_validation",
        "docred_validation",
        "conll-docs_validation",
        "ontonotes_validation",
        "conll-corrected_validation",
    ]
    validation_splits = [x for x in validation_splits if x in splits]

    dm_remaps = {}
    for k, v in splits.items():
        t = next((s for s in remaps if s in k), None)
        if t is not None:
            dm_remaps[k] = remaps[t]

    print(dm_remaps)

    data = NERDataModule(
        splits=splits,
        label_list=["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"],
        remap_labels=dm_remaps,
        transformer=transformer,
        batch_size=32,
    )
    data.setup()
    data.set_validation_splits(validation_splits)
    return data
