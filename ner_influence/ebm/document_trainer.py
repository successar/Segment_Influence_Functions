from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--use-crf", action="store_true", default=False)
parser.add_argument("--seed", type=int)
args = parser.parse_args()

from ner_influence.modelling.trainer import train_ner_model
from ner_influence.ebm import load_datamodule
from conf import add_base_dir

data = load_datamodule(transformer="google/bigbird-roberta-base")
data._batch_size = 9

conll_key = lambda x: x.id.rsplit("_", 1)[0]
conll_order = lambda x: int(x.id.rsplit("_", 1)[1])

for split in ["train", "validation", "test"]:
    docs = data.combine_to_docs(data[split], key=conll_key, order=conll_order)
    data[f"{split}_docs"] = data.apply_transform([doc for doc in docs.values()] , transform=lambda x:x, retokenize=True)

data.set_train_split("train_docs")
data.set_validation_splits(["validation_docs"])

model_path = add_base_dir(f"./outputs/ebm_docs/simple_trainer/crf:{args.use_crf};seed:{args.seed}")
train_ner_model(data, model_path, args.use_crf, seed=args.seed)
