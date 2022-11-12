from ner_influence.linear_modelling.datamodule import LinearNERDataModule
data = LinearNERDataModule(
    splits={
        "train": "data/conll_corrected/train_corrected.jsonl",
        "validation": "data/conll_corrected/validation_corrected.jsonl",
        "test": "data/conll_corrected/test_corrected.jsonl",
    },
    label_list=None,
    batch_size=64,
)
data.setup()


from ner_influence.linear_modelling.scaffolding import LinearNERScaffolding
scaff = LinearNERScaffolding(data, seed=2022, reg_strength=1.0)
scaff.sample_data(1000, 200)
scaff.print_loss_and_grad()
scaff.train()
print(scaff.evaluate())

scaff.hessian()

scaff.train_predictions = scaff.generate_all_predictions("sampled_train")
scaff.val_predictions = scaff.generate_all_predictions("sampled_validation")

s = 0.0
for x in scaff.val_predictions:
    s += (x["gold_labels"] != x["predicted_labels"]).sum()
print(s)

scaff.train_gradients = [scaff.gradient(x) for x in scaff.train_predictions]
scaff.val_gradients = [scaff.gradient(x) for x in scaff.val_predictions]

import numpy as np
mispredictions = [(i, y) for i, x in enumerate(scaff.val_predictions) for y in np.where(x["gold_labels"] != x["predicted_labels"])[0]]

sampled_mispredictions = [
    mispredictions[i]
    for i in np.random.RandomState(seed=2022).choice(len(mispredictions), size=20, replace=False)
]

scaff.open_file("outputs/linear_model_exact_influence.tsv")

for val_idx, val_token_idx in sampled_mispredictions:
    influence_instances = scaff.compute_influence(val_idx, val_token_idx, k=20)

    for train_idx, train_token_idx in influence_instances:
        values = scaff.retrain(train_idx, train_token_idx, val_idx, val_token_idx)
        scaff.store(values)

scaff.close()