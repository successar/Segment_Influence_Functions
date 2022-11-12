from functools import partial
from typing import Iterable
import torch
from ner_influence.scaffolding import BaseNERScaffolding, Instance

from ner_influence.modelling.datamodule import NERDataModule
from ner_influence.modelling.model import NERModel
from tqdm import tqdm
from scipy.special import softmax


class NERTransformerScaffolding(BaseNERScaffolding):
    def __init__(
        self,
        data: NERDataModule,
        model_dir: str,
        save_outputs: bool = False
    ):
        self._data = data

        ckpt_path = f"{model_dir}/best.ckpt"
        self.output_dir = model_dir

        self.model: NERModel = NERModel.load_from_checkpoint(ckpt_path).cuda()
        assert self.model.label_list == self._data._label_list
        assert self.model._num_labels == len(self._data._label_list)

        self.class_names = self._data._label_list
        self.feature_vector_size = self.model.feature_vector_size
        self.num_classes = self.model._num_labels

        self._save_outputs = save_outputs
        self._keep_outputs = {}

        self.conditional_prob_function = self.model.conditional_probs()
        self.token_conditional_prob_function = self.model.conditional_probs_token()

    def token_conditional_prob(self, logits, token_idx, prev_label, next_label):
        return self.token_conditional_prob_function(logits[token_idx], prev_label, next_label)

    def generate_outputs(self, split: str, with_feature_vectors: bool) -> Iterable[Instance]:   
        dataloader = self.get_dataloader(split)
        with torch.no_grad():
            outputs = []
            for batch in tqdm(iter(dataloader), total=len(dataloader)):
                batch_outputs = self.model.move_and_predict_step(batch)
                for instance in batch_outputs:
                    conditional_probs = partial(self.conditional_prob_function, logits=instance["logits"])
                    token_conditional_probs = partial(self.token_conditional_prob, logits=instance["logits"])
                    output = Instance(
                        id=instance["sentence"].id,
                        loss=instance["loss"],
                        tokens=instance["sentence"].tokens,
                        conditional_probs=conditional_probs,
                        token_conditional_probs=token_conditional_probs,
                        marginal_probs=softmax(instance["marginal_logits"], axis=-1),
                        predicted_labels=instance["predicted_labels"],
                        gold_labels=instance["gold_labels"],
                        metadata=instance["sentence"].metadata,
                        token_feature_vectors=instance["token_embeddings"] if with_feature_vectors else None,
                    )
                    outputs.append(output)

                    yield output

            if self._save_outputs:
                self._keep_outputs[split] = outputs

    def get_outputs(self, split: str, with_feature_vectors: bool) -> Iterable[Instance]:
        if split in self._keep_outputs:
            return self._keep_outputs[split]
        else :
            return self.generate_outputs(split, with_feature_vectors)

    def get_dataloader(self, split: str) -> torch.utils.data.DataLoader:
        if split in self._data._dataset:
            return self._data.dataloader_from_split(split)
        
        raise KeyError("Split does not exist")
