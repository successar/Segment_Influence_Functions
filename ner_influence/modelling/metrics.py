from typing import Optional
from seqeval.metrics import classification_report as span_classification_report
from sklearn.metrics import classification_report
from spacy.training.iob_utils import iob_to_biluo, tags_to_entities


class NERMetrics:
    @classmethod
    def token_metrics(cls, predictions: list[dict], label_list: Optional[list[str]] = None):
        predicted_labels = [
            label_list[x] if label_list else x
            for instance in predictions
            for x in instance["predicted_labels"]
        ]
        gold_labels = [
            label_list[x] if label_list else x for instance in predictions for x in instance["gold_labels"]
        ]
        return classification_report(gold_labels, predicted_labels, output_dict=True)

    @staticmethod
    def labels_to_spans(tags: list[str]) -> list[tuple[int, int, str]]:
        return [(start, end + 1, enttype) for enttype, start, end in tags_to_entities(iob_to_biluo(tags))]

    @classmethod
    def exact_match_metrics(cls, predictions: list[dict], label_list: Optional[list[str]] = None):
        predicted_labels = [
            [
                label_list[x] if label_list else x
                for instance in predictions
                for x in instance["predicted_labels"]
            ]
        ]
        gold_labels = [
            [label_list[x] if label_list else x for instance in predictions for x in instance["gold_labels"]]
        ]
        return span_classification_report(gold_labels, predicted_labels, output_dict=True)
