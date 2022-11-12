import random
from itertools import permutations, product
from typing import Optional

from ner_influence.modelling.metrics import NERMetrics
from ner_influence.modelling.utilities import encode_spans_to_BIO

def sampled_variations(labels, sample_size: Optional[int] = None):
    spans = NERMetrics.labels_to_spans(labels)
    spans = [[(x, y, r) for r in t.split("|")] for x, y, t in spans]
    product_spans = list(product(*spans))
    if sample_size is not None and len(product_spans) > sample_size:
        product_spans = random.sample(product_spans, k=sample_size)
    return [encode_spans_to_BIO(spans, len(labels)) for spans in product_spans]

def precedence_variations(labels, label_set):
    spans = NERMetrics.labels_to_spans(labels)
    spans = [(x, y, t.split("|")) for x, y, t in spans]
    perm_variants = []
    for perm in list(permutations(label_set)):
        perm_spans = [(x, y, sorted(t, key=perm.index)[0]) for x, y, t in spans]
        perm_labels = encode_spans_to_BIO(perm_spans, len(labels))
        perm_variants.append(perm_labels)

    return perm_variants

def get_variations_list(dataset, variant_function, **kwargs):
    return [variant_function(x.labels, **kwargs) for x in dataset]

def get_variations_map(dataset, variant_function, **kwargs):
    return {x.id : variant_function(x.labels, **kwargs) for x in dataset}
