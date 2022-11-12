'''
Contains random functions that I don't know where else to put.
'''
import itertools
from seqeval.scheme import Tokens, IOB2

def bio_to_spans(labels: list[str]) -> list[tuple[int, int, str]]:
    spans = [(e.start, e.end, e.tag) for e in Tokens(labels, IOB2).entities]
    return spans

def io_to_spans(labels: list[str]) -> list[tuple[int, int, str]]:
    element_index = 0 
    spans = []
    for element, occurrences in itertools.groupby(labels, key=lambda x: x):
        count = len(list(occurrences))
        if element != "O":
            spans.append((element_index, element_index + count, element))
        element_index += count

    return spans

def spans_to_bio(spans, length):
    labels = ["O"] * length 
    for start, end, entity_type in spans:
        labels[start] = "B-" + entity_type
        for i in range(start + 1, end):
            labels[i] = "I-" + entity_type

    return labels

def spans_to_io(spans, length):
    labels = ["O"] * length 
    for start, end, entity_type in spans:
        for i in range(start, end):
            labels[i] = entity_type

    return labels

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    
        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)