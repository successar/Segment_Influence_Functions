from bisect import bisect_left, bisect_right
import re

data = open("extras/regExPatterns.txt").read().strip().split("\n")
data = [x for x in data if len(x.strip()) > 0 and x[0] != "#"]

variables = {}
patterns = {}

for line in data:
    if line.startswith("@"):
        name, pat = line.split("::")
        variables[name] = pat 
    else:
        name, pat = line.split("::")
        for k, v in variables.items():
            if k in pat:
                pat = pat.replace(k, v)
        if name not in patterns:
            patterns[name] = []
        patterns[name].append(pat)

sp = [x.replace("\\\\", "\\") for x in patterns["strength"]]
dosage_pattern = re.compile("|".join(sp))

def find_le(a, x):
    "Find rightmost value less than or equal to x"
    i = bisect_right(a, x)
    if i:
        return i - 1
    return 0


def find_ge(a, x):
    "Find leftmost item greater than or equal to x"
    i = bisect_left(a, x)
    if i != len(a):
        return i
    return -1

def dosages(tokens):
    x = 0
    starts = ([0] + [x := x + len(t) + 1 for t in tokens])[:-1]
    for match in re.finditer(dosage_pattern, " ".join(tokens)):
        start = find_le(starts, match.start())
        end = find_le(starts, match.end() - 1) + 1
        if tokens[start][0].isdigit() and tokens[end - 1] != "%":
            yield start, end

def labels_to_spans(sentence, label):
    i = 0
    spans = []
    while i < len(sentence.tokens):
        if sentence.labels[i] == label:
            start = i
            while i < len(sentence.tokens) and sentence.labels[i] == label:
                i += 1
            spans.append((start, i))
        else :
            i += 1

    return spans