import numpy as np
from collections import Counter
from math import sqrt

from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from model import dameraulevenshtein


def ngrams(sentence, n):
  return zip(*[sentence.split()[i:] for i in range(n)])


def jaccard(first, second):
    intersection = len(first & second)
    union = len(first | second)
    res = intersection / union

    return res


def cosine(first, second):
    li = len(first & second)
    la = len(first)
    lb = len(second)
    result = li / sqrt(la) * sqrt(lb)

    return result


def tanimoto(first, second):
    li = len(first & second)
    la = len(first)
    lb = len(second)
    result = li / (la + lb - li)

    return result
