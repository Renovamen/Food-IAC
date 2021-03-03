

import numpy as np
from typing import List, Tuple

def jaccard_similarity(a: List[str], b: List[str]) -> float:
    intersection = len(list(set(a).intersection(b)))
    union = (len(a) + len(b)) - intersection
    return float(intersection) / union

class Diversity:
    """Compute "Diversity" score for a set of candidate sentences."""

    def __init__(self, threshold: float = 0.3, n: int = 4):
        self.threshold = threshold
        self._n = n

    def ngrams(self, sentence: str, n: int) -> List[Tuple[str]]:
        """Convert a sentence into a n-gram list."""
        words = sentence.split(" ")
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngrams.append(ngram)
        return ngrams

    def difference(self, candidate, refs, n: int) -> bool:
        token_c = self.ngrams(candidate, n=n)
        token_r = []

        for reference in refs:
            token_r.extend(self.ngrams(reference, n=n))

        # compute the jaccard similarity
        jaccard = jaccard_similarity(token_r, token_c)

        if jaccard < self.threshold:
            return True

        return False

    def compute_score_ngram(self, sentences, n: int):
        n_diff, n_similar = 0, 0

        for i, sent_a in enumerate(sentences):
            for j, sent_b in enumerate(sentences):
                if i == j:
                    continue

                # sanity check
                assert(type(sent_a) is list)
                assert(len(sent_a) >= 1)
                assert(type(sent_b) is list)
                assert(len(sent_b) >= 1)

                if self.difference(sent_a[0], sent_b[0], n=n):
                    n_diff += 1
                else:
                    n_similar += 1

        score = n_diff / (n_diff + n_similar)
        return score

    def compute_score(self, hypothesis):
        scores = []
        for k in range(1, self._n + 1):
            scores.append(self.compute_score_ngram(hypothesis, n=k))
        return scores

    def method(self):
        return "Diversity"
