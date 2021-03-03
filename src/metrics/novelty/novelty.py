import numpy as np
from typing import List, Tuple

def jaccard_similarity(a: List[str], b: List[str]) -> float:
    intersection = len(list(set(a).intersection(b)))
    union = (len(a) + len(b)) - intersection
    return float(intersection) / union

class Novelty:
    """Compute "novelty" score for a set of candidate sentences."""

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

    def compute_score_ngram(self, reference, hypothesis, n: int):
        assert len(reference) == len(hypothesis)

        n_diff, n_similar = 0, 0

        for i, hypo in enumerate(hypothesis):
            hypo = hypo
            ref = reference[i]

            # sanity check
            assert(type(hypo) is list)
            assert(len(hypo) >= 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

            if self.difference(hypo[0], ref, n=n):
                n_diff += 1
            else:
                n_similar += 1

        score = n_diff / (n_diff + n_similar)
        return score

    def compute_score(self, reference, hypothesis):
        scores = []
        for k in range(1, self._n + 1):
            scores.append(self.compute_score_ngram(reference, hypothesis, n=k))
        return scores

    def method(self):
        return "Novelty"
