from .bleu_scorer import BleuScorer

class Bleu:
    """Main Class to compute the BLEU metric"""

    def __init__(self, n = 4):
        # default compute Blue score up to 4
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, reference, hypothesis):
        """
        Compute BLEU score

        Args:
            reference (list): Reference sentences ([[ref1a, ref1b, ref1c], ..., [refna, refnb]])
            hypothesis (list): Predicted sentences ([[hypo1], [hypo2], ..., [hypon]])

        Returns:
            score: computed BLEU1-BLEU4 score for the corpus
            scores: computed BLEU1-BLEU4 scores for each image
        """

        assert len(reference) == len(hypothesis)

        bleu_scorer = BleuScorer(n = self._n)

        for id, hypo in enumerate(hypothesis):
            hypo = hypo
            ref = reference[id]

            # sanity check
            assert(type(hypo) is list)
            assert(len(hypo) >= 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

            bleu_scorer += (hypo[0], ref)

        # score, scores = bleu_scorer.compute_score(option = 'shortest')
        score, scores = bleu_scorer.compute_score(option = 'closest', verbose = 0)
        # score, scores = bleu_scorer.compute_score(option = 'average', verbose = 1)

        return score, scores

    def method(self):
        return "Bleu"
