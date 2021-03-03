from typing import Tuple, List, Union
from .bleu import Bleu
from .cider import Cider
from .meteor import Meteor
from .rouge import Rouge
from .spice import Spice
from .novelty import Novelty
from .diversity import Diversity

class Metrics:
    """
    Compute metrics on given reference and candidate sentences set.
    Now supports BLEU, CIDEr, METEOR and ROUGE-L

    Args:
        references ([[[ref1a], [ref1b], [ref1c]], ..., [[refna], [refnb]]]):
            Reference sencences (list of word ids)
        candidates ([[hyp1], [hyp2], ..., [hypn]]): Candidate sencences (list of
            word ids)
        rev_word_map (dict): ix2word map
    """

    def __init__(
        self,
        references: List[List[List[int]]],
        candidates: List[List[int]],
        rev_word_map: dict
    ) -> None:
        corpus = setup_corpus(references, candidates, rev_word_map)
        self.ref_sentence = corpus[0]
        self.hypo_sentence = corpus[1]

        self.bleu_computer = Bleu()
        self.cider_computer = Cider()
        self.rouge_computer = Rouge()
        self.meteor_computer = Meteor()
        self.spice_computer = Spice()
        self.novelty_computer = Novelty()
        self.diversity_computer = Diversity()

    @property
    def belu(self) -> Tuple[float]:
        bleu_score = self.bleu_computer.compute_score(self.ref_sentence, self.hypo_sentence)
        return bleu_score[0][0], bleu_score[0][0], bleu_score[0][2], bleu_score[0][3]

    @property
    def cider(self) -> float:
        cider_score = self.cider_computer.compute_score(self.ref_sentence, self.hypo_sentence)
        return cider_score[0]

    @property
    def rouge(self) -> float:
        rouge_score = self.rouge_computer.compute_score(self.ref_sentence, self.hypo_sentence)
        return rouge_score[0]

    @property
    def meteor(self) -> float:
        meteor_score = self.meteor_computer.compute_score(self.ref_sentence, self.hypo_sentence)
        return meteor_score[0]

    @property
    def spice(self) -> float:
        spice_score = self.spice_computer.compute_score(self.ref_sentence, self.hypo_sentence)
        return spice_score[0]

    @property
    def novelty(self) -> Tuple[float]:
        novelty_score = self.novelty_computer.compute_score(self.ref_sentence, self.hypo_sentence)
        return novelty_score[0], novelty_score[3]

    @property
    def diversity(self) -> Tuple[float]:
        diversity_score = self.diversity_computer.compute_score(self.hypo_sentence)
        return diversity_score[0], diversity_score[3]

    @property
    def all_metrics(self) -> Tuple[Union[float, Tuple[float]]]:
        """
        Compute all meterics

        Returns:
            scores (Tuple[Union[float, Tuple[float]]]): BLEU-1, BLEU-2, BLEU-3,
                BLEU-4, CIDEr, ROUGE-L, METEOR, Spice
        """
        return self.belu, self.cider, self.rouge, self.meteor, self.spice


def setup_corpus(references, candidates, rev_word_map):
    ref_sentence = []
    hypo_sentence = []

    for cnt, each_image in enumerate(references):

        # ground truths
        cur_ref_sentence = []
        for cap in each_image:
            sentence = [rev_word_map[ix] for ix in cap]
            cur_ref_sentence.append(' '.join(sentence))

        ref_sentence.append(cur_ref_sentence)

        # predictions
        sentence = [rev_word_map[ix] for ix in candidates[cnt]]
        hypo_sentence.append([' '.join(sentence)])

    return ref_sentence, hypo_sentence
