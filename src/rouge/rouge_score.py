from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re

from absl import logging
import nltk
import numpy as np
import six
from six.moves import map
from six.moves import range
from rouge_score.rouge import scoring
from konlpy.tag import Mecab


class RougeScorer(scoring.BaseScorer):

    __doc__ = r"""
        I adopted the implementation of google-research, which is also proposed by Hugging Face.
        https://github.com/google-research/google-research/tree/master/rouge
        https://huggingface.co/spaces/evaluate-metric/rouge
        
        Instead of DefaultTokenizer, Mecab is used for Korean language.
        It was also adopted in Dacon 한국어 문서 생성요약 AI 경진대회. 
        
        The separator between sentences must be a '\n'.
        
        The difference between rougeL and rougeLsum is that rougeL ignores '\n' between sentences, 
        but rougeLsum does not. It makes rougeL consider the order of sentences in given text.
    """

    def __init__(
            self,
            rouge_types,
            split_summaries=False,
            tokenizer=None
    ):
        self.rouge_types = rouge_types
        if tokenizer:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = Mecab()
            logging.info("Using default tokenizer for ROUGE.")

        self._split_summaries = split_summaries


    def score_multi(self, targets, prediction):
        score_dicts = [self.score(t, prediction) for t in targets]
        max_score = {}
        for k in self.rouge_types:
            index = np.argmax([s[k].fmeasure for s in score_dicts])
            max_score[k] = score_dicts[index][k]

        return max_score


    def score(self, target, prediction):
        """
        Calculates rouge scores between the target and prediction.
          Args:
            target: Text containing the target (ground truth) text, or if a list
            prediction: Text containing the predicted text.
          Returns:
            A dict mapping each rouge type to a Score object.
          Raises:
            ValueError: If an invalid rouge type is encountered.
        """
        # Pre-compute target tokens and prediction tokens for use by different
        # types, except if only "rougeLsum" is requested.

        if len(self.rouge_types) == 1 and self.rouge_types[0] == "rougeLsum":
            target_tokens = None
            prediction_tokens = None
        else:
            target_tokens = self._tokenizer.morphs(target)
            prediction_tokens = self._tokenizer.morphs(prediction)
        result = {}

        for rouge_type in self.rouge_types:
            if rouge_type == "rougeL":
                # Rouge from longest common subsequences.
                scores = _score_lcs(target_tokens, prediction_tokens)

            elif rouge_type == "rougeLsum":
                # Note: Does not support multi-line text.
                def get_sents(text):
                    if self._split_summaries:
                        sents = nltk.sent_tokenize(text)
                    else:
                        # Assume sentences are separated by newline.
                        sents = six.ensure_str(text).split("\n")
                    sents = [x for x in sents if len(x)]
                    return sents

                target_tokens_list = [
                    self._tokenizer.morphs(s) for s in get_sents(target)]
                prediction_tokens_list = [
                    self._tokenizer.morphs(s) for s in get_sents(prediction)]

                scores = _summary_level_lcs(target_tokens_list,
                                            prediction_tokens_list)

            elif re.match(r"rouge[0-9]$", six.ensure_str(rouge_type)):
                # Rouge from n-grams.
                n = int(rouge_type[5:])
                if n <= 0:
                    raise ValueError("rougen requires positive n: %s" % rouge_type)
                target_ngrams = _create_ngrams(target_tokens, n)
                prediction_ngrams = _create_ngrams(prediction_tokens, n)
                scores = _score_ngrams(target_ngrams, prediction_ngrams)
            else:
                raise ValueError("Invalid rouge type: %s" % rouge_type)
            result[rouge_type] = scores

        return result


def _create_ngrams(tokens, n):
    ngrams = collections.Counter()
    for ngram in (tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)):
        ngrams[ngram] += 1
    return ngrams


def _score_lcs(target_tokens, prediction_tokens):
    if not target_tokens or not prediction_tokens:
        return scoring.Score(precision=0, recall=0, fmeasure=0)

    # Compute length of LCS from the bottom up in a table (DP appproach).
    lcs_table = _lcs_table(target_tokens, prediction_tokens)
    lcs_length = lcs_table[-1][-1]

    precision = lcs_length / len(prediction_tokens)
    recall = lcs_length / len(target_tokens)
    fmeasure = scoring.fmeasure(precision, recall)

    return scoring.Score(precision=precision, recall=recall, fmeasure=fmeasure)


def _lcs_table(ref, can):
    rows = len(ref)
    cols = len(can)
    lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if ref[i - 1] == can[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
    return lcs_table


def _backtrack_norec(t, ref, can):
    i = len(ref)
    j = len(can)
    lcs = []
    while i > 0 and j > 0:
        if ref[i - 1] == can[j - 1]:
            lcs.insert(0, i - 1)
            i -= 1
            j -= 1
        elif t[i][j - 1] > t[i - 1][j]:
            j -= 1
        else:
            i -= 1
    return lcs


def _summary_level_lcs(ref_sent, can_sent):
    if not ref_sent or not can_sent:
        return scoring.Score(precision=0, recall=0, fmeasure=0)

    m = sum(map(len, ref_sent))
    n = sum(map(len, can_sent))
    if not n or not m:
        return scoring.Score(precision=0, recall=0, fmeasure=0)

    # get token counts to prevent double counting
    token_cnts_r = collections.Counter()
    token_cnts_c = collections.Counter()
    for s in ref_sent:
        # s is a list of tokens
        token_cnts_r.update(s)
    for s in can_sent:
        token_cnts_c.update(s)

    hits = 0
    for r in ref_sent:
        lcs = _union_lcs(r, can_sent)
        # Prevent double-counting:
        # The paper describes just computing hits += len(_union_lcs()),
        # but the implementation prevents double counting. We also
        # implement this as in version 1.5.5.
        for t in lcs:
            if token_cnts_c[t] > 0 and token_cnts_r[t] > 0:
                hits += 1
                token_cnts_c[t] -= 1
                token_cnts_r[t] -= 1

    recall = hits / m
    precision = hits / n
    fmeasure = scoring.fmeasure(precision, recall)
    return scoring.Score(precision=precision, recall=recall, fmeasure=fmeasure)


def _union_lcs(ref, c_list):
    lcs_list = [lcs_ind(ref, c) for c in c_list]
    return [ref[i] for i in _find_union(lcs_list)]


def _find_union(lcs_list):
    return sorted(list(set().union(*lcs_list)))


def lcs_ind(ref, can):
    t = _lcs_table(ref, can)
    return _backtrack_norec(t, ref, can)


def _score_ngrams(target_ngrams, prediction_ngrams):
    intersection_ngrams_count = 0
    for ngram in six.iterkeys(target_ngrams):
        intersection_ngrams_count += min(target_ngrams[ngram],
                                         prediction_ngrams[ngram])
    target_ngrams_count = sum(target_ngrams.values())
    prediction_ngrams_count = sum(prediction_ngrams.values())

    precision = intersection_ngrams_count / max(prediction_ngrams_count, 1)
    recall = intersection_ngrams_count / max(target_ngrams_count, 1)
    fmeasure = scoring.fmeasure(precision, recall)

    return scoring.Score(precision=precision, recall=recall, fmeasure=fmeasure)
