from typing import Optional, List


def get_ngrams(
        tokens: List[str],
        n: int,
):
    ngrams = set()
    num = len(tokens) - n
    for i in range(num + 1):
        ngrams.add(tuple(tokens[i:i + n]))
    return ngrams


def ngram_blocking(
        sent: str,
        can_sum: List[str],
        ngram: int,
):
    sent_tri = get_ngrams(sent.split(), ngram)
    for can_sent in can_sum:
        can_tri = get_ngrams(can_sent.split(), ngram)
        if len(sent_tri.intersection(can_tri)) > 0:
            return True
    return False


def tri_blocking(sent, can_sum):
    return ngram_blocking(sent, can_sum, 3)


def quad_blocking(sent, can_sum):
    return ngram_blocking(sent, can_sum, 4)


def get_candidate_sum(
        text: str,
        prediction: List[int],
        sum_size: Optional[int] = None,
        n_block: int = 3
):
    can_sum = []
    for i, sent_id in enumerate(prediction):
        sent = text[sent_id]
        if not ngram_blocking(sent, can_sum, n_block):
            can_sum.append(sent)

        if sum_size and (len(can_sum) == sum_size):
            break

    return can_sum

