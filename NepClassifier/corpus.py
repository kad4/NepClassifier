import os
import logging
from gensim.corpora import Dictionary


logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO
)


def process_corpus(corpus):
    """
    Iterates through the given Corpus and constructs a token dictionary to
    use
    """

    dictionary = Dictionary(corpus)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=1000)
    dictionary.compactify()

    tokens_path = os.path.join(os.path.dirname(__file__), "tokens.dict")

    dictionary.save(tokens_path)
