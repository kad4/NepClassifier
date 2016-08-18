import os
import gzip
import pickle

from .stemmer import NepStemmer


class NeptextData():
    """
    General Nepali text dataset
    """

    def load_data():
        # Base path to use
        base_path = os.path.dirname(__file__)

        # Path for data set
        data_path = os.path.join(base_path, "data", "neptext.pkl.gz")

        # Return the obtained data
        documents, labels = pickle.load(gzip.open(data_path, 'rb'))
        return (documents, labels)


class NeptextCorpus():
    """
    Corpus for Nepali text dataset
    """

    def __init__(self):
        self.stemmer = NepStemmer()
        self.documents, _ = NeptextData.load_data()

    def __iter__(self):
        # Yield tokens
        for document in self.documents:
            tokens = self.stemmer.get_stems(document)
            yield tokens
