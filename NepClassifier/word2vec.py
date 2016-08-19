import os

import numpy as np

from gensim.models import word2vec

from .stemmer import NepStemmer


class Word2VecVectorizer():
    """
    Transforms text to vectors using word2vec
    """

    def __init__(self):
        # Base path
        self.base_path = os.path.dirname(__file__)

        self.word2vec_data_path = os.path.join(self.base_path, "word2vec")

        self.stemmer = NepStemmer()

        self.model = None

    def train(self, corpus):
        """
        Train the word2vec for feature extraction on given corpus
        """

        self.model = word2vec.Word2Vec(corpus)
        self.model.init_sims(replace=True)

        self.model.save(self.word2vec_data_path)

    def load_model(self):
        if not self.model:
            if not os.path.exists(self.word2vec_data_path):
                raise Exception("Word2Vec model not found.")

            self.model = word2vec.Word2Vec.load(self.word2vec_data_path)
            self.no_of_features = self.model.syn0.shape[1]

    def obtain_word_vector(self, stem):
        self.load_model()

        try:
            return self.model[stem]
        except Exception:
            return np.zeros(self.model.syn0.shape[1])

    def obtain_feature_vector(self, document):
        """
        Returns a single vector representing the document supplied
        """

        self.load_model()

        stems = self.stemmer.get_stems(document)

        feature_vector = np.zeros(self.no_of_features, dtype="float32")

        for stem in stems:
            word_vector = self.obtain_word_vector(stem)

            feature_vector = np.add(feature_vector, word_vector)

        feature_vector = np.divide(feature_vector, len(stems))

        return feature_vector

    def obtain_feature_matrix(self, documents):
        self.load_model()

        # Construct a empty matrix
        input_matrix = np.zeros(
            (len(documents), self.no_of_features),
            dtype="float32"
        )

        for i, document in enumerate(documents):
            input_matrix[i] = self.obtain_feature_vector(document)

        return input_matrix
