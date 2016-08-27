import os
import logging

from gensim import matutils
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

from .stemmer import NepStemmer


class TfidfVectorizer():
    """
    Transform text to tf-idf representation
    """

    def __init__(self):

        self.base_path = os.path.dirname(__file__)
        self.dictionary_path = os.path.join(self.base_path, "dictionary")
        self.tf_idf_model_path = os.path.join(self.base_path, "tfidf")

        self.stemmer = NepStemmer()
        self.tf_idf_model = None

    def get_tokens(self, document):
        if not self.stemmer:
            raise Exception("Stemmer not available")

        return self.stemmer.get_stems(document)

    def construct_model(self, documents):
        logging.basicConfig(
            format='%(asctime)s:%(levelname)s:%(message)s',
            level=logging.INFO
        )

        logging.info("Obtaining word tokens")
        tokens = [self.get_tokens(document) for document in documents]
        # self.tf_idf_model = TfidfModel(tokens)

        logging.info("Constructing dictionary")
        self.dictionary = Dictionary(tokens)
        self.dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=1000)
        self.dictionary.compactify()
        self.dictionary.save(self.dictionary_path)

        logging.info("Constructing TF-IDF model")
        self.tf_idf_model = TfidfModel(dictionary=self.dictionary)
        self.tf_idf_model.save(self.tf_idf_model_path)

    def load_data(self):

        if not self.tf_idf_model:
            if not os.path.exists(self.tf_idf_model_path):
                raise Exception('TF-IDF model file not found')

            self.dictionary = Dictionary.load(self.dictionary_path)
            self.tf_idf_model = TfidfModel.load(self.tf_idf_model_path)

    def doc2vector(self, document):
        """ Returns the sparse tf-idf vector for given document """

        tokens = self.get_tokens(document)
        bag_of_words = self.dictionary.doc2bow(tokens)

        return (self.tf_idf_model[bag_of_words])

    def obtain_feature_vector(self, document):
        """
        Returns a single dense tf-idf vector for a given document
        """

        self.load_data()

        tf_idf_vector = matutils.sparse2full(
            self.doc2vector(document),
            self.no_of_features
        ).reshape(1, -1)

        return tf_idf_vector

    def obtain_feature_matrix(self, documents):
        """
        Returns the tf-idf dense matrix for the given documents
        """

        self.load_data()

        input_matrix_sparse = [
            self.doc2vector(x)
            for x in documents
        ]

        no_of_features = len(self.tf_idf_model.idfs)

        input_matrix = matutils.corpus2dense(
            input_matrix_sparse,
            no_of_features
        ).transpose()

        return input_matrix
