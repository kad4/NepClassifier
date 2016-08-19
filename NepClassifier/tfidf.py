import os

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

        self.tokens_path = os.path.join(self.base_path, "tokens")

        self.stemmer = None
        self.dictionary = None
        self.no_of_features = None
        self.tf_idf_model = None

    def construct_dictionary(self, corpus):
        dictionary = Dictionary(corpus)
        dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=1000)
        dictionary.compactify()

        dictionary.save(self.tokens_path)

    def load_data(self):

        if not self.dictionary:

            if not os.path.exists(self.tokens_path):
                raise Exception('Dictionary file not found')

            self.stemmer = NepStemmer()
            self.dictionary = Dictionary.load(self.tokens_path)
            self.no_of_features = len(self.dictionary)

            # Construct the tf_idf_model
            self.tf_idf_model = TfidfModel(dictionary=self.dictionary)

    def doc2vector(self, document):
        """ Returns the sparse tf-idf vector for given document """

        tokens = self.stemmer.get_stems(document)
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

        input_matrix = matutils.corpus2dense(
            input_matrix_sparse,
            self.no_of_features
        ).transpose()

        return input_matrix
