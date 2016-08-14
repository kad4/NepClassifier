from gensim.models import TfidfModel
from gensim import matutils

from .tokenizer import Tokenizer


class TfidfVectorizer():
    """
    Transform text to tf-idf representation
    """

    def __init__(self):

        # Initialize tokenizer
        self.tokenizer = Tokenizer()
        self.no_of_features = self.tokenizer.no_of_features

        # Construct the tf_idf_model
        self.tf_idf_model = TfidfModel(dictionary=self.tokenizer.dictionary)

    def doc2vector(self, document):
        """ Returns the sparse tf-idf vector for given document """

        bag_of_words = self.tokenizer.obtain_bow(document)

        return (self.tf_idf_model[bag_of_words])

    def obtain_feature_matrix(self, documents):
        """
        Returns the tf-idf dense matrix for the given documents
        """

        input_matrix_sparse = [
            self.doc2vector(x)
            for x in documents
        ]

        input_matrix = matutils.corpus2dense(
            input_matrix_sparse,
            self.no_of_features
        ).transpose()

        return input_matrix

    def obtain_feature_vector(self, document):
        """
        Returns a single dense tf-idf vector for a given document
        """

        tf_idf_vector = matutils.sparse2full(
            self.doc2vector(document),
            self.no_of_features
        ).reshape(1, -1)

        return tf_idf_vector
