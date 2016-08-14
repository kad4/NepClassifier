import os

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim import matutils

from .stemmer import NepStemmer


class TfidfVectorizer():
    """ Class to vectorizer the given document"""

    def __init__(self):
        # Stemmer to use
        self.stemmer = NepStemmer()

        # Absolute path for dictionary file
        dict_path = os.path.join(os.path.dirname(__file__), "tokens.dict")

        if(not(os.path.exists(dict_path))):
            raise Exception('Dictionary file not found')

        # Load dictionary
        self.dictionary = Dictionary.load(dict_path)
        self.no_of_features = len(self.dictionary)

        # Construct the tf_idf_model
        self.tf_idf_model = TfidfModel(dictionary=self.dictionary)

    def doc2vector(self, text):
        """ Returns the sparse tf-idf vector for given document """

        tokens = self.stemmer.get_stems(text)
        text_bow = self.dictionary.doc2bow(tokens)

        return (self.tf_idf_model[text_bow])

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
