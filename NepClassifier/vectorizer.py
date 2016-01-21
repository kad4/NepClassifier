import os

from gensim.corpora import Dictionary
from gensim.models import TfidfModel

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
        """ Returns the tf-idf vector for given document """

        tokens = self.stemmer.get_stems(text)
        text_bow = self.dictionary.doc2bow(tokens)

        return (self.tf_idf_model[text_bow])
