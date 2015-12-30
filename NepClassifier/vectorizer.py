import os
import logging

from gensim.corpora import Dictionary
from gensim.models import TfidfModel

from .stemmer import NepStemmer


class TfidfVectorizer():
    """ Class to vectorizer the given document"""

    def __init__(self):
        self.base_path = os.path.dirname(__file__)

        # Filepath for dictionary
        self.dict_path = os.path.join(self.base_path, 'tokens.dict')

        self.dictionary = None
        self.tf_idf_model = None

        self.stemmer = NepStemmer()

        self.load_data()

    def load_data(self):
        if(not(os.path.exists(self.dict_path))):
            raise Exception('Dictionary not found')

        self.dictionary = Dictionary.load(self.dict_path)

        logging.debug('Initializing tfidf model')
        self.tf_idf_model = TfidfModel(dictionary=self.dictionary)

    def doc2vector(self, text):
        tokens = self.stemmer.get_stems(text)
        text_bow = self.dictionary.doc2bow(tokens)

        return (self.tf_idf_model[text_bow])
