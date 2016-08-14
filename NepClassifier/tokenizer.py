import os

from gensim.corpora import Dictionary

from .stemmer import NepStemmer


def construct_dictionary(corpus):
    """
    Iterates through the given Corpus and constructs a token dictionary to
    use
    """

    dictionary = Dictionary(corpus)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=1000)
    dictionary.compactify()

    tokens_path = os.path.join(os.path.dirname(__file__), "tokens.dict")

    dictionary.save(tokens_path)


class Tokenizer():
    """
    Stemmer and Tokenizer for Nepali text
    """

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

    def obtain_bow(self, document):
        """
        Return a the bag-of-words representation for the document
        """

        tokens = self.stemmer.get_stems(document)
        bag_of_words = self.dictionary.doc2bow(tokens)

        return bag_of_words
