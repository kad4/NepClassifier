import os
import json

from .tokenizer import tokenize


class NepStemmer():
    """ Stemmer for Nepali words """

    def __init__(self):
        self.stems_set = set()
        self.filter_set = set()
        self.suffixes = {}

        # Base path for files
        base_path = os.path.dirname(__file__)

        # Read stems list
        stems_path = os.path.join(base_path, 'stems.txt')
        with open(stems_path) as file:
            lines = file.readlines()

        self.stems_set = {
            line.replace("\n", "").split("|")[0]
            for line in lines
        }

        filter_path = os.path.join(base_path, 'filter.txt')
        # Reads filter words
        with open(filter_path) as file:
            self.filter_set = set(file.read().split("\n"))

        # Read suffix list
        suffixes_path = os.path.join(base_path, 'suffixes.json')
        with open(suffixes_path, 'r') as file:
            self.suffixes = json.load(file)

    def remove_suffix(self, word):
        """ Removes suffixes from a given word """
        # for L in 2, 3, 4, 5, 6:
        for L in range(2, 7):
            if len(word) > L + 1:
                for suf in self.suffixes[str(L)]:
                    if word.endswith(suf):
                        return word[:-L]
            else:
                break
        return word

    def stem(self, word):
        """ Returns the stem of the given word """

        word_stem = self.remove_suffix(word)
        if(word_stem in self.stems_set):
            return word_stem
        else:
            return word

    def get_stems(self, text):
        """ Returns stem list from a given text """

        # Obtain tokens of the text
        tokens = tokenize(text)

        return([self.stem(token) for token in tokens])
