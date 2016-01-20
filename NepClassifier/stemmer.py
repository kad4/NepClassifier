import re
import os
import json


class NepStemmer():
    """ Stemmer for Nepali words """

    def __init__(self):
        self.stems_set = set()
        self.filter_set = set()
        self.suffixes = {}

        self.read_data()

    def read_data(self):
        """ Read stems, filter words and suffixes from files"""

        # Base path for files
        base_path = os.path.dirname(__file__)

        stems_path = os.path.join(base_path, 'stems.txt')
        filter_path = os.path.join(base_path, 'filter.txt')
        suffixes_path = os.path.join(base_path, 'suffixes.json')

        with open(stems_path) as file:
            lines = file.readlines()

        # Constructing stems set
        for line in lines:
            new_line = line.replace('\n', '')
            stem = new_line.split('|')[0]
            self.stems_set.add(stem)

        # Reads filter words
        with open(filter_path) as file:
            filter_stems = file.read().split('\n')

        self.filter_set = set(filter_stems)

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

    def tokenize(self, text):
        """ Tokenize the given text """

        # Removing unnecessary items
        remove_exp = re.compile("[\d]+")
        removed_text = remove_exp.sub("", text)

        # Extracting words from text
        # Splits complex combinations in single step
        extract_exp = re.compile("[\s।|!?.,:;%+\-–*/'‘’“\"()]+")
        words = extract_exp.split(removed_text)

        # Returns the non-empty items only
        return([word for word in words if word != ''])

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
        tokens = self.tokenize(text)

        return([self.stem(token) for token in tokens])
