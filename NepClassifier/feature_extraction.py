import os
import pickle
from math import log
from operator import itemgetter

import numpy as np

from .stemmer import NepStemmer


class TfidfVectorizer():
    """
        Feature extraction from given text
        by using tf-idf
    """
    def __init__(self, max_stems=1000):
        # Stemmer to use
        self.stemmer = NepStemmer()

        # Base path to use
        self.base_path = os.path.dirname(__file__)

        # Folder containing data
        self.corpus_path = os.path.join(self.base_path, 'data')

        # File containing the corpus info
        self.data_path = os.path.join(self.base_path, 'data.p')

        # Maximum stems to use
        self.max_stems = max_stems

        # Number of stems selected
        self.stems_size = None

        # Stems to use as feature
        self.stems = None

        # Vector to hold the IDF of stems
        self.idf_vector = None

    def process_corpus(self):
        """
            Class method to process corpus located at path provided
            at self.corpus_path

            The data must be organized as utf-8 encoded raw text file
            having following structure

            data/
                class1/
                    *.txt
                    *.txt
                class2/
                    *.txt
                    *.txt
                ...
        """

        # Vectors for stems
        count_vector = {}
        occurrence_vector = {}

        total_docs = 0

        for root, dirs, files in os.walk(self.corpus_path):
            for file_path in files:
                abs_path = os.path.join(self.base_path, root, file_path)

                file = open(abs_path, 'r')
                content = file.read()
                file.close()

                # Obtain known stems
                doc_stems = self.stemmer.get_stems(content)
                doc_stems_set = set(doc_stems)

                # Add the count of stems
                for stem in doc_stems:
                    count_vector[stem] = count_vector.get(stem, 0) + 1

                for stem in doc_stems_set:
                    occurrence_vector[stem] = occurrence_vector.get(stem, 0) \
                        + 1

                total_docs += 1

        # Sort stems based on frequecy
        stem_tuple = sorted(
            count_vector.items(),
            key=itemgetter(1),
            reverse=True
        )

        # Filter stems which occur in less than 4 documents
        stem_tuple = [item for item in stem_tuple if item[1] > 3]

        # Construct a ordered list of stems
        stems = [item[0] for item in stem_tuple]

        # IDF vector for the stems
        idf_vector = [
            log(total_docs / (1 + occurrence_vector[k]))
            for k in stems
        ]

        # Dump the data obtained
        data = {
            'stems': stems,
            'idf_vector': idf_vector
        }

        pickle.dump(data, open(self.data_path, 'wb'))

    def load_corpus_info(self):
        """ Load the corpus information a file """

        # Load dump data
        data = pickle.load(open(self.data_path, 'rb'))

        # Limit the number of stems
        self.stems = data['stems'][:self.max_stems]
        self.idf_vector = data['idf_vector'][:self.max_stems]

        self.stems_size = len(self.stems)

    def tf_vector(self, text):
        """ Compute tf vector for a given text """

        if (not(self.stems)):
            raise Exception('Corpus info not available')

        # Find stems in document
        doc_stems = self.stemmer.get_stems(text)

        # Contruct dictionary of stems
        doc_vector = {}
        for stem in doc_stems:
            doc_vector[stem] = doc_vector.get(stem, 0) + 1

        # Maximum count in the document
        max_count = max(doc_vector.values())
        if(max_count == 0):
            max_count = 1

        # Calculate the tf of text
        tf_vector = 0.5 + (0.5 / max_count) * np.array(
            [doc_vector.get(stem, 0) for stem in self.stems]
        )

        return(tf_vector)

    def tf_idf_vector(self, text):
        """ Calculates the tf-idf vector for a given text """

        if (not(self.stems)):
            raise Exception('Corpus info not available')

        tf_vector = self.tf_vector(text)
        idf_vector = np.array(self.idf_vector)

        # Element wise product
        return(tf_vector * idf_vector)
