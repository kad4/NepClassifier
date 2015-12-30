import os
import logging
from gensim.corpora import Dictionary

from stemmer import NepStemmer


class NewsCorpus():
    """ Corpus for Nepali news """

    def __iter__(self):
        stemmer = NepStemmer()

        # Iterate to yield files
        for root, dirs, files in os.walk('data'):
            for file in files:
                absolute_path = os.path.join(root, file)
                with open(absolute_path, 'r') as file_ptr:
                    content = file_ptr.read()
                    tokens = stemmer.get_stems(content)
                    yield tokens


def main():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO
    )

    news_corpus = NewsCorpus()
    dictionary = Dictionary(news_corpus)
    dictionary.filter_extremes()
    dictionary.compactify()

    dictionary.save('tokens.dict')

if __name__ == '__main__':
    main()
