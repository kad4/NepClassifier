import logging
from gensim.corpora import Dictionary

from stemmer import NepStemmer
from datasets import NewsData


class NewsCorpus():
    """ Corpus for Nepali news """
    def __init__(self):
        self.stemmer = NepStemmer()
        self.documents, _ = NewsData.load_data()

    def __iter__(self):
        # Yield tokens
        for document in self.documents:
            tokens = self.stemmer.get_stems(document)
            yield tokens


def main():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO
    )

    news_corpus = NewsCorpus()
    dictionary = Dictionary(news_corpus)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=1000)
    dictionary.compactify()

    dictionary.save('tokens.dict')

if __name__ == '__main__':
    main()
