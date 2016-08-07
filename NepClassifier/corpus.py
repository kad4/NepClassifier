import logging
from gensim.corpora import Dictionary

from datasets import NewsCorpus


def main():
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(message)s',
        level=logging.INFO
    )

    news_corpus = NewsCorpus()
    dictionary = Dictionary(news_corpus)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=1000)
    dictionary.compactify()

    dictionary.save('tokens.dict')

if __name__ == '__main__':
    main()
