from NepClassifier import TfidfVectorizer, Word2VecVectorizer, NeptextData, SVMClassifier
import sys

if __name__ == "__main__":
    # Load data
    documents, labels = NeptextData.load_data()

    # Initialize vectorizer
    vectorizer = None
    if (len(sys.argv) > 1 and sys.argv[1] == "w2v"):
        vectorizer = Word2VecVectorizer()
    else:
        vectorizer = TfidfVectorizer()

    clf = SVMClassifier(vectorizer)
    clf.train(documents, labels)
