from NepClassifier import Word2VecVectorizer, NeptextData, SVMClassifier


if __name__ == "__main__":
    # Load data
    documents, labels = NeptextData.load_data()

    # Initialize vectorizer
    vectorizer = Word2VecVectorizer()

    clf = SVMClassifier(vectorizer)
    clf.train(documents, labels)
