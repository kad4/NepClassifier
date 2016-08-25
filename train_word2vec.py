from NepClassifier import Word2VecVectorizer, NeptextData


if __name__ == "__main__":
    # Load data
    documents, labels = NeptextData.load_data()

    # Train vectorizer
    vectorizer = Word2VecVectorizer()
    vectorizer.train(documents)
