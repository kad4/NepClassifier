from NepClassifier import TfidfVectorizer, NeptextData


if __name__ == "__main__":
    # Load data
    documents, labels = NeptextData.load_data()

    # Train vectorizer
    vectorizer = TfidfVectorizer()
    vectorizer.construct_model(documents)
