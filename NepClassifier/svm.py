import os
import logging
import pickle
from statistics import mean

from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import shuffle
from gensim import matutils

from .vectorizer import TfidfVectorizer
from .datasets import NewsData


class SVMClassifier():
    """
        SVM based classifier for Nepali news
    """

    def __init__(self):
        # Base path
        self.base_path = os.path.dirname(__file__)

        # Path for pre-trained classifier
        self.clf_path = os.path.join(self.base_path, "clf.p")

        # Path for pre-trained classifier
        self.labels_path = os.path.join(self.base_path, "labels.p")

        # Initialize tf-idf vectorizer
        self.vectorizer = TfidfVectorizer()
        self.no_of_features = self.vectorizer.no_of_features

        # Regularization paramater
        self.C = 1.0

        # Classifier to use
        self.classifier = None
        self.labels = None

        # Test data size
        self.test_data_size = 1000

    def train(self):
        """ Train classifier and evaluate performance """

        logging.debug("Loading dataset")
        documents, labels = NewsData.load_data()
        documents, labels = shuffle(documents, labels)

        # Obtain unique labels
        unique_labels = list(set(labels))
        output_array = [unique_labels.index(x) for x in labels]

        # Obtain training data
        training_documents = documents[self.test_data_size:]
        training_data_y = output_array[self.test_data_size:]

        # Obtain testing data
        test_documents = documents[:self.test_data_size]
        test_data_y = output_array[:self.test_data_size]

        logging.debug("Obtaining tf-idf matrix for training data")
        training_corpus = [
            self.vectorizer.doc2vector(x)
            for x in training_documents
        ]

        training_data_x = matutils.corpus2dense(
            training_corpus,
            self.no_of_features
        ).transpose()

        # Initialize SVM
        logging.debug("Training SVM")
        classifier = svm.SVC(self.C, kernel="linear")
        classifier.fit(training_data_x, training_data_y)

        # Dumping trained SVM
        joblib.dump(classifier, self.clf_path)

        # Dump output labels
        pickle.dump(unique_labels, open(self.labels_path, "wb"))

        logging.debug("Evaluating model")
        test_corpus = [
            self.vectorizer.doc2vector(x)
            for x in test_documents
        ]

        test_data_x = matutils.corpus2dense(
            test_corpus,
            self.no_of_features
        ).transpose()

        logging.debug("Predicting classes")
        predicted_class = classifier.predict(test_data_x)

        precision, recall, fscore, __ = precision_recall_fscore_support(
            test_data_y,
            predicted_class
        )

        precision = mean(precision)
        recall = mean(recall)
        fscore = mean(fscore)

        logging.info("Precision: " + str(precision))
        logging.info("Recall: " + str(recall))
        logging.info("fscore: " + str(fscore))

    def load_clf(self):
        """ Load the pre-trained classifier """

        if (not(os.path.exists(self.clf_path))):
            raise Exception("Pre trained classifier not found")

        if (not(os.path.exists(self.labels_path))):
            raise Exception("Labels for classifier not found")

        self.classifier = joblib.load(self.clf_path)
        self.labels = pickle.load(open(self.labels_path, "rb"))

    def predict(self, text):
        """ Predict the class of given text """

        if(not(self.classifier)):
            raise Exception("Classifier not loaded")

        if (text == ""):
            raise Exception('Empty text provided')

        tf_idf_vector = matutils.sparse2full(
            self.vectorizer.doc2vector(text),
            self.no_of_features
        ).reshape(1, -1)

        predicted_output = self.classifier.predict(tf_idf_vector)[0]
        return(self.labels[int(predicted_output)])
