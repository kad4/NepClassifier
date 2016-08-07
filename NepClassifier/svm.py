import os
import time
import logging
import pickle

from sklearn import svm
from sklearn import cross_validation
from sklearn.utils import shuffle

from gensim import matutils

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from .tfidf import TfidfVectorizer
from .datasets import NewsData


class SVMClassifier():
    """
    SVM based classifier for Nepali textxt
    """

    def __init__(self):
        # Base path
        self.base_path = os.path.dirname(__file__)

        # Path for pre-trained classifier
        self.clf_path = os.path.join(self.base_path, "clf.p")

        # Path for pre-trained classifier
        self.labels_path = os.path.join(self.base_path, "labels.p")

        # Path for training input/output matrix
        self.matrix_path = os.path.join(self.base_path, "matrix.p")

        # Initialize tf-idf vectorizer
        self.vectorizer = TfidfVectorizer()
        self.no_of_features = self.vectorizer.no_of_features

        # Regularization paramater
        self.C = 1.0

        # Classifier to use
        self.classifier = None
        self.labels = None

        # Test data size
        self.test_data_size = 0.33

    def contruct_cost_function(self, input_matrix, output_matrix):
        def eval(c):

            logging.debug("Training using c={}".format(c))

            # Construct classifier
            classifier = svm.SVC(c, kernel="linear")

            scores = cross_validation.cross_val_score(
                classifier,
                input_matrix,
                output_matrix,
                cv=5
            )

            score = scores.mean()
            logging.debug("Mean score={}".format(score))

            data = {
                "loss": 1-score,
                "status": STATUS_OK,

                "eval_time": time.time(),
                "score": score
            }

            return data

        return eval

    def load_matrix(self):
        logging.debug("Loading dataset")

        if not os.path.exists(self.matrix_path):
            # Obtain corpus data
            documents, labels = shuffle(NewsData.load_data())

            # Encode output label
            unique_labels = list(set(labels))
            self.output_matrix = [self.unique_labels.index(x) for x in labels]

            logging.debug("Obtaining tf-idf matrix for data")
            input_matrix_sparse = [
                self.vectorizer.doc2vector(x)
                for x in documents
            ]

            self.input_matrix = matutils.corpus2dense(
                input_matrix_sparse,
                self.no_of_features
            ).transpose()

            pickle.dump(
                (self.input_matrix, self.output_matrix),
                open(self.matrix_path, "wb")
            )

            # Dump output labels
            pickle.dump(unique_labels, open(self.labels_path, "wb"))
        else:
            self.input_matrix, self.output_matrix = pickle.load(
                open(self.matrix_path, "rb")
            )

    def train(self):
        """ Train classifier and evaluate performance """

        # Load input/output matrix
        self.load_matrix()

        eval = self.contruct_cost_function(
            self.input_matrix,
            self.output_matrix
        )

        trials = Trials()

        # Perform hyper paramater optimization
        best_c = fmin(
            fn=eval,
            space=hp.lognormal('c', 0, 1),
            algo=tpe.suggest,
            max_evals=10,
            trials=trials
        )

        # Initialize SVM
        logging.debug("Training SVM")
        classifier = svm.SVC(best_c, kernel="linear")
        classifier.fit(self.input_matrix, self.output_matrix)

        # Dumping trained SVM
        pickle.dump(classifier, open(self.clf_path, "wb"))

    def load_clf(self):
        """ Load the pre-trained classifier """

        logging.debug("Loading classifier data")

        if (not(os.path.exists(self.clf_path))):
            raise Exception("Pre trained classifier not found")

        if (not(os.path.exists(self.labels_path))):
            raise Exception("Labels for classifier not found")

        self.classifier = pickle.load(self.clf_path)
        self.labels = pickle.load(open(self.labels_path, "rb"))

    def predict(self, text):
        """
        Predict the class of given text
        """

        # Check and load classifier data
        self.load_clf()

        if (text == ""):
            raise Exception('Empty text provided')

        tf_idf_vector = matutils.sparse2full(
            self.vectorizer.doc2vector(text),
            self.no_of_features
        ).reshape(1, -1)

        predicted_output = self.classifier.predict(tf_idf_vector)[0]
        return(self.labels[int(predicted_output)])
