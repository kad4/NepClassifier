import os
import logging
import pickle

from sklearn import svm
from sklearn import cross_validation
from sklearn.utils import shuffle

from hyperopt import fmin, tpe, hp

from .tfidf import TfidfVectorizer
from .datasets import NewsData


class SVMClassifier():
    """
    SVM based classifier for Nepali text
    """

    def __init__(self):
        # Base path
        self.base_path = os.path.dirname(__file__)

        # Path for pre-trained classifier
        self.clf_path = os.path.join(self.base_path, "svm.p")

        # Path for pre-trained classifier
        self.labels_path = os.path.join(self.base_path, "labels.p")

        # Path for training input/output matrix
        self.matrix_path = os.path.join(self.base_path, "matrix.p")

        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer()
        self.no_of_features = self.vectorizer.no_of_features

        # Classifier to use
        self.classifier = None
        self.labels = None

        # Test data size
        self.test_data_size = 0.33

        self.max_evals = 10

    def contruct_cost_function(self, input_matrix, output_matrix):
        def eval(c):

            logging.info("Training using c={}".format(c))

            # Construct classifier
            classifier = svm.SVC(c, kernel="linear")

            scores = cross_validation.cross_val_score(
                classifier,
                input_matrix,
                output_matrix,
                cv=5
            )

            score = scores.mean()
            logging.info("Mean score={}".format(score))

            # Return loss
            return 1-score

        return eval

    def load_matrix(self):
        logging.info("Loading dataset")

        if not os.path.exists(self.matrix_path):
            # Obtain corpus data
            documents, labels = shuffle(NewsData.load_data())

            # Encode output label
            unique_labels = list(set(labels))
            self.output_matrix = [unique_labels.index(x) for x in labels]

            logging.info("Obtaining tf-idf matrix for data")
            self.input_matrix = self.vectorizer.obtain_feature_matrix(
                documents
            )

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
        """
        Train classifier and perform hyper paramater optimization
        """

        logging.basicConfig(
            format='%(asctime)s:%(levelname)s:%(message)s',
            level=logging.INFO
        )

        # Load input/output matrix
        self.load_matrix()

        eval = self.contruct_cost_function(
            self.input_matrix,
            self.output_matrix
        )

        # Perform hyper paramater optimization
        best_parameters = fmin(
            fn=eval,
            space=hp.lognormal("c", 0, 1),
            algo=tpe.suggest,
            max_evals=self.max_evals,
        )

        best_c = best_parameters["c"]

        # Train SVM
        logging.info("Training SVM")
        classifier = svm.SVC(best_c, kernel="linear")
        classifier.fit(self.input_matrix, self.output_matrix)

        # Dumping trained SVM
        pickle.dump(classifier, open(self.clf_path, "wb"))

    def load_clf(self):
        """ Load the pre-trained classifier """

        logging.info("Loading classifier data")

        if (not(os.path.exists(self.clf_path))):
            raise Exception("Pre trained classifier not found")

        if (not(os.path.exists(self.labels_path))):
            raise Exception("Labels for classifier not found")

        self.classifier = pickle.load(open(self.clf_path, "rb"))
        self.labels = pickle.load(open(self.labels_path, "rb"))

    def predict(self, document):
        """
        Predict the class of given text
        """

        # Check and load classifier data
        self.load_clf()

        if (document == ""):
            raise Exception("Empty text provided")

        tf_idf_vector = self.vectorizer.obtain_feature_vector(document)

        predicted_output = self.classifier.predict(tf_idf_vector)[0]
        return(self.labels[int(predicted_output)])
