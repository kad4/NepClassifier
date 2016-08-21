import os
import logging
import pickle

from sklearn import svm
from sklearn import cross_validation
from sklearn.utils import shuffle

from hyperopt import fmin, tpe, hp

from .tfidf import TfidfVectorizer
from .word2vec import Word2VecVectorizer


class SVMClassifier():
    """
    SVM based classifier for Nepali text
    """

    def __init__(self, word2vec=False):
        # Base path
        self.base_path = os.path.dirname(__file__)

        # Path for pre-trained classifier and labels
        self.clf_path = os.path.join(self.base_path, "svm.pkl")

        # Initialize vectorizer to use
        if word2vec:
            self.vectorizer = Word2VecVectorizer()
        else:
            self.vectorizer = TfidfVectorizer()

        # Classifier to use
        self.classifier = None

        # Test data size
        self.test_data_size = 0.33

        self.max_evals = 10
        self.cross_validation_folds = 5

    def contruct_cost_function(self, input_matrix, output_matrix):
        def eval(c):

            logging.info("Training using c={}".format(c))

            # Construct classifier
            classifier = svm.SVC(c, kernel="linear")

            scores = cross_validation.cross_val_score(
                classifier,
                input_matrix,
                output_matrix,
                cv=self.cross_validation_folds
            )

            score = scores.mean()
            logging.info("Mean score={}".format(score))

            # Return loss
            return 1-score

        return eval

    def train(self, documents, labels):
        """
        Train classifier and perform hyper paramater optimization
        """

        if len(documents) != len(labels):
            raise Exception("No of documents doesn't match the number of labels")

        logging.basicConfig(
            format='%(asctime)s:%(levelname)s:%(message)s',
            level=logging.INFO
        )

        # Obtain corpus data
        logging.info("Shuffling dataset")
        documents, labels = shuffle(documents, labels)

        logging.info("Obtaining feature matrix for data")
        self.input_matrix = self.vectorizer.obtain_feature_matrix(documents)

        logging.info("Constructing evaluation function for hyper paramater optimization")
        eval = self.contruct_cost_function(
            self.input_matrix,
            labels
        )

        # Perform hyper paramater optimization
        best_parameters = fmin(
            fn=eval,
            space=hp.lognormal("c", 0, 1),
            algo=tpe.suggest,
            max_evals=self.max_evals,
        )

        best_c = best_parameters["c"]

        logging.info("Best value obtained for c={}".format(best_c))

        # Train SVM
        logging.info("Training SVM".format(best_c))
        classifier = svm.SVC(best_c, kernel="linear")
        classifier.fit(self.input_matrix, labels)

        # Dumping trained SVM
        pickle.dump(classifier, open(self.clf_path, "wb"))

    def load_clf(self):
        """ Load the pre-trained classifier """

        logging.info("Loading classifier data")

        if (not(os.path.exists(self.clf_path))):
            raise Exception("Pre trained classifier not found")

        self.classifier = pickle.load(open(self.clf_path, "rb"))

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
        return predicted_output
