import os
import logging
import pickle

from statistics import mean

from sklearn import svm
from sklearn import cross_validation
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from hyperopt import fmin, tpe, hp


logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)
hdlr = logging.FileHandler("svm.log")
logger.addHandler(hdlr)


def calculate_scores(estimator, x, y):

    # Calculate the predicted output
    y_pred = estimator.predict(x)

    accuracy = accuracy_score(y, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y, y_pred)

    logger.info("Accuracy Score: {}".format(accuracy))
    logger.info("Precision: {}".format(mean(precision)))
    logger.info("Recall: {}".format(mean(recall)))
    logger.info("F-Score: {}".format(mean(fscore)))

    return accuracy


class SVMClassifier():
    """
    SVM based classifier for Nepali text
    """

    def __init__(self, vectorizer):
        # Base path
        self.base_path = os.path.dirname(__file__)

        # Path for pre-trained classifier and labels
        self.clf_path = os.path.join(self.base_path, "svm.pkl")

        # Assign vectorizer to class
        self.vectorizer = vectorizer

        # Classifier to use
        self.classifier = None

        # Test data size
        self.test_data_size = 0.33

        self.max_evals = 10
        self.cross_validation_folds = 5

    def contruct_cost_function(self, input_matrix, output_matrix):
        def eval(c):

            logger.info("Training using c={}".format(c))

            # Construct classifier
            classifier = svm.SVC(c, kernel="linear")

            scores = cross_validation.cross_val_score(
                classifier,
                input_matrix,
                output_matrix,
                cv=self.cross_validation_folds,
                scoring=calculate_scores
            )

            score = scores.mean()
            logger.info("Mean score={}".format(score))

            # Return loss
            return 1-score

        return eval

    def train(self, documents, labels):
        """
        Train classifier and perform hyper paramater optimization
        """

        if len(documents) != len(labels):
            raise Exception("No of documents doesn't match the number of labels")


        # Obtain corpus data
        logger.info("Shuffling dataset")
        documents, labels = shuffle(documents, labels)

        logger.info("Obtaining feature matrix for data")
        self.input_matrix = self.vectorizer.obtain_feature_matrix(documents)

        logger.info("Constructing evaluation function for hyper paramater optimization")
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

        logger.info("Best value obtained for c={}".format(best_c))

        # Train SVM
        logger.info("Training SVM".format(best_c))
        classifier = svm.SVC(best_c, kernel="linear")
        classifier.fit(self.input_matrix, labels)

        # Dumping trained SVM
        pickle.dump(classifier, open(self.clf_path, "wb"))

    def load_clf(self):
        """ Load the pre-trained classifier """

        logger.info("Loading classifier data")

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
