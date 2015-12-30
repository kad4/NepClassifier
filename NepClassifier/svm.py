import os
import random
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

from .feature_extraction import TfidfVectorizer


class SVMClassifier():
    """
        Classifier for Nepali News

        It categorizes the given text in one among the following groups

        economy, entertainment, news, politics, sports, world
    """

    def __init__(self):
        # Feature extractor to use
        self.feature_extractor = TfidfVectorizer(max_stems=1000)
        self.feature_extractor.load_corpus_info()

        # Number of stems
        self.stems_size = self.feature_extractor.stems_size

        # Base path to use
        self.base_path = os.path.dirname(__file__)

        # Folder containing data
        self.corpus_path = os.path.join(self.base_path, 'data')

        # Classifier path
        self.clf_path = os.path.join(self.base_path, 'clf.p')

        # Training data size
        self.train_data_size = 10000

        # Test data size
        self.test_data_size = 1000

        # Training data to use
        self.train_data = None
        self.test_data = None

        # Document categories
        self.categories = [
            'economy',
            'entertainment',
            'news',
            'politics',
            'sports',
            'international'
        ]

        # Classifier
        self.clf = None

        # Regularization parameter
        self.C = 50.0

    def load_dataset(self):
        """
            Load training data from the path specified by
            self.corpus_path

            The files are loaded as a dictionary similar to one
            given below
            doc = {
                'path' : '../data/text1.txt',
                'category' : 'news'
            }
        """

        documents = []
        for category in Path(self.corpus_path).iterdir():

            # Convert path to posix notation
            category_name = category.as_posix().split('/')[-1]

            if (not(category_name in self.categories)):
                continue

            for filepath in category.iterdir():
                documents.append({
                    'path': filepath.as_posix(),
                    'category': category_name
                })

        sample_docs = random.sample(
            documents,
            self.train_data_size + self.test_data_size
        )

        self.test_data = sample_docs[-self.test_data_size:]
        self.train_data = sample_docs[:-self.test_data_size]

    def compute_matrix(self, data):
        """ Compute the input and output matrix for given documents """

        docs_size = len(data)

        input_matrix = np.ndarray(
            (docs_size, self.stems_size),
            dtype='float16'
        )

        output_matrix = np.ndarray((docs_size, 1), dtype='float16')

        # Loop to construct matrix
        for i, doc in enumerate(data):

            with open(doc['path'], 'r') as file:
                content = file.read()

                # Compute the tf-idf and append it
                input_matrix[i, :] = self.feature_extractor\
                    .tf_idf_vector(content)

                output_matrix[i, 0] = self.categories.index(doc['category'])

        output_matrix = output_matrix.ravel()

        return (input_matrix, output_matrix)

    def train(self):
        """ Obtain the training matrix and train the classifier """

        if (not(self.train_data)):
            raise Exception('Training data not selected')

        input_matrix, output_matrix = self.compute_matrix(self.train_data)

        # Assign and train a SVM
        clf = svm.SVC(self.C)
        clf.fit(input_matrix, output_matrix)

        # Dumping extracted data
        joblib.dump(clf, self.clf_path)

    def load_clf(self):
        """ Loads the trained classifier from file """

        self.clf = joblib.load(self.clf_path)

    def predict(self, text):
        """ Predict the class of given text """

        if (text == ''):
            raise Exception('Empty text provided')

        if (not(self.clf)):
            raise Exception('Classifier not loaded')

        tf_idf_vector = self.feature_extractor.tf_idf_vector(text)
        output_val = self.clf.predict(tf_idf_vector)[0]

        class_id = int(output_val)
        return (self.categories[class_id])

    def evaluate_model(self):
        """ Performs the model evaluation """

        if (not(self.clf)):
            raise Exception('Classifier not loaded')

        input_matrix, output_matrix = self.compute_matrix(self.test_data)

        pred_output = self.clf.predict(input_matrix)

        accuracy = accuracy_score(output_matrix, pred_output)

        precision, recall, fscore, __ = precision_recall_fscore_support(
            output_matrix,
            pred_output
        )

        precision = mean(precision)
        recall = mean(recall)
        fscore = mean(fscore)

        conf_mat = confusion_matrix(output_matrix, pred_output)

        return(precision, recall, fscore, accuracy, conf_mat)
