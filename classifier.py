import os
import random
import pickle
import logging
from math import log
from pathlib import Path
from operator import itemgetter
from statistics import mean

import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

from stemmer import NepStemmer

class NepClassifier():
    """ 
        Classifier for Nepali News

        It categorizes the given text in one among the following groups:

        economy, entertainment, news, politics, sports, world
    """

    def __init__(self):
        # Stemmer to use
        self.stemmer = NepStemmer()

        # Base path to use
        self.base_path = os.path.dirname(__file__)

        # Folder containing data
        self.folder_path = os.path.join(self.base_path, 'data')

        # File containing the corpus info
        self.data_path = os.path.join(self.base_path, 'data.p')

        # Training data size
        self.train_data_size = 10000
        
        # Test data size
        self.test_data_size = 1000

        # Maximum stems to use
        self.max_stems = 1000

        # Stems to use as feature
        self.stems = None

        # Vector to hold the IDF of stems
        self.idf_vector = None

        # Training data to use
        self.train_data = []
        self.test_data = []

        # Document categories
        self.categories = [
            'economy', 
            'entertainment',
            'news',
            'politics', 
            'sports', 
            'world'
        ]

        # Classifier
        self.clf = None

    def process_corpus(self):
        """ 
            Class method to process corpus located at path provided 
            at self.folder_path

            The data must be organized as utf-8 encoded raw text file
            having following structure

            root/
                class1/
                    text11.txt
                    text12.txt
                class2/
                    text21.txt
                    text22.txt
                ...
        """

        # Vectors for stems
        count_vector = {}
        idf_vector_total = {}

        total_docs = 0

        for root, dirs, files in os.walk(self.folder_path):
            for file_path in files:
                abs_path = os.path.join(self.base_path, root, file_path)

                file = open(abs_path, 'r')
                content = file.read()
                file.close()

                # Obtain known stems
                doc_stems = self.stemmer.get_known_stems(content)
                doc_stems_set = set(doc_stems)

                # Add the count of stems
                for stem in doc_stems:
                    count_vector[stem] = count_vector.get(stem, 0) + 1

                for stem in doc_stems_set:
                    idf_vector_total[stem] = idf_vector_total.get(stem, 0) + 1

                total_docs += 1

        # Obtain frequently occuring stems
        stem_tuple=sorted(
            count_vector.items(),
            key = itemgetter(1),
            reverse = True
        )[:self.max_stems]
    
        # Construct a ordered list of frequent stems
        stems = [item[0] for item in stem_tuple]

        # IDF vector for the stems
        idf_vector = [
            log(total_docs / (1 + idf_vector_total[k])) 
            for k in stems
        ]
        
        # Dump the data obtained
        data = {
            'stems' : stems,
            'idf_vector' : idf_vector
        }

        pickle.dump(data, open(self.data_path, 'wb'))

    def load_corpus_info(self):
        """ Load the corpus information a file """

        # Load dump data
        data = pickle.load(open(self.data_path, 'rb'))
        
        self.stems = data['stems']
        self.idf_vector = data['idf_vector']

    def load_dataset(self):
        """
            Load training data from the path specified by
            self.folder_path

            The files are loaded as a dictionary similar to one
            given below
            doc = {
                'path' : '../data/text1.txt',
                'category' : 'news'
            }
        """

        documents = []
        for category in Path(self.folder_path).iterdir():

            # Convert path to posix notation
            category_name = category.as_posix().split('/')[-1]
            
            if (not(category_name in self.categories)):
            	continue

            for filepath in category.iterdir():
                documents.append({
                    'path' : filepath.as_posix(),
                    'category' : category_name
                })

        sample_docs = random.sample(
            documents,
            self.train_data_size + self.test_data_size
        )
        
        self.test_data = sample_docs[-self.test_data_size:]
        self.train_data = sample_docs[:-self.test_data_size]

    def tf_vector(self, text):
        """ Compute tf vector for a given text """

        # Find stems in document
        doc_stems = self.stemmer.get_known_stems(text)

        # Contruct dictionary of stems
        doc_vector = {}
        for stem in doc_stems:
            doc_vector[stem] = doc_vector.get(stem, 0) + 1

        # Convert dictionary into list
        doc_vector_list = [doc_vector.get(stem, 0) for stem in self.stems]

        max_count = max(doc_vector_list)
        if(max_count == 0):
            max_count = 1

        # Calculate the tf of text
        tf_vector = 0.5 + (0.5 / max_count) * np.array(
            [doc_vector.get(stem, 0) for stem in self.stems]
        )

        return(tf_vector)

    def compute_matrix(self, data):
        """ Compute the input and output matrix for given documents """

        stems_size = len(self.stems)
        docs_size = len(data)

        # Tf matrix
        tf_matrix = np.ndarray(
            (docs_size, stems_size),
            dtype = 'float16'
        )

        idf_matrix = np.array(self.idf_vector)

        # Training matrix
        input_matrix = np.ndarray(
            (docs_size, stems_size),
            dtype='float16'
        )

        output_matrix = np.ndarray((docs_size, 1), dtype = 'float16')

        # Loop to construct the training matrix
        for i, doc in enumerate(data):

            with open(doc['path'], 'r') as file: 
                content = file.read()
            
            # Compute the tf and append it
            tf_vector = self.tf_vector(content)
            tf_matrix[i, :] = tf_vector 

            output_matrix[i, 0] = self.categories.index(doc['category'])

        # Element wise multiplication
        for i in range(docs_size):
            input_matrix[i, :] = tf_matrix[i, :] * idf_matrix

        output_matrix = output_matrix.ravel()

        return (input_matrix, output_matrix)
    
    def train(self):
        """ Obtain the training matrix and train theclassifier """

        if (not(self.stems)):
            raise Exception('Corpus info not available.')

        if (not(self.train_data)):
            raise Exception('Training data not selected')

        input_matrix, output_matrix = self.compute_matrix(self.train_data)

        # Assign and train a SVM
        clf = svm.SVC(C = 50.0)
        clf.fit(input_matrix, output_matrix)

        # Dumping extracted data
        clf_file = os.path.join(self.base_path, 'clf.p')
        joblib.dump(clf, clf_file)
    
    def load_clf(self):
        """ Loads the trained classifier from file """

        if (not(self.stems)):
            self.load_dataset()

        clf_file = os.path.join(self.base_path, 'clf.p')
        self.clf = joblib.load(clf_file)

    def tf_idf_vector(self, text):
        """ Calculates the tf-idf vector for a given text """

        if (not(self.stems)):
            raise Exception('Corpus info not available')       

        tf_vector = self.tf_vector(text)

        tf_idf_vector = []
        for i in range(len(self.stems)):
            tf_idf_vector.append(tf_vector[i] * self.idf_vector[i])

        tf_idf_vector = tf_vector * np.array(self.idf_vector)

        return(tf_idf_vector)

    def predict(self, text):
        """ Predict the class of given text """

        if (not(self.clf)):
            raise Exception('Classifier not loaded')
        
        if (text == ''):
            raise Exception('Empty text provided')
        
        tf_idf_vector = self.tf_idf_vector(text)
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

if __name__ == '__main__':
	with open('test.txt', 'r') as file:
            content = file.read()

            # Initialize the classifier
            clf = NepClassifier()

            # Loads the corpus info
            clf.load_corpus_info()

            # Loads the trained classifier
            clf.load_clf()
            
            # Predicted category
            category = clf.predict(content)

            print('The category is : ', category)
