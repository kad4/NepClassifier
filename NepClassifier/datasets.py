import os
import gzip
import pickle


class NeptextData():
    """
    General Nepali text dataset
    """

    def load_data(use_all_data=True):
        # Base path to use
        base_path = os.path.dirname(__file__)

        # Path for data set
        data_path = os.path.join(base_path, "data", "neptext.pkl.gz")

        # Return the obtained data
        documents, labels = pickle.load(gzip.open(data_path, 'rb'))

        # Filter data if data is being used to train classifier
        if use_all_data:
            return (documents, labels)
        else:
            exclude = ["others"]

            documents, labels = zip(*(
                (document, label) for document, label in zip(documents, labels)
                if label not in exclude
            ))

            return documents, labels
