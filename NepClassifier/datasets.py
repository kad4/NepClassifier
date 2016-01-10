import os
import gzip
import pickle


class NewsData():
    """
        News data for classifier
    """

    def load_data():
        # Base path to use
        base_path = os.path.dirname(__file__)

        # Path for data set
        data_path = os.path.join(base_path, "newsdata.pkl.gz")

        # Return the obtained data
        documents, labels = pickle.load(gzip.open(data_path, 'rb'))
        return (documents, labels)
