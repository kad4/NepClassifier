from NepClassifier import NepClassifier

if __name__ == '__main__':
    with open('example.txt', 'r') as file:
        content = file.read()

        # Initialize the classifier
        clf = NepClassifier()

        # Loads the trained classifier
        clf.load_clf()
        
        # Predicted category
        category = clf.predict(content)

        print('The category is : ', category)
