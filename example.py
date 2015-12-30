from NepClassifier import SVMClassifier


def main():
    with open('example.txt', 'r') as file:
        content = file.read()

    # Initialize the classifier
    clf = SVMClassifier()

    # Loads the trained classifier
    clf.load_clf()

    # Predicted category
    category = clf.predict(content)

    print('The category is : ', category)

if __name__ == '__main__':
    main()
