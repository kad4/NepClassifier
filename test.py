from classifier import NepClassifier

def main():
    clf = NepClassifier()

    # print('Processing Corpus')
    # clf.process_corpus()
    
    print('Loading corpus info')
    clf.load_corpus_info()

    print('Loading dataset')
    clf.load_dataset()
    
    print('Training classifier')
    clf.train()

    print('Loading classifier')
    clf.load_clf()
    
    print('Evaluating model')
    pre, rec, fs, acc = clf.evaluate_model()

    print('Precision : ', pre)
    print('Recall : ', rec)
    print('fscores : ', fs)
    print('Accuracy : ', acc)

if __name__ == '__main__':
    main()