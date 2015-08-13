from classifier import NepClassifier

def run_batch(total_runs = 1):
    clf = NepClassifier()

    # print('Processing corpus')
    # clf.process_corpus()

    print('Corpus info loaded')
    clf.load_corpus_info()

    pres = []
    recs = []
    fss = []
    accs = []

    for i in range(total_runs):
    	# Load dataset 
        clf.load_dataset()

    	# Train and evaluate 
        clf.train()

        clf.load_clf()
        pre, rec, fs, acc, __ = clf.evaluate_model()

        pres.append(pre)
        recs.append(rec)
        fss.append(fs)
        accs.append(acc)

        print('Accuracy : ', acc)

    print('Precisions : ')
    print(pres)

    print('Recalls : ')
    print(recs)

    print('Fscores : ')
    print(fss)

    print('Accuracies : ')
    print(accs)

def compare():
    clf = NepClassifier()

	# print('Processing corpus')
    # clf.process_corpus()

    # Fixed dataset
    clf.load_dataset()

    # Baseline method
    clf.load_corpus_info()

    print('Training classifier')
    clf.train()
    clf.load_clf()

    print('Evaluating model')
    prec, rec, fs, acc, conf_mat = clf.evaluate_model()

    print('Accuracy of baseline : ', acc)
    print('Confusion Matrix :\n', conf_mat)
    
    # New method

if __name__ == '__main__':
    run_batch()
