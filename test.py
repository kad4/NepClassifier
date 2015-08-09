from classifier import NepClassifier

clf = NepClassifier()

def single_run():
    print('Training classifier')
    clf.train()

    print('Loading classifier')
    clf.load_clf()
    
    print('Evaluating model')
    pre, rec, fs, acc, __ = clf.evaluate_model()

    return (pre, rec, fs, acc)

def batch_run(total_runs = 4):
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
    	pre, rec, fs, acc = single_run()

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
    clf.load_dataset()

    # Baseline method
    clf.load_corpus_info()

    print('Training classifier')
    clf.train()
    clf.load_clf()

    print('Evalutaing model')
    prec, rec, fs, acc, conf_mat = clf.evaluate_model()

    print('Accuracy of baseline : ', acc)
    
    # New method

if __name__ == '__main__':
    compare()
