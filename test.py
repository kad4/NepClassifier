from NepClassifier import SVMClassifier


def run_batch(total_runs=1):
    clf = SVMClassifier()

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

if __name__ == '__main__':
    run_batch(5)
