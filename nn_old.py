import numpy as np
import pickle
import time

import pandas as pd
from MaLeNeuralNetworkFactory import MaLeNeuralNetworkFactory

EPOCHS = [200]

print "Loading labels..."
with open("data/labels.pkl", "rb") as f:
    y = pickle.load(f)

print "Loading origins..."
with open("data/real.pkl", "rb") as f:
    o = pickle.load(f)

print "Loading dataset..."
with open("data/inputs.pkl", "rb") as f:
    x = pickle.load(f)

results = pd.DataFrame(columns=['type', 'seed', 'epoch', 'fold', 'fnr', 'fpr', 'acc'])
timestr = time.strftime("%Y%m%d-%H%M%S")


def folding(y, o, n_folds):
    negatives = np.where(y == 0)[0]
    positives = np.where(y == 1)[0]
    synthetics = np.where(o == 1)[0]
    remove = np.isin(positives, synthetics)
    synthetics = positives[remove]
    positives = positives[~remove]
    np.random.shuffle(negatives)
    train_neg_len = synthetics.shape[0]
    step = abs(negatives.shape[0]-train_neg_len)/n_folds

    folds = []

    for i in range(n_folds):
        mask = np.ones(negatives.shape, dtype=bool)
        mask[(i * step): (i * step + train_neg_len)] = False
        testing = negatives[mask]
        training = negatives[~mask]
        testing = np.append(testing, positives)
        np.random.shuffle(synthetics)
        training = np.append(training, synthetics)
        np.random.shuffle(training)
        folds.append((training, testing))
    return folds


def do_learning(x, y, o):
    # removed = np.where(np.max(np.abs(x), axis=1) < 30)[0]
    mask = np.ones(len(y), dtype=bool)
    # mask[removed] = False
    ny = y[mask]
    nx = x[mask, :]
    no = o[mask]

    negatives = np.where(ny == 0)[0]
    synthetics = np.where(no == 0)[0]
    pos_len = nx.shape[0] - negatives.shape[0]
    print "# pos:", pos_len, "# neg:", negatives.shape[0], "# syn:", synthetics.shape[0]

    networks = ['MaLeConvSeq']
    for n in networks:
        fold = 0

        for train_index, test_index in folding(ny, no, 10):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            for seed in range(5):
                for epoch in EPOCHS:
                    network = MaLeNeuralNetworkFactory.get_network(n, x_train.shape, 100)
                    x_train2, x_test2 = network.input_scale(x_train, x_test)
                    e = network.train(x_train2, y_train, epoch)
                    score = network.evaluate(x_test2, y_test)
                    print "fold: %d FNR: %.2f FPR: %.2f ACC: %.2f" % (
                        fold, score[3] / float(score[3] + score[4]), score[1] / float(score[1] + score[2]), score[5])
                    results.loc[len(results)] = [n, seed, e, fold, score[3] / float(score[3] + score[4]),
                                                 score[1] / float(score[1] + score[2]),
                                                 score[5]]
                    results.to_csv("results/results" + timestr + ".csv", sep="\t", encoding='utf-8')
                    network.save("models/" + n + "-" + str(fold) + "-" + str(seed) + ".h5")
            fold += 1


do_learning(x, y, o)
