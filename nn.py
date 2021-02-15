import numpy as np
import pickle
import time
import pandas as pd
# import matplotlib.pyplot as plt
from MaLeNeuralNetworkFactory import MaLeNeuralNetworkFactory

EPOCHS = [20]

print("Loading labels...")
print('Synthetic (0=Real,1=Synthetic), Manual_Pulse, Baker_Tp, Chang_Tp, Deniz_Tp')
with open("data/labels.pkl", "rb") as f:
    y = pickle.load(f)

print("Loading dataset...")
with open("data/inputs.pkl", "rb") as f:
    x = pickle.load(f)

with open("data/synthetic.pkl", "rb") as f:
    syn = pickle.load(f)

results = pd.DataFrame(columns=['type', 'seed', 'epoch', 'fold', 'fnr', 'fpr', 'acc'])
timestr = time.strftime("%Y%m%d-%H%M%S")


def folding(y, o, n_folds):
    negatives = np.where(y == 0)[0]
    positives = np.where((y == 1))[0]
    synthetics = np.where(o == 1)[0]
    # remove = np.isin(positives, synthetics)
    # synthetics = positives[remove]
    # positives = positives[~remove]
    # synthetics = synthetics + len(y)
    np.random.shuffle(negatives)
    train_neg_len = positives.shape[0]
    # train_neg_len = len(negatives) - len(positives)
    step = abs(negatives.shape[0] - train_neg_len) // n_folds
    folds = []

    for i in range(n_folds):
        mask = np.ones(negatives.shape, dtype=bool)
        mask[(i * step): (i * step + train_neg_len)] = False
        testing = negatives[mask]
        training = negatives[~mask]
        testing = np.append(np.random.choice(np.where(y == 0)[0],positives.shape[0]), positives)
        # testing = np.append(testing, positives)
        np.random.shuffle(synthetics)
        training = np.append(training, synthetics)
        np.random.shuffle(training)
        folds.append((training, testing))
    return folds


def do_learning(x, y, syn):
    mask = np.ones(len(y), dtype=bool)
    # print()
    ny = y[:, 0]
    ny = ny[mask]
    nx = x[mask, :]
    no = np.ones(len(syn))
    # no = syn
    # y_man = y[:,1]
    # y_syn = y[:,0]
    # ny = y_man[mask]
    # no = y_syn[mask]
    # nx = x[mask, :]

    x = np.append(x, syn, axis=0)
    y = np.append(ny, no)
    negatives = np.where(ny == 0)[0]
    synthetics = np.where(no == 1)[0]
    pos_len = nx.shape[0] - negatives.shape[0]
    print("# pos:", pos_len, "# neg:", negatives.shape[0], "# syn:", synthetics.shape[0])

    networks = ['MaLeConvSeq']
    for n in networks:
        fold = 0
        for train_index, test_index in folding(ny, no, 10):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print(fold)
            print(len(np.where(y_train == 0)[0]), len(np.where(y_train == 1)[0]))
            print(len(np.where(y_test == 0)[0]), len(np.where(y_test == 1)[0]))

            for seed in range(5):
                for epoch in EPOCHS:
                    network = MaLeNeuralNetworkFactory.get_network(n, x_train.shape, 100)
                    x_train2, x_test2 = network.input_scale(x_train, x_test)
                    # tmp = np.where(y_train == 1)
                    # import matplotlib.pyplot as plt
                    # print(x_train[0].shape)
                    # for i in tmp[0]:
                    #     print(i)
                    #     plt.plot(x_train[i])
                    #     plt.show() 
                    e = network.train(x_train2, y_train, epoch)
                    score = network.evaluate(x_test2, y_test)
                    print(score)
                    ''' Score:
                    0 - 
                    1 - FP
                    2 - TN
                    3 - FN
                    4 - TP
                    5 - ACC
                    '''
                    print("fold: %d FNR: %.2f FPR: %.2f ACC: %.2f" % (
                        fold,
                        score[3] / float(score[3] + score[4]),
                        score[1] / float(score[1] + score[2]),
                        score[5])
                          )
                    fnr = score[3] / float(score[3] + score[4])
                    fpr = score[1] / float(score[1] + score[2])
                    # if float(score[3] + score[4]) == 0:
                    #     fnr = 0
                    # else:
                    #     fnr = score[3] / float(score[3] + score[4])
                    # if float(score[1] + score[2]) == 0:
                    #     fpr = 0
                    # else:
                    #     fpr = score[1] / float(score[1] + score[2])

                    results.loc[len(results)] = [n, seed, e, fold, fnr, fpr, score[5]]
                    results.to_csv("results/results" + timestr + ".csv", sep="\t", encoding='utf-8')
                    # network.save("models/" + n + "-" + str(fold) + "-" + str(seed) + ".h5")
            fold += 1


do_learning(x, y, syn)
