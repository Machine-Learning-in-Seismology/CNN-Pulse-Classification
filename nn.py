from MaLeNeuralNetworkFactory import MaLeNeuralNetworkFactory
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import pickle, time, sys

''' 
Give the noise information
1- sabetta 
2- w1std 
3- w2std 
4- w3std
Give the synthetic method information
1- k2
2- mavro
'''
noise = sys.argv[1]
s_method = sys.argv[2]

EPOCHS = [200]

print("Loading labels...")
with open("/data/data_real/labels.pkl".format(n_arg), "rb") as f:
	y = pickle.load(f)

print("Loading dataset...")
with open("/data/data_real/inputs.pkl".format(n_arg), "rb") as f:
	x = pickle.load(f)
with open("data/data_" + noise + "/" + s_method + "/synthetic.pkl".format(n_arg), "rb") as f:
	syn = pickle.load(f)
results = pd.DataFrame(columns=['type', 'seed', 'epoch', 'fold', 'fnr', 'fpr', 'acc'])
timestr = time.strftime("%Y%m%d-%H%M%S")


def do_learning(x, y, syn):
	mask = np.ones(len(y), dtype=bool)
	ny = y[:, 0]
	ny = ny[mask]
	nx = x[mask, :]
	no = np.ones(len(syn))

	negatives = np.where(ny == 0)[0]
	x = np.array(list(x[negatives]))

	y = np.concatenate((np.zeros(x.shape[0]),np.ones(syn.shape[0])))
	x = x[:syn.shape[0]]
	x = np.append(x, syn, axis=0)


	kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
	fold = 0
	for train_index, test_index in kfold.split(x, y):
		x_train, x_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]

		for seed in range(5):
			for epoch in EPOCHS:
				network = MaLeNeuralNetworkFactory.get_network('MaLeConvSeq', x_train.shape, 100)
				x_train2, x_test2 = network.input_scale(x_train, x_test)
				unique, counts = np.unique(y_train, return_counts=True)
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

				results.loc[len(results)] = ['MaLeConvSeq', seed, e, fold, fnr, fpr, score[5]]
				results.to_csv("results/results" + noise + "-" + s_method + "-" + timestr + ".csv", sep="\t", encoding='utf-8')
				network.save("models/" + noise + "-" + s_method + "-" + str(fold) + "-" + str(seed) + ".h5")
		fold += 1


do_learning(x, y, syn)
