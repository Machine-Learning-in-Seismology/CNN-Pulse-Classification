import numpy as np
import pickle
import time
import pandas as pd
import sys
# import matplotlib.pyplot as plt
from MaLeNeuralNetworkFactory import MaLeNeuralNetworkFactory
import matplotlib.pyplot as plt
from scipy import integrate
from sklearn.model_selection import StratifiedKFold
EPOCHS = [20]

n_arg = sys.argv[1]
''' Format
n = Signals normalized by st.normalize
r = Signals not normalized
'''

print("Loading labels...")
print('Synthetic (0=Real,1=Synthetic), Manual_Pulse, Baker_Tp, Chang_Tp, Deniz_Tp')
with open("data/labels_{}.pkl".format(n_arg), "rb") as f:
	y = pickle.load(f)

print("Loading dataset...")
with open("data/inputs_{}.pkl".format(n_arg), "rb") as f:
	x = pickle.load(f)
#x = x[:,:799]
with open("data/mavro_synthetic_{}.pkl".format(n_arg), "rb") as f:
#with open("data/synthetic_output10.pkl", "rb") as f:
	syn = pickle.load(f)
#syn = syn[:,:799]
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
	x = np.append(x, syn, axis=0)

	# for i in range(x.shape[0]):
	# 	plt.plot(x[i,:])
	# 	plt.show()
	# 	plt.close('all')

	# for i in range(syn.shape[0]):
	# 	plt.plot(syn[i,:])
	# 	plt.show()
	# 	plt.close('all')

	# y = np.append(ny, no)
	# # negatives = np.where(ny == 0)[0]
	# # synthetics = np.where(no == 1)[0]
	# # pos_len = nx.shape[0] - negatives.shape[0]
	# # print("# pos:", pos_len, "# neg:", negatives.shape[0], "# syn:", synthetics.shape[0])

	kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
	fold = 0
	for train_index, test_index in kfold.split(x, y):
		x_train, x_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]
		# print(fold)
		# print(len(np.where(y_train == 0)[0]), len(np.where(y_train == 1)[0]))
		# print(len(np.where(y_test == 0)[0]), len(np.where(y_test == 1)[0]))

		for seed in range(5):
			for epoch in EPOCHS:
				network = MaLeNeuralNetworkFactory.get_network('MaLeConvSeq', x_train.shape, 100)
				# x_train2, x_test2 = network.input_scale(x_train, x_test)

				e = network.train(x_train, y_train, epoch)
				score = network.evaluate(x_test, y_test)
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

				results.loc[len(results)] = [n, seed, e, fold, fnr, fpr, score[5]]
				results.to_csv("results/results" + n_arg + timestr + ".csv", sep="\t", encoding='utf-8')
				#network.save("models/" + n_arg + "-" + str(fold) + "-" + str(seed) + ".h5")
		fold += 1


do_learning(x, y, syn)
