#%%
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
#plt.style.use('classic')
import numpy as np

k1706_8 = np.load("../data/plots/kernel_1706.npy")
k2015_8 = np.load("../data/plots/kernel2015.npy")
k1706_4 = np.load("../data/plots/k_1706_4angl.npy")
k2015_4 = np.load("../data/plots/k_2015_4angl.npy")
nn = np.load("../data/plots/nn.npy")
knn_1 = np.load("../data/plots/knn_1.npy")
knn_5 = np.load("../data/plots/knn_5.npy")
knn_100 = np.load("../data/plots/knn_100.npy")
# Plots

fig = plt.figure(facecolor=(1, 1, 1))
ax = plt.axes()

plt.plot(nn[0], nn[1], '-b', label='CNN')
plt.plot(k1706_8[0], k1706_8[1], '-g', label='SW-Kernel, 8 angles')
plt.plot(k2015_8[0], k2015_8[1], '-r', label='MS-Kernel, 8 angles')
plt.plot(k1706_4[0], k1706_4[1], '-c', label='SW-Kernel, 4 angles')
plt.plot(k2015_4[0], k2015_4[1], '-m', label='MS-Kernel, 4 angles')
plt.plot(knn_5[0], knn_5[1]/100, '-k', label='k-NN, k=5')
plt.plot(knn_100[0], knn_100[1]/100, '-k', label='k-NN, k=100')

plt.xlabel('Training-set size')
plt.ylabel('Accuracy')
plt.ylim([0.2, 1])
plt.legend();
plt.savefig('testplot.png')
#plt.show()
plt.savefig('../data/plots/accuracy_moreplots.png')
plt.show()
#
#plt.savefig("../data/plots/accuracy_moreplots.svg", format="svg", facecolor=(1,1,1))
