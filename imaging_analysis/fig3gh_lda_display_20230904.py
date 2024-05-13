

import numpy as np
import matplotlib.pyplot as plt
import os.path

from sklearn.cluster import KMeans

input_data_root = r'G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\1. Analysis\all\20230904\LD plots\data'
session_id = '35_1025'
# '20_0519', '25_0630', '30_1017', '31_1029', '33_1025', '35_1025'

# load data
input_data = np.load(os.path.join(input_data_root + '\\' + session_id + '.npz'))

kmeans_p = input_data['kmeans_p']
kmeans_performance_real = input_data['kmeans_performance_real']
kmeans_performance_shuffle = input_data['kmeans_performance_shuffle']
X_r2 = input_data['X_r2']
y = input_data['y']
kmeans_output = input_data['kmeans_output']
#

# plot 

# setting
mesh_margin = 0.5

x_min, x_max = X_r2[:,0].min() - mesh_margin, X_r2[:,0].max() + mesh_margin
y_min, y_max = X_r2[:,1].min() - mesh_margin, X_r2[:,1].max() + mesh_margin
#

# LD plane, real data
plt.figure()

colors = ["navy", "turquoise", "darkorange"]
target_names = ["S-S", "L-S", "S-L"]

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        X_r2[y == i, 0], X_r2[y == i, 1], alpha=0.8, color=color, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("LDA plotting (real) dataset")

ax = plt.gca()
ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])
#

# LD plane, k means clustering boundary
plt.figure()

h = 0.02 # point in mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X_r2)

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.imshow(Z, interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        X_r2[kmeans_output == i, 0], X_r2[kmeans_output == i, 1], alpha=0.8, color=color, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("K means clustering result")

ax = plt.gca()
ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])
#

# histogram shuffle vs real clustering performance
plt.figure()

hist_bin = np.asarray(range(30, 100, 5)) / 100
temp_x, temp_bins, p_hist = plt.hist(kmeans_performance_shuffle, hist_bin, density=True)

for item in p_hist:
     item.set_height(item.get_height()/sum(temp_x))

plt.plot([kmeans_performance_real, kmeans_performance_real], [0, 0.1])
ax = plt.gca()
ax.set_ylim([0, 0.4])
#

plt.show()
#