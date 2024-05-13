

import csv
import numpy as np
import matplotlib.pyplot as plt
import os.path

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans

data_save_root = r'G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\1. Analysis\all\20230904\LD plots\data'
session_id = '35_1025'

shuffle_N = 10000

# read data
csv.reader

with open(session_id + '.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',', quoting=csv.QUOTE_NONNUMERIC)
    output = []
    for row in csv_reader:
          output.append(row[:])

output = np.asarray(output)
X = output[:, :-1]
y = output[:, -1]
pca = PCA(n_components=30)
X = pca.fit(X).transform(X)
#

#

# perform lda & kmeans clustering

target_names = ['SS', 'LS', 'SL']
lda = LinearDiscriminantAnalysis(n_components=2)
replace_pairs = np.asarray([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]])
kmeans_performance = np.zeros(shuffle_N+1)

for shuffle_iter in range(shuffle_N+1):
     
     if shuffle_iter == 0:
          current_y = y

     else:
          current_y = np.random.permutation(y)
        
     current_X_r2 = lda.fit(X, current_y).transform(X)

     kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(current_X_r2)
     kmeans_output = kmeans.labels_
     kmeans_correctness = np.zeros(np.size(replace_pairs, axis=0))

     for row in range(0, np.size(replace_pairs, axis=0)):
            
         kmeans_output2 = kmeans_output + 100

         kmeans_output2[kmeans_output2 == 100] = replace_pairs[row, 0]
         kmeans_output2[kmeans_output2 == 101] = replace_pairs[row, 1]
         kmeans_output2[kmeans_output2 == 102] = replace_pairs[row, 2]

         kmeans_correctness[row] = sum(kmeans_output2 == current_y) / len(y)

     kmeans_performance[shuffle_iter] = kmeans_correctness.max()

#

# compute significance of kmeans clustering performance

kmeans_performance_real = kmeans_performance[0]
kmeans_performance_shuffle = kmeans_performance[1:]

kmeans_p = sum(kmeans_performance_real <= kmeans_performance_shuffle) / shuffle_N
print(kmeans_performance_real)
print(kmeans_p)
#

# lda & clustering for data saving
X_r2 = lda.fit(X, y).transform(X)

kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X_r2)
kmeans_output = kmeans.labels_
#

# save data
np.savez(os.path.join(data_save_root + '/' + session_id), kmeans_performance_real=kmeans_performance_real, 
         kmeans_performance_shuffle=kmeans_performance_shuffle, kmeans_p=kmeans_p,
         X_r2=X_r2, y=y, kmeans_output=kmeans_output)
#


if False:
    # plot trials in LD plane

    # pca

    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    print(
        "explained variance ratio (first two components): %s"
        % str(pca.explained_variance_ratio_)
    )

    plt.figure()
    colors = ["navy", "turquoise", "darkorange"]
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(
            X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of IRIS dataset")
    #


    # lda
    plt.figure()

    colors = ["navy", "turquoise", "darkorange"]
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(
            #X_r2[y == i, 0], X_r2[y == i, 1], alpha=0.8, color=color, label=target_name
            current_X_r2[current_y == i, 0], current_X_r2[current_y == i, 1], alpha=0.8, color=color, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("LDA plotting (shuffle) dataset")
    #

    # LDA real clustering
    plt.figure()

    colors = ["navy", "turquoise", "darkorange"]
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(
            X_r2[y == i, 0], X_r2[y == i, 1], alpha=0.8, color=color, label=target_name
            #current_X_r2[kmeans_output == i, 0], current_X_r2[kmeans_output == i, 1], alpha=0.8, color=color, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("LDA plotting (real) dataset")
    #

    # k means clustering
    plt.figure()

    colors = ["navy", "turquoise", "darkorange"]
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(
            X_r2[kmeans_output == i, 0], X_r2[kmeans_output == i, 1], alpha=0.8, color=color, label=target_name
            #current_X_r2[kmeans_output == i, 0], current_X_r2[kmeans_output == i, 1], alpha=0.8, color=color, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("K means clustering result")
    #

    plt.show()
