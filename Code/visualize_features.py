# -*- coding: utf-8 -*-

# Feature Selection Aide
# Visualizes the correlation between different features of given data
import numpy as np
import matplotlib.pyplot as plt


def visualize_features(data,y,features,save_name):
# data -> 2D array of data points
# y -> classification outputs (1 or -1)
# features -> 1D array of feature indexes to compare
# save_name -> name under which the figure is saved in local directory

    n = features.shape[0]

    f, axs = plt.subplots(n, n)
    f.set_size_inches(20,15)

    for ind1, i in enumerate(features):
        for ind2, j in enumerate(features):

            ax = axs[ind1][ind2]
            if (i==j):
                pass
            else:

                # change colors to see 2 different perspectives
                if ind1>ind2:
                    color1 = [1, 0.06, 0.06]
                    color2 = [0.06, 0.06, 1]
                else:
                    color2 = [1, 0.06, 0.06]
                    color1 = [0.06, 0.06, 1]

                # Scatter
                ax.scatter(data[np.where(y == 1),i],
                           data[np.where(y == 1),j],
                           color=color1, s=3, facecolor='none')
                ax.scatter(data[np.where(y == -1),i],
                           data[np.where(y == -1),j],
                           color=color2, s=3,facecolor='none')
                ax.set_xlabel("Feature {}".format(i))
                ax.set_ylabel("Feature {}".format(j))
                ax.grid()

    plt.tight_layout()
    plt.savefig(save_name,dpi=200)