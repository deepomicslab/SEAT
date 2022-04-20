# SEAT
Structure Entropy hierArchy deTection

The SEAT packages provides sklearn style API for structure entropy based hierarchy detection and clustering.


## Prerequisite
+ numpy>=1.21
+ scikit-learn>=0.23.1


## Quick Start


### Run `SEAT.SEAT`
This example shows the usage of `SEAT.SEAT`. The API is similar to clustering algorithms provided in `sklearn.cluster` except `SEAT.SEAT` automatically tuning the best cluster number `k` associated with the minimal structural entropy in the structure entropy tree.

```Python
from sklearn.neighbors import kneighbors_graph
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from SEAT import SEAT

X = np.array([[1, 1], [1, 2], [10, 11],
              [9, 12], [5, 7], [6, 6]])
graph = kneighbors_graph(X, n_neighbors=3).toarray()

seat = SEAT()
seat.fit_predict(graph)
print('The best k is:', seat.optimal_k)
print('SE clustering result: \n', seat.labels_)
print('Candidate k for tuning: \n', seat.ks)
print('SE score for tuning k: \n', seat.se_scores)

# plot the structure entropy agglomerative tree
label = seat.labels_
label_colors = dict(zip(set(label), sns.color_palette('Spectral', len(set(label)))))
label_colors = [label_colors[l] for l in label]
g = sns.clustermap(X,
                   row_linkage=seat.Z_,
                   col_cluster=False,
                   row_colors=label_colors,
                   cmap='YlGnBu')
```
Outputs:

![SEAT](https://raw.githubusercontent.com/deepomicslab/SEAT/main/readme_fig2.png)

## class API

###  `class SEAT.SEAT`
Parameters:
> + `min_k`: The minimal number of clusters for searching, default 2.
> + `max_k`: The maximal number of clusters for searching, default 10 or the number of submodules in the tree.
> + `affinity`: Metric used to compute the similarity linkage. Currently, only “precomputed” available. If “precomputed”, a similarity matrix is needed as input for the fit method.

Attributes:
> + `optimal_k`: The best cluster number `k` associated with the minimal structural entropy in the structure entropy agglomerative tree.
> + `labels_`: Cluster labels for the `optimal_k`.
> + `Z_`: The linkage matrix used to plot the dendrogram.
> + `ks`: The list of candicate `k` for tuning.
> + `se_scores`: The structure entropy score for each `k`.

## Update

+ `v0.0.1.1`: realsed at 2021/04/20, the manuscript version.
+ `v0.0.1.1`: realsed at 2021/10/07, the initial version.
