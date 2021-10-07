# SEAT
Structure entropy agglomerative tree (SEAT) for clustering


The SEAT packages provides sklearn style API for structure entropy based clustering.


## Prerequisite
+ numpy>=1.21
+ scikit-learn>=0.23.1


## Install
```shell
pip install pyseat
```

## Quick Start

### Run `SEAT.SEClustering`

This example shows the usage of `SEAT.SEClustering`. The API is similar to clustering algorithms provided in `sklearn.cluster`. 

```Python
from sklearn.neighbors import kneighbors_graph
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from pyseat.SEAT import SEClustering

X = np.array([[1, 1], [1, 2], [10, 11],
              [9, 12], [5, 7], [6, 6]])
graph = kneighbors_graph(X, n_neighbors=3).toarray()

secluster1 = SEClustering(n_clusters=3)
secluster1.fit(graph)
print('SE Clustering result for k=3: \n', secluster1.labels_)

secluster2 = SEClustering(n_clusters=[2,3])
secluster2.fit(graph)
print('SE Clustering result for k=[2,3]: \n', secluster2.labels_)


# plot the structure entropy agglomerative tree
label = secluster1.labels_
label_colors = dict(zip(set(label), sns.color_palette('Spectral', len(set(label)))))
label_colors = [label_colors[l] for l in label]
g = sns.clustermap(X,
                   row_linkage=secluster1.Z_,
                   col_cluster=False,
                   row_colors=label_colors,
                   cmap='YlGnBu')
```
Outputs:

![SEClustering](https://raw.githubusercontent.com/deepomicslab/SEAT/main/readme_fig1.png)

### Run `SEAT.SEAT`
This example shows the usage of `SEAT.SEAT`. The functionality of `SEAT.SEAT` is the same with `SEAT.SEClustering` except `SEAT.SEAT` automatically pruning the best cluster number `k` associated with the minimal structural entropy in the structure entropy agglomerative tree.

```Python
from sklearn.neighbors import kneighbors_graph
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from pyseat.SEAT import SEAT

X = np.array([[1, 1], [1, 2], [10, 11],
              [9, 12], [5, 7], [6, 6]])
graph = kneighbors_graph(X, n_neighbors=3).toarray()

seat = SEAT()
seat.fit_predict(graph)
print('The best k is:', seat.optimal_k)
print('SE clustering result: \n', seat.labels_)
print('Candates k for pruning: \n', seat.ks)
print('SE score for pruning k: \n', seat.se_scores)

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
###  `class SEAT.SEClustering`
Parameters:
> + `n_clusters`: The number of clusters to find, must be an integer or a list of integer.
> + `affinity`: Metric used to compute the similarity linkage. Currently, only “precomputed” available. If “precomputed”, a similarity matrix is needed as input for the fit method.

Attributes:
> + `labels_`: Cluster labels for each point.
> + `Z_`: The linkage matrix used to plot the dendrogram.

###  `class SEAT.SEAT`
Parameters:
> + `min_k`: The minimal number of clusters for searching, default 2.
> + `max_k`: The maximal number of clusters for searching, default 10 or the number of submodules in the tree.
> + `affinity`: Metric used to compute the similarity linkage. Currently, only “precomputed” available. If “precomputed”, a similarity matrix is needed as input for the fit method.

Attributes:
> + `optimal_k`: The best cluster number `k` associated with the minimal structural entropy in the structure entropy agglomerative tree.
> + `labels_`: Cluster labels for the `optimal_k`.
> + `Z_`: The linkage matrix used to plot the dendrogram.
> + `ks`: The list of candicate `k` for pruning.
> + `se_scores`: The structure entropy score for each `k`.

## Update

+ `v0.0.1.1`: realsed at 2021/10/07, the initial version seat.
