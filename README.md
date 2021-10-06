# SEAT
Structure entropy agglomerative tree (SEAT) for clustering


The SEAT packages provides sklearn style API for structure entropy based clustering.

# Getting started

## Prerequisite
+ numpy
+ sklearn


## Install
```shell
pip install seat
```

## Examples

Demo to use SEClustering class:
```Python
from SEAT import SEClustering

X = ...
connectivity = kneighbors_graph(
        X, n_neighbors=10)
secluster = SEClustering(n_clusters=[2,3,4])
secluster = SEClustering(n_clusters=2)
secluster.fit(connectivity)

print(secluster.labels_)
print(secluster.Z_)
```

Demo to use SEAK class:
```Python
from SEAT import SEAK

X = ...
connectivity = kneighbors_graph(
        X, n_neighbors=10)
seak = SEAK()
clusters = secluster.fit_predict(connectivity)

print(clusters)
print(seak.Z_)
print(seak.se_score)
```


## class API
###  ```class SEAK.SEClustering```
Parameters:
> + ```n_clusters```: The number of clusters to find, must be an integer or a list of integer.
> + ```affinity```: Metric used to compute the similarity linkage. Currently, only “precomputed” available. If “precomputed”, a similarity matrix is needed as input for the fit method.

Attributes:
> + ```labels_```: Cluster labels for each point.
> + ```Z_```: The linkage matrix used to plot the dendrogram.

###  ```class SEAK.SEAK```
Parameters:
> + ```min_k```: The minimal number of clusters for searching.
> + ```max_k```: The maximal number of clusters for searching.
> + ```affinity```: Metric used to compute the similarity linkage. Currently, only “precomputed” available. If “precomputed”, a similarity matrix is needed as input for the fit method.

Attributes:
> + ```labels_```: Cluster labels for each point.
> + ```Z_```: The linkage matrix used to plot the dendrogram.
> + ```se_score```: The structure entropy score for each `k`.
