import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

def silhouette(data, labels, metric="sqeuclidean"):
    return silhouette_score(
        data, labels, metric=metric)

def overlap(data, labels, metric="sqeuclidean"):
    dists = squareform(pdist(data, metric=metric))
    masked_dists = np.ma.argmin(
        np.ma.MaskedArray(dists, mask=dists == 0), axis=1
    )
    olap = 1 - (np.sum(labels == labels[masked_dists])/len(labels))
    return olap

def dimensionality(data, labels):
    return data.shape[1]

# def num_clusters(data, labels):
#     return int(np.unique(labels).shape[0])