"""Defines the problem features for use in the analysis and instance space. All functions in this script are scraped via the :py:mod:`inspect` module. See the source code for implementation details.

.. todo::
    Standardize format with e.g. a wrapper class.
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

def silhouette(data, labels, precomp_dict, metric="sqeuclidean"):
    if f"dists_{metric}" in precomp_dict:
        return silhouette_score(precomp_dict[f"dists_{metric}"], labels, metric="precomputed")
    else:
        return silhouette_score(data, labels, metric=metric)

def overlap(data, labels, precomp_dict, metric="sqeuclidean"):
    dists = precomp_dict.get(
        "dists_" + metric,
        squareform(pdist(data, metric=metric))
    )
    masked_dists = np.ma.argmin(
        np.ma.MaskedArray(dists, mask=dists == 0), axis=1
    )
    olap = 1 - (np.sum(labels == labels[masked_dists])/len(labels))
    return olap

def dimensionality(data, *args):
    return data.shape[1]

# def num_clusters(data, labels):
#     return int(np.unique(labels).shape[0])