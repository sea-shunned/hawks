"""
All functions in this script are scraped via the inspect module when checking the arguments given in the config under the 'constraints' object/subdict.

A single value should be returned for any constraint.
"""
import numpy as np

import hawks.objectives

def overlap(indiv):
    '''Calculate the amount of overlap (the percentage of points whose nearest neighbour is in a different cluster)
    
    Arguments:
        indiv {Genotype} -- An individual (i.e. dataset)
    
    Returns:
        float -- The percentage of overlap
    '''
    # Check if the individual has the distances already
    if not hasattr(indiv, "distances"):
        indiv.distances = hawks.objectives.Silhouette.calc_distances(indiv.all_values)
    # Use a masked array to ignore the 0s on the diagonal (but retain shape)
    masked_dists = np.ma.argmin(
        np.ma.MaskedArray(indiv.distances, mask=indiv.distances == 0), axis=1) 
    # Sum the number of Trues we get for this condition
    # The Falses are the overlaps (cluster numbers are different), so return 1 minus the number of Trues
    return 1 - (np.sum(indiv.labels == indiv.labels[masked_dists])/len(indiv.labels))

def eigenval_ratio(indiv):
    '''Calculate the eigenvalue ratio (or amount of eccentricity). This is ratio between the largest and smallest eigenvalues of the diagonal covariance matrix.
    
    Arguments:
        indiv {Genotype} -- An individual of the population
    
    Returns:
        float -- The ratio of the largest to smallest eigenvalues
    '''
    # Need to make diagonal to get rid of the 0s
    # Take the maximum of all the eigenvalue ratios
    # i.e. the cluster with the highest eigenvalue ratio
    return np.max([np.max(np.diag(clust.cov))/np.min(np.diag(clust.cov)) for clust in indiv])
    