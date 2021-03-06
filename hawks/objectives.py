"""Defines the Objective class, and its subclasses. Anything that will be used in the fitness function should be implemented here as a relevant class.

Class hierarchy is set up for expansions to more objectives that can be selected from.
"""
import abc
from itertools import permutations

import numpy as np
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform, cdist

class Objective(abc.ABC):
    """Overall wrapper class for the objectives, defining the mandatory methods.

    Attributes:
        weight (float): The objective weight, where -1 is minimization and 1 is maximization.
    """
    # Objective weight
    weight = None

    def __init__(self):
        pass

    # Subclass implementations will still need to declare @staticmethod
    @staticmethod
    @abc.abstractmethod
    def eval_objective(indiv):
        """Evaluates the objective on an individual/solution
        """

    @classmethod
    def set_kwargs(cls, kwargs_dict):
        """Used to set the arguments for the objective from the config
        """
        for name, value in kwargs_dict.items():
            setattr(cls, name, value)

    @classmethod
    def set_objective_attr(cls, indiv, value):
        """Use setattr with the name of the objective for results saving
        """
        setattr(indiv, cls.__name__.lower(), value)

class ClusterIndex(Objective):
    """For handling shared computation of more cluster indices if that is expanded. There is method to this madness.
    """
    pass

class Silhouette(ClusterIndex):
    """Class to calculate the `silhouette width <https://www.sciencedirect.com/science/article/pii/0377042787901257>`_. See the source code for computation.

    Attributes:
        target (float): The target value of the silhouette width to optimize the datasets towards.
        method (str, optional): The method to use for calculating the silhouette width. Either ``"own"`` or ``"sklearn"``. Defaults to "own", which is recommended.
    """
    # Minimization
    weight = -1.0
    # Use our method, not sklearn (avoids unnecessary compute)
    method = "own"
    # Target silhouette width
    target = None

    def __init__(self):
        super(Silhouette, self).__init__(self)

    @staticmethod
    def setup_indiv(indiv):
        indiv.distances = Silhouette.calc_distances(indiv.all_values)
        indiv.a_vals = np.zeros((indiv.distances.shape[0], ))

    @staticmethod
    def calc_distances(data, metric="sqeuclidean"):
        return squareform(
            pdist(data, metric=metric)
        )

    @staticmethod
    def calc_intraclusts(indiv, clust_list):
        # Calculate the intracluster variance
        for i in clust_list:
            start, stop = indiv.positions[i]
            # Calculate the numerator values
            clust_array = indiv.distances[start:stop, start:stop]
            indiv.a_vals[start:stop] = np.true_divide(
                np.sum(clust_array, axis=0),
                np.sum(clust_array != 0, axis=0)
            )

    @staticmethod
    def calc_interclusts(indiv):
        # Calculate the intercluster variance
        # Need to recalc everything for the b(i) term
        combs = permutations(range(len(indiv.positions)), 2)
        indiv.b_vals = np.full((indiv.distances.shape[0], ), np.inf)
        for clust_num1, clust_num2 in combs:
            c1_start, c1_stop = indiv.positions[clust_num1]
            c2_start, c2_stop = indiv.positions[clust_num2]
            clust_array = indiv.distances[c1_start:c1_stop, c2_start:c2_stop]
            # Get the minimum average distance
            indiv.b_vals[c1_start:c1_stop] = np.minimum(
                np.mean(clust_array, axis=1), indiv.b_vals[c1_start:c1_stop])

    @staticmethod
    def recompute_dists(indiv, clust_list, metric="sqeuclidean"):
        # Recompute the distances only for the clusters that have moved
        for i in clust_list:
            start, stop = indiv.positions[i]
            new_dists = cdist(
                indiv.all_values,
                indiv.all_values[start:stop, :],
                metric=metric
            )
            # Update the matrix
            indiv.distances[:, start:stop] = new_dists
            indiv.distances[start:stop, :] = new_dists.T

    @staticmethod
    def calc_silh(indiv):
        # Calculate the silhouette width 
        top_term = indiv.b_vals - indiv.a_vals
        bottom_term = np.maximum(indiv.b_vals, indiv.a_vals)
        res = top_term / bottom_term
        # Handle singleton clusters (equiv to sklearn)
        res = np.nan_to_num(res)
        return np.mean(res)

    @staticmethod
    def eval_objective(indiv):
        # Required function to evaluate the silhouette width objective
        # Option to use the sklearn version
        if Silhouette.method == "sklearn":
            return silhouette_score(
                indiv.all_values,
                indiv.labels,
                metric="sqeuclidean")
        else:
            # *TODO*: Add hooks for metric here if we want it to be dynamic
            # Check if any recomputation is needed
            for cluster in indiv:
                if cluster.changed:
                    break
            # Nothing has changed
            else:
                # Handle the first time it's evaluated
                if indiv.silhouette is None:
                    # Calculate the real silhouette width
                    silh_width = Silhouette.calc_silh(indiv)
                    # Set the silhouette width as an attribute
                    Silhouette.set_objective_attr(indiv, silh_width)
                else:
                    silh_width = indiv.silhouette
                # Return the difference between the objective target and the true SW
                return np.abs(Silhouette.target - silh_width)
            # Container to only update the clusters that have changes
            clust_list = []
            for i, cluster in enumerate(indiv):
                if cluster.changed:
                    clust_list.append(i)
            # Update the distance matrix
            if len(clust_list) < len(indiv):
                Silhouette.recompute_dists(indiv, clust_list)
            # Do the full array calc if all clusters have changed
            else:
                indiv.distances = Silhouette.calc_distances(indiv.all_values)
            # Calculate the new a(i) values
            Silhouette.calc_intraclusts(indiv, clust_list)
            # Calculate the new b(i) values
            Silhouette.calc_interclusts(indiv)
            # Calculate the silhouette width
            silh_width = Silhouette.calc_silh(indiv)
            # Set the silhouette width as an attribute
            Silhouette.set_objective_attr(indiv, silh_width)
            # Return the distance to the target
            return np.abs(Silhouette.target - silh_width)
