"""Defines the :class:`~hawks.dataset.Dataset` class, which handles general properties of the dataset that is being evolved (and is consistent across individuals in the population).
"""
import warnings

import numpy as np

class Dataset:
    """Class for properties about the dataset (such as the size, number of dimensions) as a whole go here.

    Separation may be useful for potential future development.

    Attributes:
        num_examples (int): Total size of the dataset.
        num_clusters (int): Number of clusters.
        num_dims (int): Number of dimensions.
        equal_clusters (bool): If the clusters should be equally sized.
        min_clust_size (int): Minimum size (number of data points) a cluster should have.
        global_rng (:py:class:`~numpy.random.mtrand.RandomState`): The global :py:class:`~numpy.random.mtrand.RandomState` instance used as a common RNG.
    """
    # Global RandomState reference
    global_rng = None
    def __init__(self, num_examples, num_clusters, num_dims, equal_clusters, min_clust_size):
        # Error checking for minimum cluster size
        if min_clust_size is None:
            self.min_clust_size = 2
        elif min_clust_size > num_examples:
            warnings.warn(
                message=f"The minimum cluster size ({min_clust_size}) is larger than the total specified size of the dataset ({num_examples}) - setting to 2",
                category=UserWarning
            )
            self.min_clust_size = 2
        elif min_clust_size > num_examples/num_clusters:
            warnings.warn(
                message=f"The minimum cluster size ({min_clust_size}) is larger than the expected size of each clusters ({num_examples/num_clusters}) - setting to 2",
                category=UserWarning
            )
            self.min_clust_size = 2
        elif isinstance(min_clust_size, float):
            warnings.warn(
                message=f"The minimum cluster size ({min_clust_size}) should not be a float - setting to 2",
                category=UserWarning
            )
            self.min_clust_size = 2
        else:
            self.min_clust_size = min_clust_size
        # Set attributes
        self.num_examples = num_examples
        self.num_clusters = num_clusters
        self.num_dims = num_dims
        self.equal_clusters = equal_clusters
        # Initialise cluster sizes
        self.cluster_sizes = None
        self.gen_cluster_sizes()
        # Attributes that the Cluster class needs
        # Need to ascertain which of these we actually need
        self.cluster_vars = {
            'num_dims':         self.num_dims,
            'num_clusters':     self.num_clusters,
            'cluster_sizes':    self.cluster_sizes
        }

    def gen_cluster_sizes(self, method="auto"):
        # Select method based on inputs provided
        if method == "auto":
            if self.equal_clusters:
                method = "equal"
            else:
                method = "random"
        # Generate the clusters based on method
        if method == "random":
            self._random_clust_sizes()
        elif method == "equal":
            self._equal_clust_sizes()
        else:
            raise ValueError(
                f"Method '{method}' for generating cluster sizes is not implemented")
        # **TODO** consider adding a size tuple method for some control over relative sizes

    def _random_clust_sizes(self):
        # https://stackoverflow.com/questions/29187044/generate-n-random-numbers-within-a-range-with-a-constant-sum
        weights = [-np.log(self.global_rng.rand()) for _ in range(self.num_clusters)]
        sum_val = np.sum(weights)
        # Scale the weights to sum to 1
        weights = [i/sum_val for i in weights]
        # Create cluster sizes (possibly a bit long...)
        self.cluster_sizes = [
            int(np.around(self.min_clust_size + i*(self.num_examples-(self.num_clusters*self.min_clust_size)))) for i in weights
        ]

    def _equal_clust_sizes(self):
        # Generate equal sizes that roughly sum to target
        clust_size = int(np.around(self.num_examples/self.num_clusters))
        # Create the list of cluster sizes
        self.cluster_sizes = [clust_size] * self.num_clusters
