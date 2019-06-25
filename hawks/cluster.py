"""
Defines the Cluster class, which represents a single cluster. Contains properties of the cluster (size, mean, covariance, data point values etc.). 

Responsible for the methods defining the mutation of a cluster.
"""
from itertools import count

import numpy as np
from scipy.stats import special_ortho_group, dirichlet
from scipy.linalg import fractional_matrix_power

class Cluster:
    # Get unique id value for each cluster
    id_value = count()
    # Global RandomState reference
    global_rng = None
    def __init__(self, size):
        # The cluster size (num data points)
        self.size = size
        # Mean of the cluster (Gaussian)
        self.mean = None
        # Covariance (diagonal matrix!)
        self.cov = None
        # Rotation matrix (rotation & scaling)
        self.rotation = None
        # The actual data points in the cluster
        self.values = None
        # The id of the cluster
        self.id = next(Cluster.id_value)
        # Maintain the seed number to reset the RandomState
        self.num_seed = None
        # Where to store the RandomState instance which we use to sample from
        self.clust_randstate = None
        # Flag we can use to determine whether recomputation of any distances involving this cluster is needed
        self.changed = True
        # Setup the cluster for use
        self.initial_cluster_setup()

    def initial_cluster_setup(self):
        # Set the seed for the cluster
        self.set_seed()
        # Set the random state (using the seed)
        self.set_state()
        # Generate the mean/centroid of the cluster
        self.gen_mean()
        # Generate the initial covariance for the cluster
        self.gen_initial_cov()
        # Generate an initial rotation matrix for the covariance
        self.rotation = self._gen_rotation()
        # Sample our first set of values (rotation is performed here)
        # self.sample_values()

    def set_seed(self):
        """Generate a random number for the seed.
        """
        # This needs to use the Dataset random_state so that we ensure consistency
        self.num_seed = self.global_rng.randint(
            low=0, high=10000000)

    def set_state(self):
        """Set the random state using the pre-defined seed number for this cluster. Organised like this so we can reset the state to sample, using our static seed number.
        """
        self.clust_randstate = np.random.RandomState(self.num_seed)
    
    def gen_mean(self):
        """Generate the mean vector for the cluster. Uses the class variables taken from the relevant Dataset instance to generate the mean.
        """
        # Try to generate the mean (using a uniform distribution)
        try:
            self.mean = self.global_rng.uniform(
                0, 1, self.num_dims)
        except AttributeError as e:
            raise Exception(f"Num_dims is not set as an attr for Cluster - this should have come from Dataset") from e

    def gen_initial_cov(self, method="eigen"):
        # Generate initial covariance matrix
        if method == "eigen":
            self.cov = np.diag(
                self.global_rng.uniform(
                    0, (1/2), self.num_dims
                )
            )

    def _gen_rotation(self):
        # Generate a random rotation matrix
        return special_ortho_group.rvs(
            dim=self.num_dims, random_state=self.global_rng)
    
    def _gen_scaling(self):
        """Generates a matrix to scale the covariance of a cluster. Ensures that the resulting determinant is unchanged.
        """
        return np.exp(
            dirichlet(alpha=3*np.ones(self.num_dims)).rvs(random_state=self.global_rng) - (1/self.num_dims)
        )[0]

    def rotate_cov(self):
        # Rotate the covariance matrix
        return self.rotation.dot(self.cov).dot(self.rotation.T)
    
    def _scale_cov(self, S):
        # Scale the covariance matrix by the scaling matrix
        return np.diag(self.cov.dot(S))
    
    @staticmethod
    def _reduce_rotation(R, power):
        # Reduce the rotation by a fractional power to reduce perturbation
        return fractional_matrix_power(R, power)

    def sample_values(self):
        # Reinitialise RandomState to sample consistent points
        self.set_state()
        # Obtain the current (rotated) covariance
        cov = self.rotate_cov()
        # Use [:] to ensure view remains
        self.values[:] = self.clust_randstate.multivariate_normal(
            mean=self.mean,
            cov=cov,
            size=self.size,
            check_valid='ignore') # ignores check for PSD

    def mutate_mean_random(self, scale, dims="each"):
        if dims == "each":
            # Probability test each dimension to mutate the mean
            return [
                self.global_rng.normal(loc=self.mean[i], scale=scale)
                if self.global_rng.rand() < (1/self.num_dims)
                else self.mean[i]
                for i in range(self.num_dims)
            ]
        elif dims == "all":
            # Mutate the mean in all dimensions
            return self.global_rng.normal(loc=self.mean, scale=scale)
        else:
            raise ValueError(f"{dims} is not a recognised option")

    def mutate_cov_haar(self, power):
        # Generate the scaling matirx
        S = self._gen_scaling()
        #print("cov before scaling", self.cov)
        self.cov = self._scale_cov(S)
        #print("cov after scaling", self.cov, "\n")
        # Generate a new rotation matrix
        R_new = self._gen_rotation()
        # Reduce this rotation matrix (so it's more of a perturbation)
        # and convert to float (to remove close-to-zero imaginary component)
        R_new = np.real(Cluster._reduce_rotation(R_new, power))
        # Modify existing rotation matrix
        self.rotation = R_new.dot(self.rotation)
