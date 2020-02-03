"""
Defines the Genotype class, representing a single individual. Given a light wrapper by DEAP, but retains all functionality. Essentially a list of :class:`hawks.cluster.Cluster` objects.

Handles the overall array of the data, as the individual clusters have views into the :attr:`hawks.genotype.Genotype.all_values`. Calculates the constraints for an individual if requried. Also handles the mutation (calling the defined method of the :class:`hawks.cluster.Cluster` class) and crossover.
"""
import inspect
import numpy as np

import hawks.constraints

class Genotype(list):
    """Wrapper class for the genotype/individual. A list of :class:`~hawks.cluster.Cluster` objects, handling operations on the individual-level (such as crossover). Contains the actual data array, which the :class:`~hawks.cluster.Cluster` s view into.

    Attributes:
        all_values (:py:class:`numpy.ndarray`):
        constraints (dict): Actual dictionary of the constraints and their violation.
        constraints_dict (dict): Container dictionary for the constraint functions we calculate. Scraped from :mod:`~hawks.constraints`.
        cxpb (float): Crossover probability.
        feasible (bool): Whether the individual is feasible (i.e. no penalty).
        global_rng (:py:class:`~numpy.random.mtrand.RandomState`)
        labels (list): A list (or array) of the true labels.
        mutpb_mean (float): Mutation probability for the mean.
        mutpb_cov (float): Mutation probability for the covariance.
        penalty (float): Constraint penalty violation.
        positions (list): A list of tuples that provide the indices for the start and end of each cluster (allows for partial modification of arrays).
    """
    # Get the available constraints
    constraints_dict = dict(
        inspect.getmembers(hawks.constraints, inspect.isfunction)
    )
    # Global RandomState reference
    global_rng = None
    # Mutation probabilities
    mutpb_mean = None
    mutpb_cov = None
    # Crossover probability
    cxpb = None
    def __init__(self, clusters):
        # Attributes for the data
        self.all_values = None
        self.labels = None
        self.positions = []
        # Attributes for feasibility/constraints
        self.feasible = True
        self.constraints = {}
        self.penalty = 0
        # Inherit from list
        super(Genotype, self).__init__(clusters)

    def create_views(self):
        """Create the views for each of the :attr:`~hawks.cluster.Cluster.values` attribute to the main array (:attr:`~hawks.genotype.Genotype.all_values`) for this individual.
        """
        num_rows = np.sum([clust.size for clust in self])
        num_dims = self[0].num_dims
        # Create the initial empty array
        self.all_values = np.zeros([
            num_rows, num_dims])
        # Create an array (vector) to store the labels for easy selection of clusters (and scikit-learn funcs)
        self.labels = np.zeros(
            [num_rows,],
            dtype=np.uint8
        )
        # Create a value to mark the start index for the array view
        start = 0
        # Loop over the sizes for each of the clusters
        # They are generated in order so can use the index/counter
        for counter, cluster in enumerate(self):
            # Set the values of the relevant cluster to be the specific view
            self[counter].values = self.all_values[start:start+cluster.size]
            # Set the labels
            self.labels[start:start+cluster.size] = counter
            # Maintain a list of the relevant start:stop indices for each cluster
            self.positions.append((start,start+cluster.size))
            # Get the right start position for the next cluster
            start += cluster.size

    def recreate_views(self):
        """Recreate the numpy array views to ensure values update when the individual :class:`~hawks.cluster.Cluster` s are changed.
        """
        for i, cluster in enumerate(self):
            start, stop = self.positions[i]
            cluster.values = self.all_values[start:stop]

    def recreate_single_view(self, index):
        """Recreate a single cluster's view (rather than all).

        Args:
            index (int): The index of the cluster
        """
        # Get the relevant indices for this cluster
        start, stop = self.positions[index]
        # Recreate the view
        self[index].values = self.all_values[start:stop]

    def resample_values(self):
        """Loop over the genotype and resample the values for any cluster that has been changed.
        """
        # Loop over the clusters
        for cluster in self:
            if cluster.changed:
                # Sample new values if the cluster has changed
                cluster.sample_values()

    @classmethod
    def validate_constraints(cls, constraint_params):
        """Ensure that the constraints in the config match what is available.

        Args:
            constraint_params (dict): Dictionary for the constraint parameters (from the config).
        """
        for name in constraint_params:
            try:
                cls.constraints_dict[name]
            except KeyError:
                print(f"{name} is not implemented (in constraints.py)")

    def calc_constraints(self, constraint_params):
        """Calculate the constraints for a given individual.

        Args:
            constraint_params (dict): Dictionary for the constraint parameters (from the config).
        """
        # Initialize penalty
        pen = 0
        # Loop over the provided constraints
        for name, func in self.constraints_dict.items():
            self.constraints[name] = func(self)
            # Check if the constraint is an upper bound
            if constraint_params[name]["limit"] == "upper":
                # Check if constraint is violated
                if self.constraints[name] > constraint_params[name]["threshold"]:
                    # Add the penalty
                    pen += ((self.constraints[name] - constraint_params[name]["threshold"])**2)
                    self.feasible = False
            # Check if it's a lower bound
            elif constraint_params[name]["limit"] == "lower":
                # Check if constraint is violated
                if self.constraints[name] < constraint_params[name]["threshold"]:
                    # Add the penalty
                    pen += ((constraint_params[name]["threshold"] - self.constraints[name])**2)
                    self.feasible = False
            # Otherwise fail
            else:
                print(f"{constraint_params[name]['limit']} is not recognised")
                raise ValueError
        # Assign the penalty
        self.penalty = pen

    def recalc_constraints(self, constraint_params):
        """Recalculate constraints only if a cluster has changed.

        Args:
            constraint_params (dict): Dictionary for the constraint parameters (from the config).
        """
        for cluster in self:
            if cluster.changed:
                break
        else:
            return
        self.calc_constraints(constraint_params)

    @staticmethod
    def reconst_values(parent1, parent2, index):
        """Reconstructs the values of the individual clusters, useful for when the view has become disentangled by moving :class:`~hawks.cluster.Cluster` objects between :class:`~hawks.genotype.Genotype` instances.

        Args:
            parent1 (:class:`~hawks.genotype.Genotype`): The first parent.
            parent2 (:class:`~hawks.genotype.Genotype`): The second parent.
            index (int): The cluster that needs reconstruction
        """
        start, stop = parent1.positions[index]
        parent1.all_values[start:stop] = parent1[index].values.copy()
        parent2.all_values[start:stop] = parent2[index].values.copy()

    @staticmethod
    def xover_cluster(parent1, parent2, mixing_ratio=0.5):
        """Uniform crossover with one probability test for the mean and covariance together.

        Args:
            parent1 (:class:`~hawks.genotype.Genotype`): The first parent.
            parent2 (:class:`~hawks.genotype.Genotype`): The second parent.
            mixing_ratio (float, optional): The probability of mixing the parents. Defaults to 0.5.

    Returns:
        tuple: 2-element tuple containing the two mixed parents of type :class:`~hawks.genotype.Genotype`.
        """
        # Avoid repeated len calls
        size = len(parent1)
        # Loop over each gene
        for i in range(size):
            # Flag for view change
            change = False
            if Genotype.global_rng.rand() < mixing_ratio:
                parent1[i], parent2[i] = parent2[i], parent1[i]
                change = True
            # Link the cluster to its new genotype
            if change:
                # Reconstruct the all_values arrays with the new data
                Genotype.reconst_values(parent1, parent2, i)
                # Reconstruct the views
                parent1.recreate_single_view(i)
                parent2.recreate_single_view(i)
                # Note that the clusters have changed
                parent1[i].changed, parent2[i].changed = True, True
        return parent1, parent2

    @staticmethod
    def xover_genes(parent1, parent2, mixing_ratio=0.5):
        """Uniform crossover with separate probability tests for the mean and covariance.

        Args:
            parent1 (:class:`~hawks.genotype.Genotype`): The first parent.
            parent2 (:class:`~hawks.genotype.Genotype`): The second parent.
            mixing_ratio (float, optional): The probability of mixing the parents. Defaults to 0.5.

    Returns:
        tuple: 2-element tuple containing the two mixed parents of type :class:`~hawks.genotype.Genotype`.
        """
        # Avoid repeated len calls
        size = len(parent1)
        # Loop over each gene
        for i in range(size):
            change = False
            # Test for swapping the mean
            if Genotype.global_rng.rand() < mixing_ratio:
                parent1[i].mean, parent2[i].mean = parent2[i].mean, parent1[i].mean
                change = True
            # Test for swapping the covariance
            if Genotype.global_rng.rand() < mixing_ratio:
                parent1[i].cov, parent2[i].cov = parent2[i].cov, parent1[i].cov
                # Also swap the rotation matrix (as it's part of the cov)
                parent1[i].rotation, parent2[i].rotation = parent2[i].rotation, parent1[i].rotation
                # Also swap the seed, as this affects the covariance most
                parent1[i].num_seed, parent2[i].num_seed = parent2[i].num_seed, parent1[i].num_seed
                change = True
            # Link the cluster to its new genotype
            if change:
                # Recreate the views
                parent1.recreate_single_view(i)
                parent2.recreate_single_view(i)
                # Note that the clusters have changed
                parent1[i].changed, parent2[i].changed = True, True
        return parent1, parent2

    def save_clusters(self, folder, fname):
        """Save the dataset (values and corresponding labels) defined by this individual, given a folder and filename. Saves as a CSV via :func:`numpy.savetxt`.

        Args:
            folder (:obj:`str`, :class:`pathlib.Path`): The path to the folder where the datasets should be saved.
            fname (str): The filename to use.
        """
        # Put data and labels together
        full_dataset = np.hstack((self.all_values, self.labels[:, np.newaxis]))
        # Save the dataset as comma separated
        # Change the format to ensure the labels are integers
        np.savetxt(
            f"{folder/fname}.csv",
            full_dataset,
            delimiter=",",
            fmt=["%.18e"]*self.all_values.shape[1] + ["%i"]
        )

    def mutation(self, mut_mean_func, mut_cov_func):
        """Generic mutation function.

        Arguments given must be functions for mutating the mean and covariance. Can use :func:`functools.partial` to freeze other arguments if need be - see :func:`~hawks.ga.select_mutation` for options.

        Args:
            mut_mean_func (:obj:`~typing.Callable`): The mutation operator function for the mean.
            mut_cov_func (:obj:`~typing.Callable`): The mutation operator function for the covariance.
        """
        for i, cluster in enumerate(self):
            # Mean mutation
            if Genotype.global_rng.rand() < Genotype.mutpb_mean:
                cluster.mean = mut_mean_func(cluster, genotype=self)
                cluster.changed = True
            # Covariance mutation
            if Genotype.global_rng.rand() < Genotype.mutpb_cov:
                mut_cov_func(cluster)
                cluster.changed = True
