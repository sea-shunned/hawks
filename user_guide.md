# HAWKS User Guide
Below is a more extensive explanation of some of the components of `hawks`. Please also see the examples provided for practical guidance on how to use this package.

## Generator Object
When calling `hawks.create_generator()`, a HAWKS generator object is returned, according to the config provided.

You can use `get_stats()` (or access via `generator.stats`) to get a Pandas DataFrame of data stored during the optimization, such as the fitness and penalities of each individual at every generation. This is useful to track the evolution, and when filtered to get the final population can give further information about the nature of the datasets.

To load in previous runs of `hawks`, `hawks.io.load_folder()` provides the ability to create a generator from a folder of a previous run.

## Config File
Parameters for `hawks` can be provided either as a dictionary, or from an external JSON file. A config can define either a single set of parameters, or when a list of parameters are provided then all possible parameter combinations are used (this is the `multi-config` setting).

An example of both a single-set and multi-set config can be found in the `examples/` folder. In both cases, any parameter that is missing will be taken from the `hawks/defaults.json` file.

Below is a brief description of each of the parameters, split into their major sections:

#### HAWKS Params
* `"folder_name"`: The name of the directory with which to save everything. This is empty by default to avoid saving in an interactive session.
* `"mode"`: This is the mode for `hawks`. Currently, only `"single"` is available.
* `"n_objectives"`: This is the number of objectives to be optimized, which can only be 1 at the moment.
* `"num_runs"`: The number of runs with `hawks` with different seed numbers.
* `"seed_num"`: This is the seed number that will be used to generate the seed numbers for each individual run (according to `num_runs`). If one is not provided, it is generated randomly, and saved back into the original config for reproducibility. When `"num_runs"` > 1, the subsequent seeds are based on this initial one.
* `"save_best_data"`: Save the dataset from the best (most fit) individual across for each run for each config.
* `"save_stats"`: Save the output values (fitness, penalities etc.) for every individual.
* `"plot_best"`: Flag to call the `plot_best_indivs()` method to plot the best individual for each run for each config.
* `"save_plot"`: For the above command, determines whether the plot should be saved (`True`) or just displayed (`False`).

#### Objective Params
For the `"objectives"` JSON object, it is expected to be in the form below:

```JSON
"objectives": {
    "objective_name": {
        "arg": value
    },
    "objective_name_2": {
        "arg": value,
        "arg2": value
    }
}
```
At present, only one objective is available, under the name `"silhouette"`, which takes only the target silhouette width (`"target"`) as an argument.

#### Dataset Params
* `"num_examples"`: The size of the dataset i.e. number of datapoints/examples. High values of this can slow down `hawks` very quickly, as the silhouette width does not scale well.
* `"num_clusters"`: The number of clusters to be generated.
* `"num_dims"`: The dimensionality of the datasets.
* `"equal_clusters"`: Boolean value of whether the clusters should be equally sized or not
* `"min_clust_size"`: The minimum number of datapoints that a cluster should have. This will guarantee that each cluster is at least of this size.

#### GA Params
* `"num_gens"`: The number of generations to evolve over.
* `"num_indivs"`: The number of individuals in a population.
* `"mut_method_mean"`: The method used to mutate the mean. At present, only `"random"` is available.
* `"mut_args_mean"`: The arguments for the above, in the format required by the function for this mutation.
* `"mut_method_cov"`: The method used to mutate the covariance. At present, only `"haar"` is available.
* `"mut_args_cov"`: The arguments for the above, in the format required by the function for this mutation.
* `"mut_prob_mean"`: The mutation probability to mutate the mean. Either a float between 0 & 1, or use `"length"` to calculate the probability based on the length of the genotype (recommended).
* `"mut_prob_cov"`: The mutation probability to mutate the covariance. Either a float between 0 & 1, or use `"length"` to calculate the probability based on the length of the genotype (recommended).
* `"mate_scheme"`: The method for crossover. At the moment, accepts either `"dv"` (which can swap the mean and covariance separately between individuals), or `"cluster"` (which swaps whole clusters between individuals).
* `"mate_prob"`: The probability of crossover.
* `"prob_fitness"`: The probabilitity that comparison will be performed based on fitness in stochastic ranking.
* `"elites"`: The percentage of elites (the most fit individuals in the population) that will be preserved between generations.

#### Constraint Params
For the `"constraints"`, it is expected to be in the form below:
```JSON
"constraints": {
    "constraint_name": {
        "threshold": value,
        "limit": value
    }
}
```
where the `"constraint_name"` must match the corresponding function name in `hawks/constraints.py`, the `"threshold"` is a value of this constraint, and `"limit"` takes either `"upper"` or `"lower"` to denote whether this threshold is an upper limit or lower limit. In the former case, values above the threshold will be penalized, and the inverse for a lower limit.

## Plotting
There are some functions for plotting available in `hawks/plotting.py`, or the `generator.plot_best_indivs()` method for a common use-case. These allow for individuals to be plotting together or separately. For datasets whose clusters are greater than 2 dimensions, PCA is used to visualize the datasets. Of course, other methods can be used beforehand, and then the new transformed 2D datasets can be given to the plotting functions.

By default, the most recently seen population is accessible through the `.population` attribute, which can be useful to glimpse how the datasets look. Note that, if the number of rows and columns for how to arrange these plots are not supplied, then a roughly square number of rows and columns will be used.

A filename and folder location can be specified to save the resulting plot (if these are given then saving is assumed), overriding previous config settings. If `save` is set to `True` but no location is given either to the function or from the config, the current working directory is used.

For analysis on the results, the `.stats` attribute contains a Pandas dataframe which provides easy access to Pandas' plotting capabilities. Some methods to facilitate this directly may be added in future versions.

## Instance Space
An important part of this work is the visualization of the datasets according to their problem features in an "instance space". This is the main function of `hawks/analysis.py`. For an example of using this, see ??, and for further background see the associated paper.

As seen in `examples/instance_example.py`, this is designed to work with both `hawks` and datasets from external sources. Just modify the `source` parameter, and pass whatever arguments are needed for `np.loadtxt` to load the dataset, or give a custom function that extracts the data and associated labels.

In `hawks/plotting.py`, the `instance_space()` function allows for the easy plotting of the instance space generated from `analyse_datasets()`. The instance space from the example is shown [here](https://github.com/sea-shunned/hawks/blob/master/examples/instance_space_source.pdf).

As a single set of values for the silhouette width and constraints were used for HAWKS, the diversity is a lot less than that produced from the `sklearn` functions, hence the difference.

### Cluster Analysis
By running `analyse_datasets(clustering=True)` we can run a series of clustering algorithms (defined in `hawks/analysis.py`) on the provided datasets.

In `examples/clustering_example.py` we can see multiple sets of datasets being run with the different clustering algorithms. We can then plot the clustering performance (Adjusted Rand Index) for the different sets of datasets, shown [here](https://github.com/sea-shunned/hawks/blob/master/examples/clustering_performance.pdf).