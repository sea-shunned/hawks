# HAWKS User Guide
## Generator Object
When calling `hawks.create_generator()`, a Generator object is returned. Alongside the `get_best_dataset()` method seen in the example, other methods are available that could be of use.

To plot the best individual (or best individuals from a multi-set config), use `plot_best_indiv()`. You can use `get_stats()` to get a pandas DataFrame of information stored during the optimization, such as the fitness and penalities of each individual at every generation. This is useful to track the evolution, and when filtered to get the final population can give further information about the nature of the datasets.

## Config File
JSON files are primarily used to define the parameters for `hawks`. A config can define either a single set of parameters, or when a list of parameters are provided than all possible parameter combinations are tested.

An example of both a single-set and multi-set config can be found in the `examples/` folder. In both cases, any parameter that is missing will be taken from the `defaults.json`.

Below is a brief description of each of the parameters, split into their major sections:

#### HAWKS Params
* `"folder_name"`: The name of the directory with which to save everything. This is empty by default to avoid saving in an interactive session.
* `"mode"`: This is the mode for `hawks`. Currently, only `"single"` is available.
* `"n_objectives"`: This is the number of objectives to be optimized, which can only be 1 at the moment.
* `"num_runs"`: The number of runs with `hawks` with different seed numbers.
* `"seed_num"`: This is the seed number that will be used to generate the seed numbers for each individual run (according to `num_runs`). If one is not provided, it is generated randomly, and saved back into the original config for reproducibility. When `"num_runs"` > 1, the subsequent seeds are based on this initial one.
* `"save_best_data"`: Save the dataset from the best (most fit) individual across all runs for each config.
* `"save_stats"`: Save the output values (fitness, penalities etc.) for every individual .
* `"plot_best"`: Flag to call the `plot_best_indiv()` method to plot the best individual for each set of parameters.
* `"save_plot"`: For the above command, determines whether the plot should be saved or just displayed.

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
where the `"constraint_name"` must match the corresponding function name in `constraints.py`, the `"threshold"` is a value of this constraint, and `"limit"` takes either `"upper"` or `"lower"` to denote whether this threshold is an upper limit or lower limit. In the former case, values above the threshold will be penalized, and the inverse for a lower limit.

## Plotting
Some plotting functions have been made to visualize the generated datasets. The current two methods for the generator are `plot_best_indiv()`, and `plot_indivs()`. The former creates a separate plot for each best individual (only one for the single config case), the latter plots all given individuals onto the same plot. In both cases, for datasets over 2 dimensions, PCA is used to project them down to 2D.

By default, the most recently seen population is accessible through the `.population` attribute, which when given as the argument to `plot_indivs()` can be useful to glimpse how the datasets look. The `plot_indivs()` method can also be given the `.best_indiv` attribute, for when (in the multi-config case) you want to view them on one graph. Note that, if the number of rows and columns for how to arrange these plots are not supplied, then a roughly square number of rows and columns will be used.

A filename and folder location can be specified to save the resulting plot (if these are given then saving is assumed), overriding previous config settings. If `save` is set to `True` but no location is given either to the function or from the config, the current working directory is used.

For analysis on the results, the `.stats` attribute contains a Pandas dataframe which provides easy access to Pandas' plotting capabilities. Some methods to facilitate this directly may be added in future versions.