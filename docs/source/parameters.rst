.. _parameters_page:

Parameters
==========

This section will introduce the various parameters of HAWKS, giving a brief description of what they are and what they do. For some, the technical background is beyond the scope of this documentation and is omitted (for details, please refer to the associated `paper <https://doi.org/10.1145/3321707.3321761>`_).

The parameters can be passed into HAWKS via :func:`~hawks.generator.create_generator` as either a ``dict`` or (the path to) a JSON file. 

.. note::

    Certain future functionality may require the use of a ``dict``, but this will be for limited use cases.

Explanation
-----------

The full set of parameters can be seen in the :ref:`defaults_section`. In the following, each of the config sub-dicts (``"hawks"``, ``"dataset"``, ``"constraints"``, ``"ga"``, and ``"objectives"``) will be explained separately.

The Python type of parameter is also given. For the relevant JSON type, see `this conversion table <https://docs.python.org/3.6/library/json.html#encoders-and-decoders>`_.

HAWKS Parameters
^^^^^^^^^^^^^^^^
Parameters for the ``"hawks"`` sub-dict. Primarily used for :class:`~hawks.generator.BaseGenerator`.

.. tabularcolumns:: JJp{0.5}

.. list-table::
    :widths: 15 10 75
    :header-rows: 1

    *   - Name
        - Type
        - Description
    *   - ``"folder_name"``
        - :obj:`str`
        - The name of the directory with which to save everything. This is empty by default to avoid saving in an interactive session.
    *   - ``"mode"``
        - :obj:`str`
        - This is the mode for HAWKS. Currently, only `"single"` is available.
    *   - ``"n_objectives"``
        - :obj:`int`
        - This is the number of objectives to be optimized, which can only be 1 at the moment.
    *   - ``"num_runs"``
        - :obj:`int`
        - The number of runs of HAWKS with different seed numbers.
    *   - ``"seed_num"``
        - :obj:`int`
        - This is the seed number that will be used to generate the seed numbers for each individual run (according to `num_runs`). If one is not provided, it is generated randomly, and saved back into the original config for reproducibility. When `"num_runs"` > 1, the subsequent seeds are based on this initial one.
    *   - ``"comparison"``
        - :obj:`str`
        - Method to extract the best individual from the final population. Using "fitness" selects the individual with the best fitness, whereas "ranking" selects the first individual from the population sorted by stochastic ranking.
    *   - ``"save_best_data"``
        - :obj:`bool`
        - Save the dataset from the best (most fit) individual across for each run for each config.
    *   - ``"save_stats"``
        - :obj:`bool`
        - Save the output values (fitness, penalities etc.) for every individual.
    *   - ``"save_config"``
        - :obj:`bool`
        - Flag to save the (full) config associated with the run.


Objective Parameters
^^^^^^^^^^^^^^^^^^^^
Parameters for the ``"objectives"`` sub-dict. Primarily used for :class:`~hawks.objectives.Objective`. For the ``"objectives"`` JSON object, it is expected to be in the form below:

.. code-block:: json

    {
        "objectives": {
            "objective_name": {
                "arg": "value"
            },
            "objective_name_2": {
                "arg": "value",
                "arg2": "value"
            }
        }
    }

where:

.. list-table::
    :widths: 15 10 75
    :header-rows: 1

    *   - Name
        - Type
        - Description
    *   - ``"objective_name"``
        - :obj:`str`
        - The name of the objective (it must, when converted to lowercase, match the name of the desired objective in :mod:`hawks.objectives`.
    *   - ``"arg"``
        - :obj:`str`
        - The name of an argument to the objective. The value of this will vary by objective, and the value depends on this argument. See either :ref:`defaults_section` for an example, or in the relevant :mod:`hawks.objectives` documentation.


Dataset Parameters
^^^^^^^^^^^^^^^^^^
Parameters for the ``"dataset"`` sub-dict. Primarily used for :class:`~hawks.dataset.Dataset`.

.. list-table::
    :widths: 15 10 75
    :header-rows: 1

    *   - Name
        - Type
        - Description
    *   - ``"num_examples"``
        - :obj:`int`
        - The size of the dataset i.e. number of datapoints/examples. Warning: High values of this can slow down HAWKS, as the silhouette width does not scale well (despite some computational tricks used here).
    *   - ``"num_clusters"``
        - :obj:`int`
        - The number of clusters to be generated.
    *   - ``"num_dims"``
        - :obj:`int`
        - The dimensionality of the datasets.
    *   - ``"equal_clusters"``
        - :obj:`bool`
        - Whether the clusters should be equally sized or not. if not, they are randomly sized such that they sum to the ``"num_examples"`` (though this might be ± a few data points).
    *   - ``"min_clust_size"``
        - :obj:`int`
        - The minimum number of datapoints that a cluster should have. This will guarantee that each cluster is at least of this size.


GA Parameters
^^^^^^^^^^^^^
Parameters for the ``"ga"`` sub-dict. Primarily used for :mod:`~hawks.ga`.

.. list-table::
    :widths: 15 10 75
    :header-rows: 1

    *   - Name
        - Type
        - Description
    *   - ``"num_gens"``
        - :obj:`int`
        - The number of generations to evolve over.
    *   - ``"num_indivs"``
        - :obj:`int`
        - The number of individuals in the population.
    *   - ``"mut_method_mean"``
        - :obj:`str`
        - The method used to mutate the mean. At present, only ``"random"`` is available.
    *   - ``"mut_args_mean"``
        - :obj:`str`
        - The arguments for the above, in the format required by the function for this mutation (with the name of the method as the key). See the :ref:`defaults_section` for all possible arguments.
    *   - ``"mut_method_cov"``
        - :obj:`str`
        - The method used to mutate the covariance. At present, only ``"haar"`` is available.
    *   - ``"mut_args_cov"``
        - :obj:`str`
        - The arguments for the above, in the format required by the function for this mutation (with the name of the method as the key). See the :ref:`defaults_section` for all possible arguments.
    *   - ``"mut_prob_mean"``
        - :obj:`str`
        - The mutation probability to mutate the mean. Either a :obj:`float` between 0 & 1, or ``"length"`` to calculate the probability based on the length of the genotype (recommended).
    *   - ``"mut_prob_cov"``
        - :obj:`str`
        - The mutation probability to mutate the covariance. Either a :obj:`float` between 0 & 1, or ``"length"`` to calculate the probability based on the length of the genotype (recommended).
    *   - ``"mate_scheme"``
        - :obj:`str`
        - The method for crossover. Accepts either ``"dv"`` (which can swap the mean and covariance separately between individuals), or ``"cluster"`` (which swaps whole clusters between individuals).
    *   - ``"mate_prob"``
        - :obj:`str`
        - The probability of crossover.
    *   - ``"prob_fitness"``
        - :obj:`str`
        - The probabilitity that comparison will be performed based on fitness in the stochastic ranking. Requires ``"environ_selection" = "sr"`` (though that is the only option at present).
    *   - ``"elites"``
        - :obj:`str`
        - The percentage of elites (the most fit individuals in the population) that will be preserved between generations. Not currently recommended, as it interferes with stochastic ranking.
    *   - ``"initial_mean_upper"``
        - :obj:`float`
        - The initial upper range for initializing the means.
    *   - ``"initial_cov_upper"``
        - :obj:`float`
        - - The initial upper range for initializing the covariances.
    *   - ``"environ_selection"``
        - :obj:`str`
        - The environmental selection operator. See :func:`~hawks.ga.select_parent_func` for details.
    *   - ``"parent_selection"``
        - :obj:`str`
        - The parental selection operator. At present, only stochastic ranking (``"sr"``) is available. See :func:`~hawks.ga.select_environ_func` for details.


Constraint Parameters
^^^^^^^^^^^^^^^^^^^^^
Parameters for the ``"constraints"`` sub-dict. It is expected to be in the form below, with any number of ``"constraint_name"`` sub-dicts.

.. code-block:: json

    {
        "constraints": {
            "constraint_name": {
                "threshold": "value",
                "limit": "value"
            }
        }
    }

where:

.. list-table::
    :widths: 15 10 75
    :header-rows: 1

    *   - Name
        - Type
        - Description
    *   - ``"constraint_name"``
        - :obj:`str`
        - The name of the constraint, which must match the function for it specified in :mod:`hawks.constraints`
    *   - ``"threshold"``
        - :obj:`float`
        - The value which is used to identify penalty violation. The type can vary. but generally is a :obj:`float`.
    *   - ``"limit"``
        - :obj:`str`
        - Whether the ``"threshold"`` is an ``"upper"`` or ``"lower"`` limit (only these two options are available). In the former case, values above the threshold will be penalized, and the inverse for a lower limit.

.. _defaults_section:

Defaults
--------
The defaults values below are pulled from the ``defaults.json``. For any variables that are not specified, these are used instead.

.. literalinclude:: ../../hawks/defaults.json
    :language: json
    :caption: Default parameters

Multi-config
------------

Multiple runs of HAWKS with varying parameters can be specified by a single config, by wrapping the parameters as a list e.g. ``"num_examples": [500, 1000, 1500]`` will run HAWKS three times with three different values for the number of examples. This works combinatorially, so a warning is raised when more than 1,000 runs are expected.

This makes experimenting easier, which is covered in :ref:`experiments_page`. An example of this is given :ref:`here <example_multiconfig>`.