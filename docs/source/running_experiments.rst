.. _experiments_page:

Running Experiments
===================

This section will provide information about running experiments and practical usage of HAWKS, providing further context and advice on top of the :ref:`examples_page`.

As future versions of HAWKS are released, the analysis functionality will increase with it, permitting a wider range of graphs to be easily created, and other potential features (of which suggestions are welcome, as discussed on the :ref:`contributing_page` page).


``scikit-learn`` integration
----------------------------

As shown in the :ref:`quick_example_section`, we can easily pass the generated datasets to other libraries, such as ``scikit-learn``.

Generally, we can use the :meth:`~hawks.generator.BaseGenerator.get_best_dataset()` method to extract the data and labels for the best dataset found from each run. It is generally not recommended to use all of the datasets from each run, as this can (over time in larger experiments) impose a memory burden, and a larger diversity of datasets comes from repeated runs.


Large number of runs/data points
--------------------------------

As discussed in :ref:`example_multiconfig`, HAWKS uses ``tqdm`` for progress bars. This can provide an estimate of running time during the run so early diagnosis of the feasibility of a run can be seen. Different parameter sets can take drastically different times however, so this may not be accurate in the early runs.

The main driver increasing computation time with HAWKS (beyond many configs or many runs), is the number of data points to generator, then followed by the number of dimensions. This is unavoidable due to the use of the silhouette width, despite computational tricks used to reduce the complexity.


Full pipeline
-------------

The example below is the full experiment script used in our `paper <https://doi.org/10.1145/3321707.3321761>`_, that covers setting up HAWKS, loading external datasets (via :func:`~hawks.io.load_datasets`), and :mod:`~hawks.plotting`.

The majority of the code is to create lots of different types of plots. Running HAWKS, loading and processing external datasets, and running clustering algorithms on all of them is quite simple.

Comments are littered throughout the example to provide context, but if anything is unclear and stops you from running an experiment yourself, please let me know.

The external datasets used are from `the "HK" paper <https://ieeexplore.ieee.org/abstract/document/1554990>`_ and `the "QJ" paper <https://link.springer.com/article/10.1007/s00357-006-0018-y>`_ .

.. literalinclude:: examples/full_pipeline.py
    :language: python