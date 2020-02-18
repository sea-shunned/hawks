HAWKS Data Generator
====================

.. image:: docs/source/images/hawks_animation.gif
   :alt: Example gif of HAWKS
   :scale: 65 %
   :align: center

.. summary-marker-1-start

HAWKS is a tool for generating controllably difficult synthetic data,
used primarily for clustering.

.. summary-marker-1-end

This `repo <https://github.com/sea-shunned/hawks>`_ is associated with the following paper:

.. paper-marker-1-start

1. `Shand, C. <http://sea-shunned.github.io/>`_, `Allmendinger, R. <https://personalpages.manchester.ac.uk/staff/Richard.Allmendinger/>`_, `Handl, J. <https://personalpages.manchester.ac.uk/staff/Julia.Handl/>`_, `Webb, A. <http://www.awebb.info/>`_, & Keane, J. (2019, July). Evolving controllably difficult datasets for clustering. In Proceedings of the Genetic and Evolutionary Computation Conference (pp. 463-471). https://doi.org/10.1145/3321707.3321761 **(Nominated for best paper on the evolutionary machine learning track at GECCO'19)**

Related BibTeX info:

.. code-block:: bibtex

    @inproceedings{shand2019evolving,
        title={Evolving controllably difficult datasets for clustering},
        author={Shand, Cameron and Allmendinger, Richard and Handl, Julia and Webb, Andrew and Keane, John},
        booktitle={Proceedings of the Genetic and Evolutionary Computation Conference},
        pages={463--471},
        year={2019},
        url={https://doi.org/10.1145/3321707.3321761}
    }

The academic/technical details can be found there. What follows here is
a practical guide to using HAWKS to generate synthetic data.

.. paper-marker-1-end

If you use HAWKS to generate data that forms part of a paper, please
cite the paper above and link to this repo.


.. installation-marker-start

Installation
------------

Installation is available through pip by:

``pip install hawks``

.. installation-marker-end

or by cloning this repo (and installing locally using
``pip install .``). HAWKS was written for Python 3.6+. Other dependencies are specified in the `setup.py <https://github.com/sea-shunned/hawks/blob/master/setup.py>`_ file.


Running HAWKS
-------------

The parameters of hawks are configured via a config file system. Details
of the parameters are found in the `documentation <https://hawks.readthedocs.io/parameters>`_. For any parameters
that are not specified, default values will be used (as defined in
``hawks/defaults.json``).

.. example-marker-start

The example below illustrates how to run ``hawks``. Either a dictionary
or a path to a JSON config can be provided to override any of the
default values. Further examples can be found `here <https://hawks.readthedocs.io/examples>`_. 

.. Need to turn the bit below into an example file and then just include that

.. code-block:: python

    """Single, simple HAWKS run, with KMeans applied to the best dataset
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    import hawks

    # Set the magic seed number
    SEED_NUM = 42
    # Set the seed number in the config
    config = {
        "hawks": {
            "folder_name": "simple_example",
            "seed_num": SEED_NUM
        },
        "dataset": {
            "num_clusters": 5
        },
        "objectives": {
            "silhouette": {
                "target": 0.9
            }
        }
    }
    # Any missing parameters will take from hawks/defaults.json
    generator = hawks.create_generator(config)
    # Run the generator
    generator.run()
    # Let's plot the best individual found
    generator.plot_best_indivs(show=True)
    # Get the best dataset found and it's labels
    datasets, label_sets = generator.get_best_dataset()
    # Stored as a list for multiple runs
    data, labels = datasets[0], label_sets[0]
    # Run KMeans on the data
    km = KMeans(
        n_clusters=len(np.unique(labels)), random_state=SEED_NUM
    ).fit(data)
    # Plot the output of KMeans
    hawks.plotting.scatter_prediction(data, km.labels_)
    # Get the Adjusted Rand Index for KMeans on the data
    ari = adjusted_rand_score(labels, km.labels_)
    print(f"ARI: {ari}")


.. example-marker-end


Documentation
-------------

For further information about how to use HAWKS, including examples, please see the `documentation <https://hawks.readthedocs.io/>`__.


Issues
------

As this work is still in development, plain sailing is not guaranteed.
If you encounter an issue, first ensure that ``hawks`` is running as
intended by navigating to the tests directory, and running
``python tests.py``. If any test fails, please add details of this
alongside your original problem to an issue on the `GitHub repo <https://github.com/sea-shunned/hawks>`__.


Contributing
------------

.. contributing-marker-start

At present, this is primarily academic work, so future developments will be released here after they have been published. If you have any suggestions or simple feature requests for HAWKS as a tool to use, please raise that on the `GitHub repo <https://github.com/sea-shunned/hawks/issues>`__.

I have various directions for HAWKS at present, and can only work on a subset of them, and so involvement with more people would be great. If you would like to extend this work or collaborate, please `contact me <https://sea-shunned.github.io/>`__.

.. contributing-marker-end