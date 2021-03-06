Changelog
=========

v0.2.0 (03-02-2020)
-------------------

* Removed old ``user_guide.md`` and replaced with more extensive `documentation <https://hawks.readthedocs.io>`_ (includes an overhaul of the :ref:`examples_page`)
* Added parameterization for initialziation ranges of means and covariances
* Added :meth:`~hawks.generator.SingleGenerator.run_step` method to step through a run (yielding control every time), and accompanying example
* Added more generic plotting method to generator (see e.g. :meth:`~hawks.generator.BaseGenerator.plot_best_indivs`)
* Enabled "multi_config" for multiple mutation operators (all mutation operators now have their defaults arguments in the default JSON file). See :ref:`parameters_page` for further details.
* Added config options for parental and environmental selection
* Added argument to select the comparison method for determining the best individual from the final population (by fitness or rank determined by stochastic ranking)
* Added more plotting functions (essentially just wrappers for `seaborn <https://seaborn.pydata.org/index.html>`_)
* Added more arguments for plotting, such as automatic cleaning of labels and placement of legends, and hatching for boxplots
* Consolidated ``prepare.py`` into the :class:`~hawks.generator.BaseGenerator`
* Fixed various inconsistencies/bugs
* Expanded tests


v0.1.0 (28-08-2019)
-------------------

* Added functions for cluster analysis and instance space (both creation and plotting)
* Problem features for instance space are in ``problem_features.py``
* Added extra test for overlap computation
* Added examples for the new features
* Added ability to produce an animation of a single run (producing the gif requires the command-line `convert` utility from ImageMagick)
* General improvements to workflow and reduced some unnecessary computation

v0.0.2 (29-06-2019)
-------------------

* Added plotting functions
* Improved handling of multi_config individuals

v0.0.1 (25-06-2019)
-------------------

* Initial release