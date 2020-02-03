.. _generator_page:

The Generator
=============

Most interactions with HAWKS are through the creation of a :class:`~hawks.generator.BaseGenerator` object. This is typically done by passing a config to :func:`hawks.generator.create_generator`, which then creates the appropriate child-class of :class:`~hawks.generator.BaseGenerator`. This is determined by the ``"mode"`` parameter, which at present can only take ``"single"`` as an option (to create a :class:`~hawks.generator.SingleObjective` object).

This can be seen in each of the :ref:`Examples`. Further documentation on this generator can be found at :mod:`hawks.generator`, along with more details on the parameters given to it on the :ref:`parameters_page` page.

At it's simplest, we can create datasets using default values by the following code:

.. code-block:: python

    import hawks

    gen = hawks.create_generator()
    gen.run()