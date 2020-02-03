.. _testing_page:

Testing
=======

HAWKS includes a batch of :mod:`unittest` s, to help ensure that there are no breaking changes to functionality. At present, the tests cover the core components of the code, but more may be added overtime to increase coverage.

For new features, a test (or multiple) should be added to ensure the component works as expected. A ``validation.json`` config is provided if required, otherwise the relevant parts of HAWKS can be created piecemeal (examples can be found in the existing tests for this).

To run the tests, just navigate to the root tests folder, and run:

.. code-block:: bash

    python tests.py

Alongside any desired command-line flags for :mod:`unittest`.