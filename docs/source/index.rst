.. module:: opensearch_py_ml

**********************************************************************
Opensearch-py-ml: DataFrames and Machine Learning backed by Opensearch
**********************************************************************

**Date**: |today| **Version**: |version|


**Useful links**:
`Source Repository <https://github.com/opensearch-project/opensearch-py-ml>`__ |
`Issues & Ideas <https://github.com/opensearch-project/opensearch-py-ml/issues>`__


Opensearch-py-ml is a Python Opensearch client for exploring and analyzing data
in Opensearch with a familiar Pandas-compatible API.

Where possible the package uses existing Python APIs and data structures to make it easy to switch between numpy,
pandas, scikit-learn to their Opensearch powered equivalents. In general, the data resides in Opensearch and
not in memory, which allows Opensearch-py-ml to access large datasets stored in Opensearch.

Installing Opensearch-py-ml
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Opensearch-py-ml can be installed from `PyPI <https://pypi.org/project/opensearch-py-ml>`_ via pip:

 .. code-block:: bash

    $ python -m pip install opensearch-py-ml


Getting Started
~~~~~~~~~~~~~~~

If it's your first time using Opensearch we recommend looking through the
:doc:`examples/index` documentation for ideas on what Opensearch-py-ml is capable of.

If you're new to Opensearch we recommend `reading the documentation <https://opensearch.org/docs/latest/>`_.

.. toctree::
   :maxdepth: 2
   :hidden:

   reference/index
   examples/index
   reference/pre_trained_models

* :doc:`reference/index`

  * :doc:`reference/dataframe`
  * :doc:`reference/series`
  * :doc:`reference/general_utility_functions`
  * :doc:`reference/io`
  * :doc:`reference/mlcommons`
  * :doc:`reference/sentencetransformer`

* :doc:`examples/index`

  * :doc:`examples/demo_notebook`
  * :doc:`examples/online_retail_analysis`
  * :doc:`examples/demo_transformer_model_train_save_upload_to_openSearch`


* `License <https://github.com/opensearch-project/opensearch-py-ml/blob/main/LICENSE>`_
* `Contributing <https://github.com/opensearch-project/opensearch-py-ml/blob/main/CONTRIBUTING.md>`_
* `Code of Conduct <https://github.com/opensearch-project/opensearch-py-ml/blob/main/CODE_OF_CONDUCT.md>`_


