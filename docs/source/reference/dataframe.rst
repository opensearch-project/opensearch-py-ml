.. _api.dataframe:

=========
DataFrame
=========
.. currentmodule:: opensearch_py_ml

Constructor
~~~~~~~~~~~
.. toctree::
   :maxdepth: 2

   api/DataFrame

Attributes and Underlying Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 2

   api/DataFrame.index
   api/DataFrame.columns
   api/DataFrame.dtypes
   api/DataFrame.select_dtypes
   api/DataFrame.values
   api/DataFrame.empty
   api/DataFrame.shape
   api/DataFrame.ndim
   api/DataFrame.size

Indexing, Iteration
~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 2

   api/DataFrame.head
   api/DataFrame.keys
   api/DataFrame.tail
   api/DataFrame.get
   api/DataFrame.query
   api/DataFrame.sample
   api/DataFrame.iterrows
   api/DataFrame.itertuples

Function Application, GroupBy & Window
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    Opensearch aggregations using cardinality (``count``) are accurate
    approximations using the `HyperLogLog++ algorithm`_ so may not
    be exact.

.. _HyperLogLog++ algorithm: https://static.googleusercontent.com/media/research.google.com/fr//pubs/archive/40671.pdf

.. toctree::
   :maxdepth: 2

   api/DataFrame.agg
   api/DataFrame.aggregate
   api/DataFrame.groupby

.. currentmodule:: opensearch_py_ml.groupby

.. toctree::
   :maxdepth: 2

   api/groupby.DataFrameGroupBy
   api/groupby.DataFrameGroupBy.agg
   api/groupby.DataFrameGroupBy.aggregate
   api/groupby.DataFrameGroupBy.count
   api/groupby.DataFrameGroupBy.mad
   api/groupby.DataFrameGroupBy.max
   api/groupby.DataFrameGroupBy.mean
   api/groupby.DataFrameGroupBy.median
   api/groupby.DataFrameGroupBy.min
   api/groupby.DataFrameGroupBy.nunique
   api/groupby.DataFrameGroupBy.std
   api/groupby.DataFrameGroupBy.sum
   api/groupby.DataFrameGroupBy.var
   api/groupby.DataFrameGroupBy.quantile

.. currentmodule:: opensearch_py_ml

.. _api.dataframe.stats:

Computations / Descriptive Stats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 2

   api/DataFrame.count
   api/DataFrame.describe
   api/DataFrame.info
   api/DataFrame.max
   api/DataFrame.mean
   api/DataFrame.min
   api/DataFrame.median
   api/DataFrame.mad
   api/DataFrame.std
   api/DataFrame.var
   api/DataFrame.sum
   api/DataFrame.nunique
   api/DataFrame.mode
   api/DataFrame.quantile
   api/DataFrame.idxmax
   api/DataFrame.idxmin

Reindexing / Selection / Label Manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 2

   api/DataFrame.drop
   api/DataFrame.filter

Plotting
~~~~~~~~
.. toctree::
   :maxdepth: 2

   api/DataFrame.hist

Opensearch Functions
~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 2

   api/DataFrame.os_info
   api/DataFrame.es_match
   api/DataFrame.es_query
   api/DataFrame.os_dtypes

Serialization / IO / Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 2

   api/DataFrame.info
   api/DataFrame.to_numpy
   api/DataFrame.to_csv
   api/DataFrame.to_html
   api/DataFrame.to_string
   api/DataFrame.to_pandas
