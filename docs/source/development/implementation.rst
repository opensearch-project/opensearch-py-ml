.. _implementation/details:

======================
Implementation Details
======================

The goal of an ``opensearch_py_ml.DataFrame`` is to enable users who are familiar with ``pandas.DataFrame``
to access, explore and manipulate data that resides in Opensearch.

Ideally, all data should reside in Opensearch and not to reside in memory.
This restricts the API, but allows access to huge data sets that do not fit into memory, and allows
use of powerful Elasticsearch features such as aggregations.


Pandas and 3rd Party Storage Systems
------------------------------------

Generally, integrations with `3rd party storage systems <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html>`_
(SQL, Google Big Query etc.) involve accessing these systems and reading all external data into an
in-core pandas data structure. This also applies to `Apache Arrow <https://arrow.apache.org/docs/python/pandas.html>`_
structures.

Whilst this provides access to data in these systems, for large datasets this can require significant
in-core memory, and for systems such as Opensearch, bulk export of data can be an inefficient way
of exploring the data.

An alternative option is to create an API that proxies ``pandas.DataFrame``-like calls to Opensearch
queries and operations. This could allow the Opensearch cluster to perform operations such as
aggregations rather than exporting all the data and performing this operation in-core.

Implementation Options
----------------------

An option would be to replace the ``pandas.DataFrame`` backend in-core memory structures with Opensearch
accessors. This would allow full access to the ``pandas.DataFrame`` APIs. However, this has issues:

*   If a ``pandas.DataFrame`` instance maps to an index, typical manipulation of a ``pandas.DataFrame``
    may involve creating many derived ``pandas.DataFrame`` instances. Constructing an index per
    ``pandas.DataFrame`` may result in many Opensearch indexes and a significant load on Opensearch.
    For example, ``df_a = df['a']`` should not require Opensearch indices ``df`` and ``df_a``

*   Not all ``pandas.DataFrame`` APIs map to things we may want to do in Opensearch. In particular,
    API calls that involve exporting all data from Opensearch into memory e.g. ``df.to_dict()``.

*   The backend ``pandas.DataFrame`` structures are not easily abstractable and are deeply embedded in
    the implementation.

Another option is to create a ``opensearch_py_ml.DataFrame`` API that mimics appropriate aspects of
the ``pandas.DataFrame`` API. This resolves some of the issues above as:

*   ``df_a = df['a']`` could be implemented as a change to the Opensearch query used, rather
    than a new index

*   Instead of supporting the entire ``pandas.DataFrame`` API we can support a subset appropriate for
    Opensearch. If addition calls are required, we could to create a ``opensearch_py_ml.DataFrame._to_pandas()``
    method which would explicitly export all data to a ``pandas.DataFrame``

*   Creating a new ``opensearch_py_ml.DataFrame`` API gives us full flexibility in terms of implementation. However,
    it does create a large amount of work which may duplicate a lot of the ``pandas`` code - for example,
    printing objects etc. - this creates maintenance issues etc.
