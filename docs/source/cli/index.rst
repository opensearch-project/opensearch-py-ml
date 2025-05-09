.. _cli:

============================
Command Line Interface (CLI)
============================

Opensearch-py-ml supports a Command Line Interface (CLI) feature designed to streamline and simplify ML operations within the OpenSearch environment. This CLI provides an efficient and user-friendly way to manage various ML tasks, including the creation of connectors, registration of remote models, and execution of predictions.

Version Information
~~~~~~~~~~~~~~~~~~~

The CLI feature is available from opensearch-py-ml version 1.2.0 and later.

Installation
~~~~~~~~~~~~

To install the CLI feature:

``pip install opensearch-py-ml``

Verifying Installation
~~~~~~~~~~~~~~~~~~~~~~

To verify that the CLI has been installed correctly, run:

``opensearch-ml --help``

Users should see the following output:

.. code-block:: plaintext

    usage: opensearch-ml [-h] command ...

    OpenSearch ML CLI

    optional arguments:
    -h, --help  show this help message and exit

    Available Commands:
    command
        setup     Initialize and configure OpenSearch setup and AWS credentials
        connector
                  Manage ML connectors
        model     Manage ML models
    ...

Command Reference
~~~~~~~~~~~~~~~~~

The OpenSearch ML CLI provides several commands to manage various aspects of the ML workflow:

.. toctree::
   :maxdepth: 1

   setup
   create_connector
   register_model
   predict_model