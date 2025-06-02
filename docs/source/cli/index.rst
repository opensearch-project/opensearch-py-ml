.. _cli:

============================
Command Line Interface (CLI)
============================

The `opensearch-py-ml` client supports a command-line interface (CLI) designed to streamline and simplify ML operations within the OpenSearch environment. This CLI provides an efficient and user-friendly way to manage various ML tasks, including connector creation, remote model registration, and prediciton execution.

Version information
~~~~~~~~~~~~~~~~~~~

CLI is available for `opensearch-py-ml` versions 1.2.0 and later.

Installation
~~~~~~~~~~~~

To install the CLI, run the following command:

``pip install opensearch-py-ml``

Verifying installation
~~~~~~~~~~~~~~~~~~~~~~

To verify that the CLI has been installed correctly, run the following command:

``opensearch-ml --help``

You should see the following output:

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