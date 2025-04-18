.. _cli.create_connector:

===============
Crete Connector
===============

Description
~~~~~~~~~~~

The create connector command allows users to create a connector, including configuring IAM roles, mapping backend roles, and creating secrets automatically. Users can create a connector either interactively or by using a configuration file. Connectors can be created by simply selecting from a list of supported connectors and their associated models. The CLI then guides them through providing only the essential model-specific information, such as AWS region for Amazon services.

Command Syntax
~~~~~~~~~~~~~~

``opensearch-ml connector create [--path <value>]``

**Option:**

* ``--path <value>``: Path to an existing connector configuration YAML file

Usage Examples
~~~~~~~~~~~~~~

* Interactive connector creation:
    ``opensearch-ml connector create``
* Create connector using a configuration file:
    ``opensearch-ml connector create --path Documents/cli/connector_config.yml``

Setup Configuration YAML file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Template**

.. code-block:: JSON

    setup_config_path:
    connector_name:
    model_name:
    access_token:
    api_key:
    aws_access_key:
    aws_secret_access_key:
    aws_session_token:
    connector_body:
    connector_role_prefix:
    connector_secret_name:
    endpoint_arn:
    endpoint_url:
    model_id:
    project_id:
    region:

Note: The order of the fields does not matter. This template will only be used when users choose to create a connector with a configuration file.


**Field Descriptions**

.. csv-table::
   :file: connector_config.csv
   :widths: 20, 50, 30
   :header-rows: 1


Supported Connectors and Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Amazon OpenSearch Service (AOS)**

.. csv-table::
   :file: aos_connector.csv
   :widths: 40 60
   :header-rows: 1

**Open-source**

.. csv-table::
   :file: opensource_connector.csv
   :widths: 40 60
   :header-rows: 1

Note: Custom models are supported for all connectors both in AOS and open-source.