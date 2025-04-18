.. _cli.setup:

=====
Setup
=====

Description
~~~~~~~~~~~

The setup command is essential for configuring OpenSearch and AWS credentials (if required) before running any ML operations. This CLI supports two types of services:

1. Amazon OpenSearch Service (AOS)
2. Open-source service (self-managed OpenSearch)

The setup can be initiated in two ways:

1. Interactive prompt (for first-time setup)
2. Using a configuration file (for checking configurations or renewing AWS credentials)

After running the setup command, a configuration YAML file will be generated. This file is crucial for subsequent operations, including connector creation. 

Command Syntax
~~~~~~~~~~~~~~

``opensearch-ml setup [--path <value>]``

**Option:**

* ``--path <value>``: Path to an existing setup configuration YAML file

Usage Examples
~~~~~~~~~~~~~~

* First-time setup (interactive):
    ``opensearch-ml setup``
* Setup using an existing configuration file:
    ``opensearch-ml setup --path Documents/cli/setup_config.yml``

Setup Configuration YAML file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Template**

.. code-block:: JSON

    service_type:
    ssl_check_enabled:
        opensearch_config:
        opensearch_domain_region:
        opensearch_domain_endpoint:
        opensearch_domain_username:
        opensearch_domain_password:
    aws_credentials:
        aws_role_name:
        aws_user_name:
        aws_access_key:
        aws_secret_access_key:
        aws_session_token:


**Field Descriptions**

.. csv-table::
   :file: setup_config.csv
   :widths: 25, 50, 25
   :header-rows: 1

Notes
~~~~~

* For Amazon OpenSearch Service, ensure users provide either ``aws_role_name`` or ``aws_user_name``, not both.
* The generated configuration file is crucial for subsequent CLI operations. Keep it secure and accessible.
* If users need to update AWS credentials or change configurations, they can edit the YAML file directly or run the setup command with the ``--path`` option.