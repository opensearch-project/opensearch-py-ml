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

* ``--path <value>``: (Optional) Path to an existing setup configuration YAML file

Usage Examples
~~~~~~~~~~~~~~

* First-time setup (interactive):

    Command:

    ``opensearch-ml setup``

    Sample response:

    .. code-block:: JSON

        Starting connector setup...

        Do you already have a configuration file? (yes/no): no
        Let's create a new configuration file.

        Choose OpenSearch service type:
        1. Amazon OpenSearch Service
        2. Open-source
        Enter your choice (1-2): 1

        --- Amazon OpenSearch Service Setup ---
        Let's configure your AWS credentials.
        Enter your AWS Access Key ID: ****
        Enter your AWS Secret Access Key: ****
        Enter your AWS Session Token: ****
        New AWS credentials have been successfully configured and verified.

        Choose ARN type:
        1. IAM Role ARN
        2. IAM User ARN
        Enter your choice (1-2): 1
        Enter your AWS IAM Role ARN: test-arn

        Enter your AWS OpenSearch region, or press Enter for default [us-west-2]: 
        Enter your AWS OpenSearch domain endpoint: test-domain
        Enter your AWS OpenSearch username: admin
        Enter your AWS OpenSearch password: ****

        Enter the path to save the configuration information, or press Enter to save it in the current directory [/Documents/cli/setup_config.yml]: 

        Configuration information saved successfully to /Documents/cli/setup_config.yml
        Initialized OpenSearch client with host: test-domain and port: 443

        Setup complete. You are now ready to use the ML features.

* Setup using an existing configuration file:

    Command:

    ``opensearch-ml setup --path /Documents/cli/setup_config.yml``

    Assume user has setup_config.yml file with this content:

    .. code-block:: yaml

        service_type: amazon-opensearch-service
        ssl_check_enabled: true
        opensearch_config:
            opensearch_domain_region: us-west-2
            opensearch_domain_endpoint: test-domain
            opensearch_domain_username: admin
            opensearch_domain_password: pass
        aws_credentials:
            aws_role_name: test-arn
            aws_user_name: ''
            aws_access_key: test-access-key
            aws_secret_access_key: test-secret-access-key
            aws_session_token: test-session-token

    Sample response:

    .. code-block:: JSON

        Starting connector setup...

        Setup configuration loaded successfully from /Documents/cli/setup_config.yml
        Your AWS credentials are invalid or have expired.
        Let's configure your AWS credentials.
        Enter your AWS Access Key ID: ****
        Enter your AWS Secret Access Key: ****
        Enter your AWS Session Token: ****
        New AWS credentials have been successfully configured and verified.
        Configuration saved successfully to /Documents/cli/setup_config.yml

        Setup complete. You are now ready to use the ML features.

Setup Configuration YAML file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Template**

.. code-block:: yaml

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