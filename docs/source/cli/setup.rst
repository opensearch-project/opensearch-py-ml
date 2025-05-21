.. _cli.setup:

==============
Set up the CLI
==============

Description
~~~~~~~~~~~

The `setup` command is essential for configuring OpenSearch and AWS credentials (if required) before running any machine learning (ML) operations. The CLI supports the following service types:

- Amazon OpenSearch Service
- Open-source service (self-managed OpenSearch)

You can initiate the setup in the following ways:

- Using an interactive prompt (for first-time setup)
- Using a configuration file (for verifying configurations or renewing AWS credentials)

After running the `setup` command, a configuration YAML file will be generated. This file is crucial for subsequent operations, including connector creation. 

Command syntax
~~~~~~~~~~~~~~

``opensearch-ml setup [--path <value>]``

**Option:**

* ``--path <value>``: (Optional) The path to an existing setup configuration YAML file

Usage examples
~~~~~~~~~~~~~~

**First-time setup (interactive)**

    To set up the CLI interactively, run the following command:

    ``opensearch-ml setup``

    Sample response:

    .. code-block:: plaintext

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

**Setup using an existing configuration file**

    To use an existing configuration file for setup, run the following command:

    ``opensearch-ml setup --path /Documents/cli/setup_config.yml``

    This example assumes that you have a `setup_config.yml` file at the specified path with the following content:

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

    .. code-block:: plaintext

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

Setup configuration YAML file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use a setup configuration YAML file to specify your OpenSearch service settings and authentication details needed for the CLI.

**Configuration file template**

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

**Field descriptions**

.. csv-table::
   :file: setup_config.csv
   :widths: 25, 50, 25
   :header-rows: 1

Notes
~~~~~

* For Amazon OpenSearch Service, ensure that you provide either the ``aws_role_name`` or the ``aws_user_name``, but not both.
* The generated configuration file is crucial for subsequent CLI operations. Keep it secure and accessible.
* To update AWS credentials or change the configuration, edit the YAML file directly or run the setup command specifying the ``--path`` option.