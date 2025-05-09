.. _cli.create_connector:

================
Create Connector
================

Description
~~~~~~~~~~~

The create connector command allows users to create a connector in both self-managed OpenSearch and Amazon OpenSearch Service (AOS). Users can create a connector either interactively or by using a configuration file.

For self-managed OpenSearch:

* Select from supported connectors and their associated models
* Provide necessary configuration parameters (e.g., API key)

For Amazon OpenSearch Service:

* Automatically handles AWS-specific setup including:

  * IAM role creation and configuration
  * Backend role mapping
  * Secret creation

* Select from supported connectors and their associated models
* Provide AWS-specific parameters (e.g., AWS region)

The CLI guides users through providing only the essential model-specific information needed for their chosen environment.

Prerequisites
~~~~~~~~~~~~~

Before creating a connector, ensure users have completed the setup process or have a setup configuration file. 
See :ref:`setup guide <cli.setup>` for instructions on doing the setup process.

Command Syntax
~~~~~~~~~~~~~~

``opensearch-ml connector create [--path <value>]``

**Option:**

* ``--path <value>``: (Optional) Path to an existing connector configuration YAML file

Usage Examples
~~~~~~~~~~~~~~

* Interactive connector creation:

    Command:

    ``opensearch-ml connector create``

    Sample response:

    .. code-block:: plaintext

        Starting connector creation...

        Enter the path to your existing setup configuration file: /Documents/cli/setup_config.yml

        Setup configuration loaded successfully from /Documents/cli/setup_config.yml

        Please select a supported connector to create in Amazon OpenSearch Service:
        1. Amazon Bedrock
        2. Amazon Bedrock Converse
        3. Amazon Comprehend
        4. Amazon SageMaker
        5. Amazon Textract
        6. Cohere
        7. DeepSeek
        8. OpenAI
        Enter your choice (1-8): 8

        Please select a model for the connector creation: 
        1. Embedding model
        2. Custom model
        Enter your choice (1-2): 1
        Enter your OpenAI API key: ********
        Enter your connector role prefix: test-role    
        Enter a name for the AWS Secrets Manager secret: test-secret

        Creating OpenAI connector...
        Step 1: Create Secret
        test secret exists, skipping creation.
        ----------
        Step 2: Create IAM role configured in connector
        Role 'test-role-openai-connector-03ae00' does not exist.
        ----------
        Step 3: Configure IAM role in OpenSearch
        Step 3.1: Create IAM role for Signing create connector request
        Role 'test-role-openai-connector-create-03ae00' does not exist.
        ----------
        Step 3.2: Map IAM role test-role-openai-connector-create-03ae00 to OpenSearch permission role
        ----------
        Step 4: Create connector in OpenSearch
        Waiting for resources to be ready...
        Time remaining: 1 seconds....
        Wait completed, creating connector...
        Connector role arn: test-connector-role-arn
        ----------

        Successfully created OpenAI connector with ID: connector123

        Enter the path to save the output information, or press Enter to save it in the current directory [/Documents/cli/output.yml]:

        Output information saved successfully to /Documents/cli/output.yml

* Create connector using a configuration file:

    Command:

    ``opensearch-ml connector create --path /Documents/cli/connector_config.yml``

    Assume user has connector_config.yml file with this content:

    .. code-block:: yaml

        setup_config_path: /Documents/cli/setup_config.yml
        connector_name: OpenAI
        model_name: Embedding model
        api_key: test-api-key
        connector_role_prefix: test-role
        connector_secret_name: test-secret


    Sample response:

    .. code-block:: plaintext

        Starting connector creation...

        Connector configuration loaded successfully from /Documents/cli/connector_config.yml

        Setup configuration loaded successfully from /Documents/cli/setup_config.yml

        Creating OpenAI connector...
        Step 1: Create Secret
        test secret exists, skipping creation.
        ----------
        Step 2: Create IAM role configured in connector
        Role 'test-role-openai-connector-03ae00' does not exist.
        ----------
        Step 3: Configure IAM role in OpenSearch
        Step 3.1: Create IAM role for Signing create connector request
        Role 'test-role-openai-connector-create-03ae00' does not exist.
        ----------
        Step 3.2: Map IAM role test-role-openai-connector-create-03ae00 to OpenSearch permission role
        ----------
        Step 4: Create connector in OpenSearch
        Waiting for resources to be ready...
        Time remaining: 1 seconds....
        Wait completed, creating connector...
        Connector role arn: test-connector-role-arn
        ----------

        Successfully created OpenAI connector with ID: connector123

        Enter the path to save the output information, or press Enter to save it in the current directory [/Documents/cli/output.yml]:

        Output information saved successfully to /Documents/cli/output.yml

Setup Configuration YAML file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Template**

.. code-block:: yaml

    setup_config_path:
    connector_name:
    model_name:
    access_token:
    api_key:
    aws_access_key:
    aws_secret_access_key:
    aws_session_token:
    connector_body:
    connector_role_inline_policy:
    connector_role_prefix:
    connector_secret_name:
    endpoint_arn:
    endpoint_url:
    model_id:
    project_id:
    region:
    required_policy:
    required_secret:

Note: The order of the fields does not matter. This template will only be used when users choose to create a connector with a configuration file.


**Field Descriptions**

.. csv-table::
   :file: connector_config.csv
   :widths: 20, 50, 30
   :header-rows: 1

Output YAML file
~~~~~~~~~~~~~~~~

After successfully creating a connector, the CLI saves important information about the connector in an output YAML file. This file contains details that may be needed for future operations or reference. Here's an example of what the output YAML file might look like:

.. code-block:: yaml

    connector_create:
    - connector_id: connector123
      connector_name: OpenAI embedding model connector
      connector_role_arn: test-connector-role-arn
      connector_role_name: test-role-openai-connector-03ae00
      connector_secret_arn: test-connector-secret-arn
      connector_secret_name: test-secret

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

Note: Custom connectors and models are supported for all connector types in both AOS and open-source.