.. _cli.register_model:

==============
Register Model
==============

Description
~~~~~~~~~~~

The register model command allows users to register a remote model with OpenSearch. This process associates a model with a specific connector, enabling users to use the model for various ML tasks within the OpenSearch environment.

Command Syntax
~~~~~~~~~~~~~~

``opensearch-ml model register [--connectorId '<value>'][--name '<value>'][--description '<value>']``

**Options:**

* ``--connectorId '<value>'``: The connector ID to associate the model with
* ``--name '<value>'``: The name of the model
* ``--description '<value>'``: A brief description of the model

Usage Examples
~~~~~~~~~~~~~~

* Interactive model registration:

    Command:

    ``opensearch-ml model register``

    Sample response:

    .. code-block:: JSON

        Starting model registration...

        Setup configuration loaded successfully from /Documents/cli/setup_config.yml

        Enter the model name: OpenAI embedding model
        Enter the model description: This is a test model
        Enter the connector ID: connector123

        Successfully registered a model with ID: model123

        Enter the path to save the output information, or press Enter to save it in the current directory [/Documents/cli/output.yml]: 

        Output information saved successfully to /Documents/cli/output.yml

* Direct model registration with all parameters:

    Command:

    ``opensearch-ml model register --connectorId 'connector123' --name 'Test model' --description 'This is a test model'``

    Sample response:

    .. code-block:: JSON

        Starting model registration...

        Setup configuration loaded successfully from /Documents/cli/setup_config.yml

        Successfully registered a model with ID: model123

        Enter the path to save the output information, or press Enter to save it in the current directory [/Documents/cli/output.yml]: 

        Output information saved successfully to /Documents/cli/output.yml

Output YAML file
~~~~~~~~~~~~~~~~

After successfully register a model, the CLI saves important information about the model in an output YAML file. This file contains details that may be needed for future operations or reference. Here's an example of what the output YAML file might look like:

.. code-block:: yaml

    register_model:
    - model_id: model123
      model_name: OpenAI embedding model
      connector_id: connector123