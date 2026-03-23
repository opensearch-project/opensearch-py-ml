.. _cli.register_model:

================
Register a model
================

Description
~~~~~~~~~~~

Use the `model register` command to register an externally hosted model with OpenSearch. This process associates a model with a specific connector, enabling you to use the model for various machine learning (ML) tasks within the OpenSearch environment.

Command syntax
~~~~~~~~~~~~~~

``opensearch-ml model register [-h][-i][--connectorId '<value>'][--name '<value>'][--description '<value>'][-o <value>]``

**Options:**

* ``-h``: Show help message and exit
* ``-i``: (Optional) Enable interactive mode for model registration. Off by default
* ``--connectorId '<value>'``: (Optional) The connector ID to associate the model with
* ``--name '<value>'``: (Optional) The name of the model
* ``--description '<value>'``: (Optional) A brief description of the model
* ``-o <value>``: (Optional) The file path to save the output information. If not provided, the output will be saved in the current directory while in non-interactive mode

Usage examples
~~~~~~~~~~~~~~

**Registering a model interactively**

    Use the following command to register a model interactively:

    ``opensearch-ml model register -i``

    Sample response:

    .. code-block:: plaintext

        Starting model registration...

        Setup configuration loaded successfully from /Documents/cli/setup_config.yml

        Enter the model name: OpenAI embedding model
        Enter the model description: This is a test model
        Enter the connector ID: connector123

        Successfully registered a model with ID: model123

        Enter the path to save the output information, or press Enter to save it in the current directory [/Documents/cli/output.yml]: 

        Output information saved successfully to /Documents/cli/output.yml

**Registering a model directly**

    To register a model directly, provide all parameters to the following command:

    ``opensearch-ml model register --connectorId 'connector123' --name 'Test model' --description 'This is a test model' -o /Documents/cli/output.yml``

    If `-o` or `--outputPath` is not specified, the output file will be saved in the current directory in non-interactive mode.

    Sample response:

    .. code-block:: plaintext

        Starting model registration...

        Setup configuration loaded successfully from /Documents/cli/setup_config.yml

        Successfully registered a model with ID: model123

        Output information saved successfully to /Documents/cli/output.yml

Output YAML file
~~~~~~~~~~~~~~~~

After successfully registering a model, the CLI saves important information about the model in an output YAML file. This file contains details that may be needed for future operations or reference. The output YAML file appears as follows:

.. code-block:: yaml

    register_model:
    - model_id: model123
      model_name: OpenAI embedding model
      connector_id: connector123