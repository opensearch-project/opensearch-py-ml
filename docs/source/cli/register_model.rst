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
    ``opensearch-ml model register``
* Direct model registration with all parameters:
    ``opensearch-ml model register --connectorId 'connector123' --name 'Test model' --description 'This is a test model'``
