.. _cli.predict_model:

=============
Predict Model
=============

Description
~~~~~~~~~~~

The predict model command allows users to run predictions using a registered model in OpenSearch. This enables users to leverage ML models for various tasks, such as embedding creation, depending on the model type.

Command Syntax
~~~~~~~~~~~~~~

``opensearch-ml model predict [--modelId '<value>'][--body '<value>']``

**Options:**

* ``--modelId '<value>'``: The model ID to use for prediction
* ``--body '<value>'``: The prediction request body in JSON format

Usage Examples
~~~~~~~~~~~~~~

* Interactive model prediction:
    ``opensearch-ml model predict``
* Direct model prediction with all parameters:
    ``opensearch-ml model predict --modelId 'model123' --body '{"test": "body"}'``
