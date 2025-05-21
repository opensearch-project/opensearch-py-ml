.. _cli.predict_model:

======================
Run a model prediction
======================

Description
~~~~~~~~~~~

Use the `model predict` command to run predictions using a machine learning (ML) model registered in OpenSearch. Depending on the model type, you can use ML models for various tasks, such as embedding creation.

Command syntax
~~~~~~~~~~~~~~

``opensearch-ml model predict [--modelId '<value>'][--body '<value>']``

**Options:**

* ``--modelId '<value>'``: (Optional) The model ID to use for prediction
* ``--body '<value>'``: (Optional) The prediction request body in JSON format

Usage examples
~~~~~~~~~~~~~~

**Running a model prediction interactively**
    
    Use the following command to run a model prediction interactively:

    ``opensearch-ml model predict``

    Sample response:

    .. code-block:: plaintext

        Starting model prediction...

        Setup configuration loaded successfully from /Documents/cli/setup_config.yml

        Enter the model ID: model123

        Enter your predict request body as a JSON object (press Enter twice when done): 
        {
            "parameters": {
                "input": ["hello world", "how are you"]
            }
        }

        Predict Response: {"inference_results": [{"output": [{"name": "sentence_embedding", "data_type": "FLOAT32", "shape": [1536], "data": [-0.016099498, 0.001368687, -0.019484723, -0.033694793, -0.026005873, 0.0076758, -0.0..."status_code": 200}]}

        Successfully predict the model.
        Do you want to save the full prediction output? (yes/no): yes

        Enter the path to save the output information, or press Enter to save it in the current directory [/Documents/cli/output.yml]: 

        Output information saved successfully to /Documents/cli/output.yml


**Making a direct model prediction**

    To make a direct model prediction, provide all parameters to the following command:

    ``opensearch-ml model predict --modelId 'model123' --body '{"parameters": {"input": ["hello world", "how are you"]}}'``

    Sample response:
    
    .. code-block:: plaintext

        Starting model prediction...

        Setup configuration loaded successfully from /Documents/cli/setup_config.yml
        Predict Response: {"inference_results": [{"output": [{"name": "sentence_embedding", "data_type": "FLOAT32", "shape": [1536], "data": [-0.016099498, 0.001368687, -0.019484723, -0.033694793, -0.026005873, 0.0076758, -0.0..."status_code": 200}]}

        Successfully predict the model.
        Do you want to save the full prediction output? (yes/no): yes

        Enter the path to save the output information, or press Enter to save it in the current directory [/Documents/cli/output.yml]: 

        Output information saved successfully to /Documents/cli/output.yml

Output YAML file
~~~~~~~~~~~~~~~~

After successfully running a model prediction, the CLI saves important information about the prediction in an output YAML file. This file contains details that may be needed for future operations or reference. The output YAML file appears as follows:

.. code-block:: yaml

    predict_model:
    - model_id: model123
      response: '{"inference_results": [{"output": [{"name": "sentence_embedding", "data_type": "FLOAT32", "shape": [1536], "data": [-0.016099498, 0.001368687, -0.019484723, -0.033694793, -0.026005873, ..., -0.005324199]}], "status_code": 200}]}' 