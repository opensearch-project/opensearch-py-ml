# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import pytest

from opensearchpy import OpenSearch

from opensearch_py_ml.ml_commons import MLCommonClient
from opensearch_py_ml.ml_commons import ModelTrain


def test_init(opensearch_client):
    ml_client = MLCommonClient(opensearch_client)
    assert isinstance(ml_client._client, OpenSearch)
    assert isinstance(ml_client._model_train, ModelTrain)


def test_train(iris_index_client):
    client, test_index_name = iris_index_client
    ml_client = MLCommonClient(client)
    algorithm_name = "kmeans"
    input_json_sync = {
        "parameters": {"centroids": 3, "iterations": 10, "distance_type": "COSINE"},
        "input_query": {
            "_source": ["petal_length", "petal_width"],
            "size": 10000,
        },
        "input_index": [test_index_name],
    }
    response = ml_client.train_model(algorithm_name, input_json_sync)
    assert isinstance(response, dict)
    assert "model_id" in response
    assert "status" in response
    assert response["status"] == "COMPLETED"

    input_json_async = {
        "parameters": {"centroids": 3, "iterations": 10, "distance_type": "COSINE"},
        "input_query": {
            "_source": ["petal_length", "petal_width"],
            "size": 10000,
        },
        "input_index": [test_index_name],
    }
    response = ml_client.train_model(algorithm_name, input_json_async, is_async=True)

    assert isinstance(response, dict)
    assert "task_id" in response
    assert "status" in response
    assert response['status'] == "CREATED"
