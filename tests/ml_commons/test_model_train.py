# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import pytest
from opensearchpy import OpenSearch, helpers
from sklearn.datasets import load_iris
import time
from opensearch_py_ml.ml_commons import MLCommonClient, ModelTrain
from tests import OPENSEARCH_TEST_CLIENT

ml_client = MLCommonClient(OPENSEARCH_TEST_CLIENT)


@pytest.fixture
def iris_index():
    index_name = "test__index__iris_data"
    index_mapping = {
        "mappings": {
            "properties": {
                "sepal_length": {"type": "float"},
                "sepal_width": {"type": "float"},
                "petal_length": {"type": "float"},
                "petal_width": {"type": "float"},
                "species": {"type": "keyword"},
            }
        }
    }

    if ml_client._client.indices.exists(index=index_name):
        ml_client._client.indices.delete(index=index_name)
    ml_client._client.indices.create(index=index_name, body=index_mapping)

    iris = load_iris()
    iris_data = iris.data
    iris_target = iris.target
    iris_species = [iris.target_names[i] for i in iris_target]

    actions = [
        {
            "_index": index_name,
            "_source": {
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width,
                "species": species,
            },
        }
        for (sepal_length, sepal_width, petal_length, petal_width), species in zip(
            iris_data, iris_species
        )
    ]

    helpers.bulk(ml_client._client, actions)
    # without the sleep, test is failing.
    time.sleep(2)

    yield index_name

    ml_client._client.indices.delete(index=index_name)


def test_init():
    assert isinstance(ml_client._client, OpenSearch)
    assert isinstance(ml_client._model_train, ModelTrain)


def test_train(iris_index):
    algorithm_name = "kmeans"
    input_json_sync = {
        "parameters": {"centroids": 3, "iterations": 10, "distance_type": "COSINE"},
        "input_query": {
            "_source": ["petal_length", "petal_width"],
            "size": 10000,
        },
        "input_index": [iris_index],
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
        "input_index": [iris_index],
    }
    response = ml_client.train_model(algorithm_name, input_json_async, is_async=True)

    assert isinstance(response, dict)
    assert "task_id" in response
    assert "status" in response
    assert response["status"] == "CREATED"
