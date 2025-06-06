# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import os
import shutil
import time
from json import JSONDecodeError
from os.path import exists

import pytest
from opensearchpy import OpenSearch, helpers
from opensearchpy.exceptions import RequestError
from sklearn.datasets import load_iris

from opensearch_py_ml.ml_commons import MLCommonClient
from opensearch_py_ml.ml_commons.model_uploader import ModelUploader
from opensearch_py_ml.ml_models.sentencetransformermodel import SentenceTransformerModel
from tests import OPENSEARCH_TEST_CLIENT

ml_client = MLCommonClient(OPENSEARCH_TEST_CLIENT)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

TESTDATA_FILENAME = os.path.join(
    os.path.dirname(os.path.abspath("__file__")), "tests", "sample_zip.zip"
)

TESTDATA_UNZIP_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath("__file__")), "tests", "sample_zip"
)

MODEL_FILE_ZIP_NAME = "test_model.zip"
MODEL_FILE_PT_NAME = "test_model.pt"
MODEL_CONFIG_FILE_NAME = "ml-commons_model_config.json"

TEST_FOLDER = os.path.join(PROJECT_DIR, "test_model_files")
TESTDATA_SYNTHETIC_QUERY_ZIP = os.path.join(PROJECT_DIR, "..", "synthetic_queries.zip")
MODEL_PATH = os.path.join(TEST_FOLDER, MODEL_FILE_ZIP_NAME)
MODEL_CONFIG_FILE_PATH = os.path.join(TEST_FOLDER, MODEL_CONFIG_FILE_NAME)

test_model = SentenceTransformerModel(folder_path=TEST_FOLDER, overwrite=True)

PRETRAINED_MODEL_NAME = "huggingface/sentence-transformers/all-MiniLM-L12-v2"
PRETRAINED_MODEL_VERSION = "1.0.1"
PRETRAINED_MODEL_FORMAT = "TORCH_SCRIPT"


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


def clean_test_folder(TEST_FOLDER):
    if os.path.exists(TEST_FOLDER):
        for files in os.listdir(TEST_FOLDER):
            sub_path = os.path.join(TEST_FOLDER, files)
            if os.path.isfile(sub_path):
                os.remove(sub_path)
            else:
                try:
                    shutil.rmtree(sub_path)
                except OSError as err:
                    print(
                        "Fail to delete files, please delete all files in "
                        + str(TEST_FOLDER)
                        + " "
                        + str(err)
                    )

        shutil.rmtree(TEST_FOLDER)


clean_test_folder(TEST_FOLDER)


def test_init():
    assert isinstance(ml_client._client, OpenSearch)
    assert isinstance(ml_client._model_uploader, ModelUploader)


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

    with pytest.raises(JSONDecodeError):
        ml_client.train_model(algorithm_name, "", is_async=True)

    with pytest.raises(RequestError):
        ml_client.train_model(algorithm_name, {}, is_async=True)


def test_execute():
    try:
        input_json = {"operation": "max", "input_data": [1.0, 2.0, 3.0]}
        result = ml_client.execute(
            algorithm_name="local_sample_calculator", input_json=input_json
        )
    except Exception as ex:
        pytest.fail(
            f"Raised Exception during execute API testing with dictionary. Exception info: {ex}"
        )
    else:
        assert result["output"]["result"] == 3

    try:
        input_json = '{"operation": "max", "input_data": [1.0, 2.0, 3.0]}'
        result = ml_client.execute(
            algorithm_name="local_sample_calculator", input_json=input_json
        )
    except Exception as ex:
        pytest.fail(
            f"Raised Exception during execute API testing with JSON string. Exception info: {ex}"
        )
    else:
        assert result["output"]["result"] == 3


def test_DEPRECATED_integration_pretrained_model_upload_unload_delete():
    try:
        model_id = ml_client.upload_pretrained_model(
            model_name=PRETRAINED_MODEL_NAME,
            model_version=PRETRAINED_MODEL_VERSION,
            model_format=PRETRAINED_MODEL_FORMAT,
            load_model=True,
            wait_until_loaded=True,
        )
        ml_model_status = ml_client.get_model_info(model_id)
    except Exception as ex:
        pytest.fail(
            f"Raised Exception during pretrained model registration and deployment. Exception info:{ex}"
        )
    else:
        assert ml_model_status.get("model_state") != "DEPLOY_FAILED"

    if model_id:
        try:
            ml_model_status = ml_client.get_model_info(model_id)
        except Exception as ex:
            pytest.fail(
                f"Raised Exception in getting pretrained model info. Exception info: {ex}"
            )
        else:
            assert ml_model_status.get("model_format") == "TORCH_SCRIPT"
            assert ml_model_status.get("algorithm") == "TEXT_EMBEDDING"

        try:
            ml_client.unload_model(model_id)
            ml_model_status = ml_client.get_model_info(model_id)
        except Exception as ex:
            pytest.fail(
                f"Raised Exception in pretrained model undeployment. Exception info: {ex}"
            )
        else:
            assert ml_model_status.get("model_state") != "UNDEPLOY_FAILED"

        try:
            delete_model_obj = ml_client.delete_model(model_id)
        except Exception as ex:
            pytest.fail(
                f"Raised Exception in deleting pretrained model. Exception info: {ex}"
            )
        else:
            assert delete_model_obj.get("result") == "deleted"


def test_integration_pretrained_model_register_undeploy_delete():
    try:
        model_id = ml_client.register_pretrained_model(
            model_name=PRETRAINED_MODEL_NAME,
            model_version=PRETRAINED_MODEL_VERSION,
            model_format=PRETRAINED_MODEL_FORMAT,
            deploy_model=True,
            wait_until_deployed=True,
        )
        ml_model_status = ml_client.get_model_info(model_id)
    except Exception as ex:
        pytest.fail(
            f"Raised Exception during pretrained model registration and deployment. Exception info:{ex}"
        )
    else:
        assert ml_model_status.get("model_state") != "DEPLOY_FAILED"

    if model_id:
        try:
            ml_model_status = ml_client.get_model_info(model_id)
        except Exception as ex:
            pytest.fail(
                f"Raised Exception in getting pretrained model info. Exception info: {ex}"
            )
        else:
            assert ml_model_status.get("model_format") == "TORCH_SCRIPT"
            assert ml_model_status.get("algorithm") == "TEXT_EMBEDDING"

        try:
            ml_client.undeploy_model(model_id)
            ml_model_status = ml_client.get_model_info(model_id)
        except Exception as ex:
            pytest.fail(
                f"Raised Exception in pretrained model undeployment. Exception info: {ex}"
            )
        else:
            assert ml_model_status.get("model_state") != "UNDEPLOY_FAILED"

        try:
            delete_model_obj = ml_client.delete_model(model_id)
        except Exception as ex:
            pytest.fail(
                f"Raised Exception in deleting pretrained model. Exception info: {ex}"
            )
        else:
            assert delete_model_obj.get("result") == "deleted"


def test_DEPRECATED_integration_model_train_upload_full_cycle():
    # first training the model with small epoch
    test_model.train(
        read_path=TESTDATA_SYNTHETIC_QUERY_ZIP,
        output_model_name=MODEL_FILE_PT_NAME,
        zip_file_name=MODEL_FILE_ZIP_NAME,
        num_epochs=1,
        overwrite=True,
    )
    # second generating the config file to create metadoc of the model in opensearch.
    test_model.make_model_config_json()
    model_file_exists = exists(MODEL_PATH)
    model_config_file_exists = exists(MODEL_CONFIG_FILE_PATH)
    assert model_file_exists == True
    assert model_config_file_exists == True
    if model_file_exists and model_config_file_exists:
        model_id = ""
        task_id = ""
        try:
            model_id = ml_client.upload_model(
                MODEL_PATH, MODEL_CONFIG_FILE_PATH, load_model=False, isVerbose=True
            )
            print("Model_id:", model_id)
        except Exception as ex:  # noqa: E722
            pytest.fail(f"Exception occurred when uploading model: {ex}")

        if model_id:
            try:
                ml_load_status = ml_client.load_model(model_id, wait_until_loaded=False)
                task_id = ml_load_status.get("task_id")
                ml_model_status = ml_client.get_model_info(model_id)
            except Exception as ex:
                pytest.fail(f"Exception occurred when loading model: {ex}")
            else:
                assert task_id != "" or task_id is not None
                assert ml_model_status.get("model_state") != "DEPLOY_FAILED"

            try:
                ml_model_status = ml_client.get_model_info(model_id)
            except Exception as ex:
                pytest.fail(f"Exception occurred when getting model info: {ex}")
            else:
                assert ml_model_status.get("model_format") == "TORCH_SCRIPT"
                assert ml_model_status.get("algorithm") == "TEXT_EMBEDDING"

            if task_id:
                ml_task_status = None
                try:
                    ml_task_status = ml_client.get_task_info(
                        task_id, wait_until_task_done=True
                    )
                    print("State:", ml_task_status.get("state"))
                except Exception as ex:  # noqa: E722
                    pytest.fail(f"Exception occurred when getting task info: {ex}")
                else:
                    assert ml_task_status.get("task_type") == "DEPLOY_MODEL"
                    assert ml_task_status.get("state") != "FAILED"

                # This is test is being flaky. Sometimes the test is passing and sometimes showing 500 error
                # due to memory circuit breaker.
                # Todo: We need to revisit this test.
                try:
                    sentences = ["First test sentence", "Second test sentence"]
                    embedding_result = ml_client.generate_embedding(model_id, sentences)
                    print(embedding_result)
                except Exception as ex:  # noqa: E722
                    pytest.fail(f"Exception occurred when generating embedding: {ex}")
                else:
                    assert len(embedding_result.get("inference_results")) == 2

                try:
                    text_docs = ["First test sentence", "Second test sentence"]
                    return_number = True
                    target_response = ["sentence_embedding"]
                    algorithm_name = "text_embedding"
                    prediction_result = ml_client.predict(
                        algorithm_name=algorithm_name,
                        model_id=model_id,
                        predict_object={
                            "text_docs": text_docs,
                            "return_number": return_number,
                            "target_response": target_response,
                        },
                    )
                    prediction_result_2 = ml_client.generate_model_inference(
                        model_id=model_id,
                        predict_object={
                            "text_docs": text_docs,
                            "return_number": return_number,
                            "target_response": target_response,
                        },
                    )
                    inference_results = prediction_result.get("inference_results")
                    output_1 = inference_results[0].get("output")
                    output_2 = inference_results[1].get("output")
                except Exception as ex:  # noqa: E722
                    pytest.fail(f"Exception occurred when predicting: {ex}")
                else:
                    assert prediction_result == prediction_result_2
                    assert output_1[0].get("name") == "sentence_embedding"
                    assert output_1[0].get("data_type") == "FLOAT32"
                    assert output_1[0].get("shape")[0] == 384
                    assert isinstance(output_1[0].get("data"), list)
                    assert output_2[0].get("name") == "sentence_embedding"
                    assert output_2[0].get("data_type") == "FLOAT32"
                    assert output_2[0].get("shape")[0] == 384
                    assert isinstance(output_2[0].get("data"), list)

                try:
                    delete_task_obj = ml_client.delete_task(task_id)
                except Exception as ex:  # noqa: E722
                    pytest.fail(f"Exception occurred when deleting task: {ex}")
                else:
                    assert delete_task_obj.get("result") == "deleted"

                try:
                    ml_client.unload_model(model_id)
                    ml_model_status = ml_client.get_model_info(model_id)
                except Exception as ex:  # noqa: E722
                    pytest.fail(
                        f"Exception occurred when pretrained model undeployment : {ex}"
                    )
                else:
                    assert ml_model_status.get("model_state") != "UNDEPLOY_FAILED"

                try:
                    delete_model_obj = ml_client.delete_model(model_id)
                    assert delete_model_obj.get("result") == "deleted"
                except Exception as ex:  # noqa: E722
                    pytest.fail(f"Exception occurred when deleting model: {ex}")


def test_integration_model_train_register_full_cycle():
    # first training the model with small epoch
    test_model.train(
        read_path=TESTDATA_SYNTHETIC_QUERY_ZIP,
        output_model_name=MODEL_FILE_PT_NAME,
        zip_file_name=MODEL_FILE_ZIP_NAME,
        num_epochs=1,
        overwrite=True,
        verbose=True,
    )
    # second generating the config file to create metadoc of the model in opensearch.
    test_model.make_model_config_json()
    model_file_exists = exists(MODEL_PATH)
    model_config_file_exists = exists(MODEL_CONFIG_FILE_PATH)
    assert model_file_exists == True
    assert model_config_file_exists == True
    if model_file_exists and model_config_file_exists:
        model_id = ""
        task_id = ""

        # Testing deploy_model = True for codecov/patch
        try:
            ml_client.register_model(
                model_path=MODEL_PATH,
                model_config_path=MODEL_CONFIG_FILE_PATH,
                deploy_model=True,
                isVerbose=True,
            )
        except Exception as ex:
            pytest.fail(f"Exception occurred during first model registration: {ex}")

        try:
            model_id = ml_client.register_model(
                model_path=MODEL_PATH,
                model_config_path=MODEL_CONFIG_FILE_PATH,
                deploy_model=False,
                isVerbose=True,
            )
            print("Model_id:", model_id)
        except Exception as ex:  # noqa: E722
            pytest.fail(f"Exception occurred during second model registration: {ex}")

        if model_id:
            try:
                ml_load_status = ml_client.deploy_model(
                    model_id, wait_until_deployed=False
                )
            except Exception as ex:
                pytest.fail(f"Exception occurred during model deployment: {ex}")
            else:
                task_id = ml_load_status.get("task_id")
                assert task_id != "" or task_id is not None

                ml_model_status = ml_client.get_model_info(model_id)
                assert ml_model_status.get("model_state") != "DEPLOY_FAILED"

            try:
                ml_model_status = ml_client.get_model_info(model_id)
            except Exception as ex:
                pytest.fail(f"Exception occurred when getting model info: {ex}")
            else:
                assert ml_model_status.get("model_format") == "TORCH_SCRIPT"
                assert ml_model_status.get("algorithm") == "TEXT_EMBEDDING"

            if task_id:
                ml_task_status = None
                try:
                    ml_task_status = ml_client.get_task_info(
                        task_id, wait_until_task_done=True
                    )
                except Exception as ex:  # noqa: E722
                    pytest.fail(f"Exception occurred in pulling task info: {ex}")
                else:
                    assert ml_task_status.get("task_type") == "DEPLOY_MODEL"
                    print("State:", ml_task_status.get("state"))
                    assert ml_task_status.get("state") != "FAILED"

                # This is test is being flaky. Sometimes the test is passing and sometimes showing 500 error
                # due to memory circuit breaker.
                # Todo: We need to revisit this test.
                try:
                    sentences = ["First test sentence", "Second test sentence"]
                    embedding_result = ml_client.generate_embedding(model_id, sentences)
                    print(embedding_result)
                except Exception as ex:
                    pytest.fail(
                        f"Exception occurred when generating sentence embedding: {ex}"
                    )
                else:
                    assert len(embedding_result.get("inference_results")) == 2

                try:
                    text_docs = ["First test sentence", "Second test sentence"]
                    return_number = True
                    target_response = ["sentence_embedding"]
                    algorithm_name = "text_embedding"
                    prediction_result = ml_client.predict(
                        algorithm_name=algorithm_name,
                        model_id=model_id,
                        predict_object={
                            "text_docs": text_docs,
                            "return_number": return_number,
                            "target_response": target_response,
                        },
                    )
                    prediction_result_2 = ml_client.generate_model_inference(
                        model_id=model_id,
                        predict_object={
                            "text_docs": text_docs,
                            "return_number": return_number,
                            "target_response": target_response,
                        },
                    )
                    inference_results = prediction_result.get("inference_results")
                    output_1 = inference_results[0].get("output")
                    output_2 = inference_results[1].get("output")
                except Exception as ex:  # noqa: E722
                    pytest.fail(f"Exception occurred when predicting: {ex}")
                else:
                    assert prediction_result == prediction_result_2
                    assert output_1[0].get("name") == "sentence_embedding"
                    assert output_1[0].get("data_type") == "FLOAT32"
                    assert output_1[0].get("shape")[0] == 384
                    assert isinstance(output_1[0].get("data"), list)
                    assert output_2[0].get("name") == "sentence_embedding"
                    assert output_2[0].get("data_type") == "FLOAT32"
                    assert output_2[0].get("shape")[0] == 384
                    assert isinstance(output_2[0].get("data"), list)

                try:
                    delete_task_obj = ml_client.delete_task(task_id)
                except Exception as ex:
                    pytest.fail(f"Exception occurred when deleting task: {ex}")
                else:
                    assert delete_task_obj.get("result") == "deleted"

                try:
                    ml_client.undeploy_model(model_id)
                    ml_model_status = ml_client.get_model_info(model_id)
                except Exception as ex:
                    pytest.fail(f"Exception occurred during model undeployment : {ex}")
                else:
                    assert ml_model_status.get("model_state") != "UNDEPLOY_FAILED"

                try:
                    delete_model_obj = ml_client.delete_model(model_id)
                except Exception as ex:
                    pytest.fail(f"Exception occurred during model deletion : {ex}")
                else:
                    assert delete_model_obj.get("result") == "deleted"


def test_search():
    # Search task cases
    try:
        search_task_obj = ml_client.search_task(
            input_json='{"query": {"match_all": {}},"size": 1}'
        )
    except Exception as ex:
        pytest.fail(f"Raised Exception in searching task. Exception info: {ex}")
    else:
        assert search_task_obj["hits"]["hits"] != []

    try:
        search_task_obj = ml_client.search_task(
            input_json={"query": {"match_all": {}}, "size": 1}
        )
    except Exception as ex:
        pytest.fail(f"Raised Exception in searching task. Exception info: {ex}")
    else:
        assert search_task_obj["hits"]["hits"] != []

    try:
        search_task_obj = ml_client.search_task(input_json=15)
    except Exception as ex:
        pytest.fail(f"Raised Exception in searching task. Exception info: {ex}")
    else:
        assert search_task_obj == "Invalid JSON object passed as argument."

    try:
        search_task_obj = ml_client.search_task(input_json="15")
    except Exception as ex:
        pytest.fail(f"Raised Exception in searching task. Exception info: {ex}")
    else:
        assert search_task_obj == "Invalid JSON object passed as argument."

    try:
        search_task_obj = ml_client.search_task(
            input_json='{"query": {"match_all": {}},size: 1}'
        )
    except Exception as ex:
        pytest.fail(f"Raised Exception in searching task. Exception info: {ex}")
    else:
        assert search_task_obj == "Invalid JSON string passed as argument."

    # Search model cases
    try:
        search_model_obj = ml_client.search_model(
            input_json='{"query": {"match_all": {}},"size": 1}'
        )
    except Exception as ex:
        pytest.fail(f"Raised Exception in searching model. Exception info: {ex}")
    else:
        assert search_model_obj["hits"]["hits"] != []

    try:
        search_model_obj = ml_client.search_model(
            input_json={"query": {"match_all": {}}, "size": 1}
        )
    except Exception as ex:
        pytest.fail(f"Raised Exception in searching model. Exception info: {ex}")
    else:
        assert search_model_obj["hits"]["hits"] != []

    try:
        search_model_obj = ml_client.search_model(input_json=15)
    except Exception as ex:
        pytest.fail(f"Raised Exception in searching model. Exception info: {ex}")
    else:
        assert search_model_obj == "Invalid JSON object passed as argument."

    try:
        search_model_obj = ml_client.search_model(input_json="15")
    except Exception as ex:
        pytest.fail(f"Raised Exception in searching model. Exception info: {ex}")
    else:
        assert search_model_obj == "Invalid JSON object passed as argument."

    try:
        search_model_obj = ml_client.search_model(
            input_json='{"query": {"match_all": {}},size: 1}'
        )
    except Exception as ex:
        pytest.fail(f"Raised Exception in searching model. Exception info: {ex}")
    else:
        assert search_model_obj == "Invalid JSON string passed as argument."


# Model Profile Tests. These tests will need some model train/predict run data. Hence, need
#  to be run at the end after the training/prediction tests are done.


def test_get_profile():
    res = ml_client.get_profile()
    assert isinstance(res, dict)
    assert "nodes" in res
    test_model_id = None
    test_task_id = None
    for node_id, val in res["nodes"].items():
        if test_model_id is None and "models" in val:
            for model_id, model_val in val["models"].items():
                test_model_id = {"node_id": node_id, "model_id": model_id}
                break
        if test_task_id is None and "tasks" in val:
            for task_id, task_val in val["tasks"].items():
                test_task_id = {"node_id": node_id, "task_id": task_id}
                break

    res = ml_client.get_profile(profile_type="model")
    assert isinstance(res, dict)
    assert "nodes" in res
    for node_id, node_val in res["nodes"].items():
        assert "models" in node_val

    res = ml_client.get_profile(profile_type="model", ids=[test_model_id["model_id"]])
    assert isinstance(res, dict)
    assert "nodes" in res
    assert test_model_id["model_id"] in res["nodes"][test_model_id["node_id"]]["models"]

    res = ml_client.get_profile(profile_type="model", ids=["randomid1", "random_id2"])
    assert isinstance(res, dict)
    assert len(res) == 0

    res = ml_client.get_profile(profile_type="task")
    assert isinstance(res, dict)
    if len(res) > 0:
        assert "nodes" in res
        for node_id, node_val in res["nodes"].items():
            assert "tasks" in node_val

    res = ml_client.get_profile(profile_type="task", ids=["random1", "random2"])
    assert isinstance(res, dict)
    assert len(res) == 0

    with pytest.raises(ValueError):
        ml_client.get_profile(profile_type="test")

    with pytest.raises(ValueError):
        ml_client.get_profile(profile_type="model", ids=1)

    with pytest.raises(ValueError):
        ml_client.get_profile(profile_type="model", request_body=10)

    with pytest.raises(ValueError):
        ml_client.get_profile(profile_type="task", ids=1)

    with pytest.raises(ValueError):
        ml_client.get_profile(profile_type="task", request_body=10)
