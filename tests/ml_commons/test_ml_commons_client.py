# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import os
import shutil
from os.path import exists

from opensearchpy import OpenSearch

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
    assert type(ml_client._client) == OpenSearch
    assert type(ml_client._model_uploader) == ModelUploader


def test_DEPRECATED_integration_pretrained_model_upload_unload_delete():
    raised = False
    try:
        model_id = ml_client.upload_pretrained_model(
            model_name=PRETRAINED_MODEL_NAME,
            model_version=PRETRAINED_MODEL_VERSION,
            model_format=PRETRAINED_MODEL_FORMAT,
            load_model=True,
            wait_until_loaded=True,
        )
        ml_model_status = ml_client.get_model_info(model_id)
        assert ml_model_status.get("model_state") != "DEPLOY_FAILED"
    except:  # noqa: E722
        raised = True
    assert (
        raised == False
    ), "Raised Exception during pretrained model registration and deployment"

    if model_id:
        raised = False
        try:
            ml_model_status = ml_client.get_model_info(model_id)
            assert ml_model_status.get("model_format") == "TORCH_SCRIPT"
            assert ml_model_status.get("algorithm") == "TEXT_EMBEDDING"
        except:  # noqa: E722
            raised = True
        assert raised == False, "Raised Exception in getting pretrained model info"

        raised = False
        try:
            ml_client.unload_model(model_id)
            ml_model_status = ml_client.get_model_info(model_id)
            assert ml_model_status.get("model_state") == "UNDEPLOYED"
        except:  # noqa: E722
            raised = True
        assert raised == False, "Raised Exception in pretrained model undeployment"

        raised = False
        try:
            delete_model_obj = ml_client.delete_model(model_id)
            assert delete_model_obj.get("result") == "deleted"
        except:  # noqa: E722
            raised = True
        assert raised == False, "Raised Exception in deleting pretrained model"


def test_integration_pretrained_model_register_undeploy_delete():
    raised = False
    try:
        model_id = ml_client.register_pretrained_model(
            model_name=PRETRAINED_MODEL_NAME,
            model_version=PRETRAINED_MODEL_VERSION,
            model_format=PRETRAINED_MODEL_FORMAT,
            deploy_model=True,
            wait_until_deployed=True,
        )
        ml_model_status = ml_client.get_model_info(model_id)
        assert ml_model_status.get("model_state") != "DEPLOY_FAILED"
    except:  # noqa: E722
        raised = True
    assert (
        raised == False
    ), "Raised Exception during pretrained model registration and deployment"

    if model_id:
        raised = False
        try:
            ml_model_status = ml_client.get_model_info(model_id)
            assert ml_model_status.get("model_format") == "TORCH_SCRIPT"
            assert ml_model_status.get("algorithm") == "TEXT_EMBEDDING"
        except:  # noqa: E722
            raised = True
        assert raised == False, "Raised Exception in getting pretrained model info"

        raised = False
        try:
            ml_client.undeploy_model(model_id)
            ml_model_status = ml_client.get_model_info(model_id)
            assert ml_model_status.get("model_state") == "UNDEPLOYED"
        except:  # noqa: E722
            raised = True
        assert raised == False, "Raised Exception in pretrained model undeployment"

        raised = False
        try:
            delete_model_obj = ml_client.delete_model(model_id)
            assert delete_model_obj.get("result") == "deleted"
        except:  # noqa: E722
            raised = True
        assert raised == False, "Raised Exception in deleting pretrained model"


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
        raised = False
        model_id = ""
        task_id = ""
        try:
            model_id = ml_client.upload_model(
                MODEL_PATH, MODEL_CONFIG_FILE_PATH, load_model=False, isVerbose=True
            )
            print("Model_id:", model_id)
        except:  # noqa: E722
            raised = True
        assert raised == False, "Raised Exception during model registration"

        if model_id:
            raised = False
            try:
                ml_load_status = ml_client.load_model(model_id, wait_until_loaded=False)
                task_id = ml_load_status.get("task_id")
                assert task_id != "" or task_id is not None

                ml_model_status = ml_client.get_model_info(model_id)
                assert ml_model_status.get("model_state") != "DEPLOY_FAILED"
            except:  # noqa: E722
                raised = True
            assert raised == False, "Raised Exception in model deployment"

            raised = False
            try:
                ml_model_status = ml_client.get_model_info(model_id)
                assert ml_model_status.get("model_format") == "TORCH_SCRIPT"
                assert ml_model_status.get("algorithm") == "TEXT_EMBEDDING"
            except:  # noqa: E722
                raised = True
            assert raised == False, "Raised Exception in getting model info"

            if task_id:
                raised = False
                ml_task_status = None
                try:
                    ml_task_status = ml_client.get_task_info(
                        task_id, wait_until_task_done=True
                    )
                    assert ml_task_status.get("task_type") == "DEPLOY_MODEL"
                    print("State:", ml_task_status.get("state"))
                    assert ml_task_status.get("state") != "FAILED"
                except:  # noqa: E722
                    print("Model Task Status:", ml_task_status)
                    raised = True
                assert raised == False, "Raised Exception in pulling task info"
                # This is test is being flaky. Sometimes the test is passing and sometimes showing 500 error
                # due to memory circuit breaker.
                # Todo: We need to revisit this test.
                try:
                    raised = False
                    sentences = ["First test sentence", "Second test sentence"]
                    embedding_result = ml_client.generate_embedding(model_id, sentences)
                    print(embedding_result)
                    assert len(embedding_result.get("inference_results")) == 2
                except:  # noqa: E722
                    raised = True
                assert (
                    raised == False
                ), "Raised Exception in generating sentence embedding"

                try:
                    delete_task_obj = ml_client.delete_task(task_id)
                    assert delete_task_obj.get("result") == "deleted"
                except:  # noqa: E722
                    raised = True
                assert raised == False, "Raised Exception in deleting task"

                try:
                    ml_client.unload_model(model_id)
                    ml_model_status = ml_client.get_model_info(model_id)
                    assert ml_model_status.get("model_state") == "UNDEPLOYED"
                except:  # noqa: E722
                    raised = True
                assert raised == False, "Raised Exception in model undeployment"

                raised = False
                try:
                    delete_model_obj = ml_client.delete_model(model_id)
                    assert delete_model_obj.get("result") == "deleted"
                except:  # noqa: E722
                    raised = True
                assert raised == False, "Raised Exception in deleting model"


def test_integration_model_train_register_full_cycle():
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
        raised = False
        model_id = ""
        task_id = ""
        try:
            model_id = ml_client.register_model(
                MODEL_PATH, MODEL_CONFIG_FILE_PATH, deploy_model=False, isVerbose=True
            )
            print("Model_id:", model_id)
        except:  # noqa: E722
            raised = True
        assert raised == False, "Raised Exception during model registration"

        if model_id:
            raised = False
            try:
                ml_load_status = ml_client.deploy_model(
                    model_id, wait_until_deployed=False
                )
                task_id = ml_load_status.get("task_id")
                assert task_id != "" or task_id is not None

                ml_model_status = ml_client.get_model_info(model_id)
                assert ml_model_status.get("model_state") != "DEPLOY_FAILED"
            except:  # noqa: E722
                raised = True
            assert raised == False, "Raised Exception in model deployment"

            raised = False
            try:
                ml_model_status = ml_client.get_model_info(model_id)
                assert ml_model_status.get("model_format") == "TORCH_SCRIPT"
                assert ml_model_status.get("algorithm") == "TEXT_EMBEDDING"
            except:  # noqa: E722
                raised = True
            assert raised == False, "Raised Exception in getting model info"

            if task_id:
                raised = False
                ml_task_status = None
                try:
                    ml_task_status = ml_client.get_task_info(
                        task_id, wait_until_task_done=True
                    )
                    assert ml_task_status.get("task_type") == "DEPLOY_MODEL"
                    print("State:", ml_task_status.get("state"))
                    assert ml_task_status.get("state") != "FAILED"
                except:  # noqa: E722
                    print("Model Task Status:", ml_task_status)
                    raised = True
                assert raised == False, "Raised Exception in pulling task info"
                # This is test is being flaky. Sometimes the test is passing and sometimes showing 500 error
                # due to memory circuit breaker.
                # Todo: We need to revisit this test.
                try:
                    raised = False
                    sentences = ["First test sentence", "Second test sentence"]
                    embedding_result = ml_client.generate_embedding(model_id, sentences)
                    print(embedding_result)
                    assert len(embedding_result.get("inference_results")) == 2
                except:  # noqa: E722
                    raised = True
                assert (
                    raised == False
                ), "Raised Exception in generating sentence embedding"

                try:
                    delete_task_obj = ml_client.delete_task(task_id)
                    assert delete_task_obj.get("result") == "deleted"
                except:  # noqa: E722
                    raised = True
                assert raised == False, "Raised Exception in deleting task"

                try:
                    ml_client.undeploy_model(model_id)
                    ml_model_status = ml_client.get_model_info(model_id)
                    assert ml_model_status.get("model_state") == "UNDEPLOYED"
                except:  # noqa: E722
                    raised = True
                assert raised == False, "Raised Exception in model undeployment"

                raised = False
                try:
                    delete_model_obj = ml_client.delete_model(model_id)
                    assert delete_model_obj.get("result") == "deleted"
                except:  # noqa: E722
                    raised = True
                assert raised == False, "Raised Exception in deleting model"
