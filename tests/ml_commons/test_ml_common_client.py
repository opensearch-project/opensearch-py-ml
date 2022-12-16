# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import urllib.request
from os.path import exists

from opensearchpy import OpenSearch

from opensearch_py_ml.ml_commons import MLCommonClient
from opensearch_py_ml.ml_commons.model_uploader import ModelUploader
from tests import ML_CONFIG_FILE_PATH, ML_FILE_PATH, ML_FILE_URL, OPENSEARCH_TEST_CLIENT

ml_client = MLCommonClient(OPENSEARCH_TEST_CLIENT)


def test_init():
    assert type(ml_client._client) == OpenSearch
    assert type(ml_client._model_uploader) == ModelUploader


def test_upload_and_load_model():
    file_exists = exists(ML_FILE_PATH)
    if not file_exists:
        urllib.request.urlretrieve(ML_FILE_URL, ML_FILE_PATH)

    raised = False
    model_id = ""
    try:
        model_id = ml_client.upload_model(
            ML_FILE_PATH, ML_CONFIG_FILE_PATH, isVerbose=True
        )
        print("Model_id:", model_id)
    except:  # noqa: E722
        raised = True
    assert raised == False, "Raised Exception during model upload"

    if model_id:
        raised = False
        try:
            ml_load_status = ml_client.load_model(model_id)
            assert ml_load_status.get("status") == "CREATED"
        except:  # noqa: E722
            raised = True
        assert raised == False, "Raised Exception in loading model"
