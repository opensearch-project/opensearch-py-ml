# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import pytest
import time
from unittest import TestCase
from opensearch_py_ml.ml_commons.model_access_control import ModelAccessControl
from tests import OPENSEARCH_TEST_CLIENT
from opensearchpy.exceptions import RequestError

OPENSEARCH_VERSION = OPENSEARCH_TEST_CLIENT.info()['version']['number']
@pytest.fixture
def client():
    return ModelAccessControl(OPENSEARCH_TEST_CLIENT)

@pytest.fixture
def test_model_group(client):
    model_group_name = "__test__model_group_1"
    client.delete_model_group(model_group_name=model_group_name)
    # time.sleep(0.5)
    client.register_model_group(
        name=model_group_name,
        description="test model group for opensearch-py-ml test cases",
    )
    yield model_group_name

    client.delete_model_group(model_group_name=model_group_name)


def test_register_model_group(client):
    
    model_group_name1 = "__test__model_group_A"
    # import pdb;pdb.set_trace()
    try:
        _ = client.delete_model_group(model_group_name=model_group_name1)
        time.sleep(2)
        res = client.register_model_group(name=model_group_name1)
        assert isinstance(res, dict)
        assert "model_group_id" in res
        assert "status" in res
        assert res['status'] == "CREATED"
    except Exception as ex:
        assert False,f"Failed to register model group due to {ex}"
    
    model_group_name2 = "__test__model_group_B"
    
    try:
        _ = client.delete_model_group(model_group_name=model_group_name2)
        time.sleep(2)
        res = client.register_model_group(
            name=model_group_name2,
            description="test",
            access_mode="restricted",
            backend_roles=["admin"],
            )
        assert "model_group_id" in res
        assert "status" in res
        assert res['status'] == "CREATED"
    except Exception as ex:
        assert False,f"Failed to register restricted model group due to {ex}"
    
    model_group_name3 = "__test__model_group_C"
    with pytest.raises(RequestError) as exec_info:
        _ = client.delete_model_group(model_group_name=model_group_name3)
        time.sleep(2)
        res = client.register_model_group(
            name=model_group_name3,
            description="test",
            access_mode="restricted",
            add_all_backend_roles=True
            )
    assert exec_info.value.status_code == 400
    assert exec_info.match("Admin users cannot add all backend roles to a model group")
    
    with pytest.raises(RequestError) as exec_info:
        client.register_model_group(name=model_group_name2)
    assert exec_info.value.status_code == 400
    assert exec_info.match("The name you provided is already being used by a model group")
    
    for each in "ABC":
        client.delete_model_group(model_group_name=f"__test__model_group_{each}")


def test_update_model_group():
    import os
    env_data = os.environ
    print("Environ data = ", env_data)
    assert False, "!@#"

@pytest.mark.skipif(OPENSEARCH_VERSION >= "2.8.0")
def test_delete_model_group(client):
    assert Fail

def test_search_model_group(client):
    pass


def test_search_model_group_by_name(client):
    pass
