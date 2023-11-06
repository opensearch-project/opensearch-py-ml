# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import os
import time
from unittest.mock import patch

import pytest
from opensearchpy.exceptions import NotFoundError, RequestError
from packaging.version import parse as parse_version

from opensearch_py_ml.ml_commons.model_access_control import ModelAccessControl
from tests import OPENSEARCH_TEST_CLIENT

OPENSEARCH_VERSION = parse_version(os.environ.get("OPENSEARCH_VERSION", "2.7.0"))
MAC_MIN_VERSION = parse_version("2.8.0")
MAC_UPDATE_MIN_VERSION = parse_version("2.11.0")


@pytest.fixture
def client():
    return ModelAccessControl(OPENSEARCH_TEST_CLIENT)


@pytest.fixture
def test_model_group(client):
    model_group_name = "__test__model_group_1"
    client.delete_model_group_by_name(model_group_name=model_group_name)
    time.sleep(2)
    client.register_model_group(
        name=model_group_name,
        description="test model group for opensearch-py-ml test cases",
    )
    yield model_group_name

    client.delete_model_group_by_name(model_group_name=model_group_name)


@pytest.fixture
def test_model_group2(client):
    model_group_name = "__test__model_group_2"
    client.delete_model_group_by_name(model_group_name=model_group_name)
    time.sleep(2)
    client.register_model_group(
        name=model_group_name,
        description="test model group for opensearch-py-ml test cases",
    )
    yield model_group_name

    client.delete_model_group_by_name(model_group_name=model_group_name)


@pytest.mark.skipif(
    OPENSEARCH_VERSION < MAC_MIN_VERSION,
    reason="Model groups are supported in OpenSearch 2.8.0 and above",
)
def test_register_model_group(client):
    model_group_name1 = "__test__model_group_A"
    # import pdb;pdb.set_trace()
    try:
        _ = client.delete_model_group_by_name(model_group_name=model_group_name1)
        time.sleep(2)
        res = client.register_model_group(name=model_group_name1)
        assert isinstance(res, dict)
        assert "model_group_id" in res
        assert "status" in res
        assert res["status"] == "CREATED"
    except Exception as ex:
        assert False, f"Failed to register model group due to {ex}"

    model_group_name2 = "__test__model_group_B"

    try:
        _ = client.delete_model_group_by_name(model_group_name=model_group_name2)
        time.sleep(2)
        res = client.register_model_group(
            name=model_group_name2,
            description="test",
            access_mode="restricted",
            backend_roles=["admin"],
        )
        assert "model_group_id" in res
        assert "status" in res
        assert res["status"] == "CREATED"
    except Exception as ex:
        assert False, f"Failed to register restricted model group due to {ex}"

    model_group_name3 = "__test__model_group_C"
    with pytest.raises(RequestError) as exec_info:
        _ = client.delete_model_group_by_name(model_group_name=model_group_name3)
        time.sleep(2)
        res = client.register_model_group(
            name=model_group_name3,
            description="test",
            access_mode="restricted",
            add_all_backend_roles=True,
        )
    assert exec_info.value.status_code == 400
    assert exec_info.match("Admin users cannot add all backend roles to a model group")

    with pytest.raises(RequestError) as exec_info:
        client.register_model_group(name=model_group_name2)
    assert exec_info.value.status_code == 400
    assert exec_info.match(
        "The name you provided is already being used by a model group"
    )


@pytest.mark.skipif(
    OPENSEARCH_VERSION < MAC_MIN_VERSION,
    reason="Model groups are supported in OpenSearch 2.8.0 and above",
)
def test_get_model_group_id_by_name(client, test_model_group):
    model_group_id = client.get_model_group_id_by_name(test_model_group)
    assert model_group_id is not None

    model_group_id = client.get_model_group_id_by_name("test-unknown")
    assert model_group_id is None

    # Mock NotFoundError as it only happens when index isn't created
    with patch.object(client, "search_model_group_by_name", side_effect=NotFoundError):
        model_group_id = client.get_model_group_id_by_name(test_model_group)
        assert model_group_id is None


@pytest.mark.skipif(
    OPENSEARCH_VERSION < MAC_UPDATE_MIN_VERSION,
    reason="Model groups updates are supported in OpenSearch 2.11.0 and above",
)
def test_update_model_group(client, test_model_group):
    # update model group name and description
    update_query = {
        "description": "updated description",
    }
    try:
        model_group_id = client.get_model_group_id_by_name(test_model_group)
        if model_group_id is None:
            raise Exception(f"No model group found with the name: {test_model_group}")
        res = client.update_model_group(update_query, model_group_id=model_group_id)
        assert isinstance(res, dict)
        assert "status" in res
        assert res["status"] == "Updated"
    except Exception as ex:
        assert False, f"Failed to search model group due to unhandled error: {ex}"


@pytest.mark.skipif(
    OPENSEARCH_VERSION < MAC_MIN_VERSION,
    reason="Model groups are supported in OpenSearch 2.8.0 and above",
)
def test_search_model_group(client, test_model_group):
    query1 = {"query": {"match": {"name": test_model_group}}, "size": 1}
    try:
        res = client.search_model_group(query1)
        assert isinstance(res, dict)
        assert "hits" in res and "hits" in res["hits"]
        assert len(res["hits"]["hits"]) == 1
        assert "_source" in res["hits"]["hits"][0]
        assert "name" in res["hits"]["hits"][0]["_source"]
        assert test_model_group == res["hits"]["hits"][0]["_source"]["name"]
    except Exception as ex:
        assert False, f"Failed to search model group due to unhandled error: {ex}"

    query2 = {"query": {"match": {"name": "test-unknown"}}, "size": 1}
    try:
        res = client.search_model_group(query2)
        assert isinstance(res, dict)
        assert "hits" in res and "hits" in res["hits"]
        assert len(res["hits"]["hits"]) == 0
    except Exception as ex:
        assert False, f"Failed to search model group due to unhandled error: {ex}"


@pytest.mark.skipif(
    OPENSEARCH_VERSION < MAC_MIN_VERSION,
    reason="Model groups are supported in OpenSearch 2.8.0 and above",
)
def test_search_model_group_by_name(client, test_model_group):
    try:
        res = client.search_model_group_by_name(model_group_name=test_model_group)
        assert isinstance(res, dict)
        assert "hits" in res and "hits" in res["hits"]
        assert len(res["hits"]["hits"]) == 1
        assert "_source" in res["hits"]["hits"][0]
        assert len(res["hits"]["hits"][0]["_source"]) > 1
        assert "name" in res["hits"]["hits"][0]["_source"]
        assert test_model_group == res["hits"]["hits"][0]["_source"]["name"]
    except Exception as ex:
        assert False, f"Failed to search model group due to unhandled error: {ex}"

    try:
        res = client.search_model_group_by_name(
            model_group_name=test_model_group, _source="name"
        )
        assert isinstance(res, dict)
        assert "hits" in res and "hits" in res["hits"]
        assert len(res["hits"]["hits"]) == 1
        assert "_source" in res["hits"]["hits"][0]
        assert len(res["hits"]["hits"][0]["_source"]) == 1
        assert "name" in res["hits"]["hits"][0]["_source"]
    except Exception as ex:
        assert False, f"Failed to search model group due to unhandled error: {ex}"

    try:
        res = client.search_model_group_by_name(model_group_name="test-unknown")
        assert isinstance(res, dict)
        assert "hits" in res and "hits" in res["hits"]
        assert len(res["hits"]["hits"]) == 0
    except Exception as ex:
        assert False, f"Failed to search model group due to unhandled error: {ex}"


@pytest.mark.skipif(
    OPENSEARCH_VERSION < MAC_MIN_VERSION,
    reason="Model groups are supported in OpenSearch 2.8.0 and above",
)
def test_delete_model_group(client, test_model_group):
    # create a test model group

    for each in "AB":
        model_group_name = f"__test__model_group_{each}"
        model_group_id = client.get_model_group_id_by_name(model_group_name)
        if model_group_id is None:
            continue
        res = client.delete_model_group(model_group_id=model_group_id)
        assert res is None or isinstance(res, dict)
        if isinstance(res, dict):
            assert "result" in res
            assert res["result"] in ["not_found", "deleted"]

    res = client.delete_model_group(model_group_id="test-unknown")
    assert isinstance(res, dict)
    assert "result" in res
    assert res["result"] == "not_found"


@pytest.mark.skipif(
    OPENSEARCH_VERSION < MAC_MIN_VERSION,
    reason="Model groups are supported in OpenSearch 2.8.0 and above",
)
def test_delete_model_group_by_name(client, test_model_group2):
    res = client.delete_model_group_by_name(model_group_name="test-unknown")
    assert res is None

    res = client.delete_model_group_by_name(model_group_name=test_model_group2)
    assert isinstance(res, dict)
    assert "result" in res
    assert res["result"] == "deleted"
