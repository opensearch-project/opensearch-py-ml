# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import os

import pytest
from opensearchpy.exceptions import NotFoundError, RequestError
from packaging.version import parse as parse_version

from opensearch_py_ml.ml_commons.model_connector import Connector
from tests import OPENSEARCH_TEST_CLIENT

OPENSEARCH_VERSION = parse_version(os.environ.get("OPENSEARCH_VERSION", "2.11.0"))
CONNECTOR_MIN_VERSION = parse_version("2.9.0")

@pytest.fixture
def client():
    return Connector(OPENSEARCH_TEST_CLIENT)


def _safe_delete_connector(client, connector_id):
    try:
        client.delete_connector(connector_id=connector_id)
    except NotFoundError:
        pass


@pytest.fixture
def connector_payload():
    return {
        "name": "Test Connector",
        "description": "Connector for testing",
        "version": 1,
        "protocol": "http",
        "parameters": {"endpoint": "api.openai.com", "model": "gpt-3.5-turbo"},
        "credential": {"openAI_key": "..."},
        "actions": [
            {
                "action_type": "predict",
                "method": "POST",
                "url": "https://${parameters.endpoint}/v1/chat/completions",
                "headers": {"Authorization": "Bearer ${credential.openAI_key}"},
                "request_body": '{ "model": "${parameters.model}", "messages": ${parameters.messages} }',
            }
        ],
    }


@pytest.fixture
def test_connector(client: Connector, connector_payload: dict):
    res = client.create_standalone_connector(connector_payload)
    connector_id = res["connector_id"]
    yield connector_id

    _safe_delete_connector(client, connector_id)


@pytest.mark.skipif(
    OPENSEARCH_VERSION < CONNECTOR_MIN_VERSION,
    reason="Connectors are supported in OpenSearch 2.9.0 and above",
)
def test_create_standalone_connector(client: Connector, connector_payload: dict):
    res = client.create_standalone_connector(connector_payload)
    assert "connector_id" in res

    _safe_delete_connector(client, res["connector_id"])

    with pytest.raises(ValueError):
        client.create_standalone_connector("")


@pytest.mark.skipif(
    OPENSEARCH_VERSION < CONNECTOR_MIN_VERSION,
    reason="Connectors are supported in OpenSearch 2.9.0 and above",
)
def test_list_connectors(client, test_connector):
    try:
        res = client.list_connectors()
        assert len(res["hits"]["hits"]) > 0

        # check if test_connector id is in the response
        found = False
        for each in res["hits"]["hits"]:
            if each["_id"] == test_connector:
                found = True
                break
        assert found, "Test connector not found in list connectors response"
    except Exception as ex:
        assert False, f"Failed to list connectors due to {ex}"


@pytest.mark.skipif(
    OPENSEARCH_VERSION < CONNECTOR_MIN_VERSION,
    reason="Connectors are supported in OpenSearch 2.9.0 and above",
)
def test_search_connectors(client, test_connector):
    try:
        query = {"query": {"match": {"name": "Test Connector"}}}
        res = client.search_connectors(query)
        assert len(res["hits"]["hits"]) > 0

        # check if test_connector id is in the response
        found = False
        for each in res["hits"]["hits"]:
            if each["_id"] == test_connector:
                found = True
                break
        assert found, "Test connector not found in search connectors response"
    except Exception as ex:
        assert False, f"Failed to search connectors due to {ex}"


@pytest.mark.skipif(
    OPENSEARCH_VERSION < CONNECTOR_MIN_VERSION,
    reason="Connectors are supported in OpenSearch 2.9.0 and above",
)
def test_get_connector(client, test_connector):
    try:
        res = client.get_connector(connector_id=test_connector)
        assert res["name"] == "Test Connector"
    except Exception as ex:
        assert False, f"Failed to get connector due to {ex}"

    with pytest.raises(ValueError):
        client.get_connector(connector_id=None)

    with pytest.raises(RequestError) as exec_info:
        client.get_connector(connector_id="test-unknown")
    assert exec_info.value.status_code == 400


@pytest.mark.skipif(
    OPENSEARCH_VERSION < CONNECTOR_MIN_VERSION,
    reason="Connectors are supported in OpenSearch 2.9.0 and above",
)
def test_delete_connector(client, test_connector):
    try:
        res = client.delete_connector(connector_id=test_connector)
        assert res["result"] == "deleted"
    except Exception as ex:
        assert False, f"Failed to delete connector due to {ex}"

    try:
        res = client.delete_connector(connector_id="unknown")
        assert res["result"] == "not_found"
    except Exception as ex:
        assert False, f"Failed to delete connector due to {ex}"
