from opensearch_py_ml.ml_commons.connectors import Connector
from opensearchpy.exceptions import NotFoundError, RequestError
from tests import OPENSEARCH_TEST_CLIENT
from packaging.version import parse as parse_version
import os
import pytest


OPENSEARCH_VERSION = parse_version(os.environ.get("OPENSEARCH_VERSION", "2.11.0"))
CONNECTOR_MIN_VERSION = parse_version("2.11.0")


@pytest.fixture
def client():
    return Connector(OPENSEARCH_TEST_CLIENT)


def _safe_delete_connector(client, connector_id):
    try:
        client.delete_connector(connector_id=connector_id)
    except NotFoundError:
        pass


@pytest.fixture
def test_connector(client: Connector):
    payload = {
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
    
    res = client.create_standalone_connector(payload)
    connector_id = res['connector_id']
    yield connector_id
    
    _safe_delete_connector(client, connector_id)



@pytest.mark.skipif(
    OPENSEARCH_VERSION < CONNECTOR_MIN_VERSION,
    reason="Connectors are supported in OpenSearch 2.9.0 and above",
)
def test_create_standalone_connector():
    pass


@pytest.mark.skipif(
    OPENSEARCH_VERSION < CONNECTOR_MIN_VERSION,
    reason="Connectors are supported in OpenSearch 2.9.0 and above",
)
def test_create_internal_connector():
    pass


@pytest.mark.skipif(
    OPENSEARCH_VERSION < CONNECTOR_MIN_VERSION,
    reason="Connectors are supported in OpenSearch 2.9.0 and above",
)
def test_list_connector():
    pass


@pytest.mark.skipif(
    OPENSEARCH_VERSION < CONNECTOR_MIN_VERSION,
    reason="Connectors are supported in OpenSearch 2.9.0 and above",
)
def test_search_connector():
    pass


@pytest.mark.skipif(
    OPENSEARCH_VERSION < CONNECTOR_MIN_VERSION,
    reason="Connectors are supported in OpenSearch 2.9.0 and above",
)
def test_get_connector():
    pass


@pytest.mark.skipif(
    OPENSEARCH_VERSION < CONNECTOR_MIN_VERSION,
    reason="Connectors are supported in OpenSearch 2.9.0 and above",
)
def test_delete_connector():
    pass