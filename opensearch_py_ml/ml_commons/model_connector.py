# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from opensearchpy import OpenSearch

from opensearch_py_ml.ml_commons.ml_common_utils import ML_BASE_URI


class Connector:
    def __init__(self, os_client: OpenSearch):
        self.client = os_client

    def create_standalone_connector(self, payload: dict):
        if not isinstance(payload, dict):
            raise ValueError("payload needs to be a dictionary")

        return self.client.transport.perform_request(
            method="POST", url=f"{ML_BASE_URI}/connectors/_create", body=payload
        )


    def list_connectors(self):
        search_query = {"query": {"match_all": {}}}
        return self.search_connectors(search_query)

    def search_connectors(self, search_query: dict):
        if not isinstance(search_query, dict):
            raise ValueError("search_query needs to be a dictionary")

        return self.client.transport.perform_request(
            method="POST", url=f"{ML_BASE_URI}/connectors/_search", body=search_query
        )

    def get_connector(self, connector_id: str):
        if not isinstance(connector_id, str):
            raise ValueError("connector_id needs to be a string")

        return self.client.transport.perform_request(
            method="GET", url=f"{ML_BASE_URI}/connectors/{connector_id}"
        )

    def delete_connector(self, connector_id: str):
        if not isinstance(connector_id, str):
            raise ValueError("connector_id needs to be a string")

        return self.client.transport.perform_request(
            method="DELETE", url=f"{ML_BASE_URI}/connectors/{connector_id}"
        )
