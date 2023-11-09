# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from typing import List, Optional

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError

from opensearch_py_ml.ml_commons.ml_common_utils import ML_BASE_URI
from opensearch_py_ml.ml_commons.validators.model_access_control import (
    validate_create_model_group_parameters,
    validate_delete_model_group_parameters,
    validate_search_model_group_parameters,
    validate_update_model_group_parameters,
)


class ModelAccessControl:
    API_ENDPOINT = "model_groups"

    def __init__(self, os_client: OpenSearch):
        self.client = os_client

    def register_model_group(
        self,
        name: str,
        description: Optional[str] = None,
        access_mode: Optional[str] = "private",
        backend_roles: Optional[List[str]] = None,
        add_all_backend_roles: Optional[bool] = False,
    ):
        validate_create_model_group_parameters(
            name, description, access_mode, backend_roles, add_all_backend_roles
        )

        body = {"name": name, "add_all_backend_roles": add_all_backend_roles}
        if description:
            body["description"] = description
        if access_mode:
            body["access_mode"] = access_mode
        if backend_roles:
            body["backend_roles"] = backend_roles

        return self.client.transport.perform_request(
            method="POST", url=f"{ML_BASE_URI}/{self.API_ENDPOINT}/_register", body=body
        )

    def update_model_group(
        self,
        update_query: dict,
        model_group_id: Optional[str] = None,
    ):
        validate_update_model_group_parameters(update_query, model_group_id)
        return self.client.transport.perform_request(
            method="PUT",
            url=f"{ML_BASE_URI}/{self.API_ENDPOINT}/{model_group_id}",
            body=update_query,
        )

    def search_model_group(self, query: dict):
        validate_search_model_group_parameters(query)
        return self.client.transport.perform_request(
            method="GET", url=f"{ML_BASE_URI}/{self.API_ENDPOINT}/_search", body=query
        )

    def search_model_group_by_name(
        self,
        model_group_name: str,
        _source: Optional[List] = None,
        size: Optional[int] = 1,
    ):
        query = {"query": {"match": {"name": model_group_name}}, "size": size}
        if _source:
            query["_source"] = _source
        return self.search_model_group(query)

    def get_model_group_id_by_name(self, model_group_name: str):
        try:
            res = self.search_model_group_by_name(model_group_name)
            if res["hits"]["hits"]:
                return res["hits"]["hits"][0]["_id"]
            else:
                return None
        except NotFoundError:
            return None
        except Exception as ex:
            print(f"Error in get_model_group_id_by_name: {ex}")
            return None

    def delete_model_group(self, model_group_id: str):
        validate_delete_model_group_parameters(model_group_id)
        return self.client.transport.perform_request(
            method="DELETE", url=f"{ML_BASE_URI}/{self.API_ENDPOINT}/{model_group_id}"
        )

    def delete_model_group_by_name(self, model_group_name: str):
        model_group_id = self.get_model_group_id_by_name(model_group_name)
        if model_group_id is None:
            raise NotFoundError(f"Model group {model_group_name} not found")
        return self.delete_model_group(model_group_id=model_group_id)
