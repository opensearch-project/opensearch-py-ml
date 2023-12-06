# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from opensearchpy import OpenSearch

from opensearch_py_ml.ml_commons.ml_common_utils import ML_BASE_URI


class ModelProfile:
    API_ENDPOINT = "profile"
    
    def __init__(self, os_client: OpenSearch):
        self.client = os_client
    
    def get_profile(self, payload: dict):
        if not isinstance(payload, dict):
            raise ValueError("payload needs to be a dictionary")
        return self.client.transport.perform_request(
            method="GET", url=f"{ML_BASE_URI}/{self.API_ENDPOINT}", body=payload
        )
    
    def get_models_profile(self, payload: dict):
        if not isinstance(payload, dict):
            raise ValueError("payload needs to be a dictionary")
        return self.client.transport.perform_request(
            method="GET", url=f"{ML_BASE_URI}/{self.API_ENDPOINT}/models", body=payload
        )
    
    def get_tasks_profile(self, payload: dict):
        if not isinstance(payload, dict):
            raise ValueError("payload needs to be a dictionary")
        return self.client.transport.perform_request(
            method="GET", url=f"{ML_BASE_URI}/{self.API_ENDPOINT}/tasks", body=payload
        )