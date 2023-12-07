# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from typing import Optional

from opensearchpy import OpenSearch

from opensearch_py_ml.ml_commons.ml_common_utils import ML_BASE_URI


class ModelProfile:
    API_ENDPOINT = "profile"

    def __init__(self, os_client: OpenSearch):
        self.client = os_client

    def _validate_input(self, path_parameter, payload):
        if path_parameter is not None and not isinstance(path_parameter, str):
            raise ValueError("payload needs to be a dictionary or None")

        if payload is not None and not isinstance(payload, dict):
            raise ValueError("path_parameter needs to be a string or None")

    def get_profile(self, payload: Optional[dict] = None):
        if payload is not None and not isinstance(payload, dict):
            raise ValueError("payload needs to be a dictionary or None")
        return self.client.transport.perform_request(
            method="GET", url=f"{ML_BASE_URI}/{self.API_ENDPOINT}", body=payload
        )

    def get_models_profile(
        self, path_parameter: Optional[str] = "", payload: Optional[dict] = None
    ):
        self._validate_input(path_parameter, payload)

        url = f"{ML_BASE_URI}/{self.API_ENDPOINT}/models/{path_parameter if path_parameter else ''}"
        return self.client.transport.perform_request(
            method="GET", url=url, body=payload
        )

    def get_tasks_profile(
        self, path_parameter: Optional[str] = "", payload: Optional[dict] = None
    ):
        self._validate_input(path_parameter, payload)

        url = f"{ML_BASE_URI}/{self.API_ENDPOINT}/tasks/{path_parameter if path_parameter else ''}"
        return self.client.transport.perform_request(
            method="GET", url=url, body=payload
        )
