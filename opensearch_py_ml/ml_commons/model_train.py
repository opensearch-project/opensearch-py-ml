# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
from typing import Optional

from opensearchpy import OpenSearch

from opensearch_py_ml.ml_commons.ml_common_utils import ML_BASE_URI


class ModelTrain:
    """
    Class for training models using ML Commons train API.
    """

    API_ENDPOINT = "models/_train"

    def __init__(self, os_client: OpenSearch):
        self._client = os_client

    def _train(
        self, algorithm_name: str, input_json: dict, is_async: Optional[bool] = True
    ) -> dict:
        """
        This method trains an ML model
        """

        params = {}
        if not isinstance(input_json, dict):
            input_json = json.loads(input_json)
        if is_async:
            params["async"] = "true"

        return self._client.transport.perform_request(
            method="POST",
            url=f"{ML_BASE_URI}/_train/{algorithm_name}",
            body=input_json,
            params=params,
        )
