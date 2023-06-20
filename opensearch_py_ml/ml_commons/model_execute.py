# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json

from opensearchpy import OpenSearch

from opensearch_py_ml.ml_commons.ml_common_utils import ML_BASE_URI


class ModelExecute:
    """
    Class for executing algorithms using ML Commons execute API.
    """

    API_ENDPOINT = "models/_execute"

    def __init__(self, os_client: OpenSearch):
        self._client = os_client

    def _execute(self, algorithm_name: str, input_json: dict) -> dict:
        """
        This method executes ML algorithms that can be only executed directly (i.e. do not support train and
        predict APIs), like anomaly localization and metrics correlation.
        The input json must be a dictionary or a deserializable Python object.
        The algorithm has to be supported by ML Commons.
        Refer to https://opensearch.org/docs/2.7/ml-commons-plugin/api/#execute

        :param algorithm_name: Name of the algorithm
        :type algorithm_name: string
        :param input_json: Dictionary of parameters or a deserializable string, byte, or bytearray
        :type input_json: dict
        :return: returns the API response
        :rtype: dict
        """

        if not isinstance(input_json, dict):
            input_json = json.loads(input_json)

        return self._client.transport.perform_request(
            method="POST",
            url=f"{ML_BASE_URI}/_execute/{algorithm_name}",
            body=input_json,
        )
