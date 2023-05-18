# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

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
        predict APIs), like anomaly localization and metrics correlation. The algorithm has to be supported by ML Commons.
        Refer to https://opensearch.org/docs/2.7/ml-commons-plugin/api/#execute

        :param algorithm_name: Name of the algorithm
        :type algorithm_name: string
        :param input_json: Dictionary of parameters
        :type input_json: dict
        :return: returns the API response
        :rtype: dict
        """

        if algorithm_name.lower() not in [
            "anomaly_localization",
            "metrics_correlation",
        ]:
            raise ValueError("Algorithm must be supported by ML Commons")

        if algorithm_name.upper() == "METRICS_CORRELATION":
            if self._validate_json_mcorr(input_json=input_json):
                return self._client.transport.perform_request(
                    method="POST",
                    url=f"{ML_BASE_URI}/_execute/METRICS_CORRELATION",
                    body=input_json,
                )
        elif algorithm_name.lower() == "anomaly_localization":
            if self._validate_json_localization(input_json=input_json):
                return self._client.transport.perform_request(
                    method="POST",
                    url=f"{ML_BASE_URI}/_execute/anomaly_localization",
                    body=input_json,
                )

    def _validate_json_localization(self, input_json: dict) -> bool:
        mandatory_fields = [
            "index_name",
            "attribute_field_names",
            "aggregations",
            "time_field_name",
            "start_time",
            "end_time",
            "min_time_interval",
            "num_outputs",
        ]
        optional_fields = ["filter_query", "anomaly_star"]

        for key in input_json:
            if key not in mandatory_fields and key not in optional_fields:
                raise ValueError(f"Parameter {key} is not supported")

        for mand_key in mandatory_fields:
            if not input_json.get(mand_key):
                raise ValueError(f"{mand_key} can not be empty")

        return True

    def _validate_json_mcorr(self, input_json: dict) -> bool:
        for key in input_json:
            if key != "metrics":
                raise ValueError(f"Parameter {key} is not supported")

        if "metrics" not in input_json:
            raise ValueError("Metrics field is missing")
        if len(input_json["metrics"]) == 0:
            raise ValueError("metrics parameter can't be empty")

        arr = input_json["metrics"]
        num_timesteps = len(arr[0])
        for row in arr:
            if len(row) != num_timesteps:
                raise ValueError(
                    "All metrics need to have an equal amount of time steps"
                )

        if num_timesteps * len(arr) > 10000:
            raise ValueError(
                "The total number of data points must not be higher than 10000"
            )

        return True
