# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from opensearchpy import OpenSearch

from opensearch_py_ml.ml_commons_integration.ml_common_utils import ML_BASE_URI


class MLCommonLoadClient:
    """
    Client for performing model upload tasks to ml-commons plugin for OpenSearch.
    """

    def __init__(self, os_client: OpenSearch):
        self._client = os_client

    def load_model(self, model_name: str, version_num: int):  # type: ignore
        """
        Load a model with name model_name and version number version_num.
        """
        return self._client.transport.perform_request(
            method="POST",
            url=f"{ML_BASE_URI}/custom_model/load",
            body={"name": f'"{model_name}"', "version": version_num},
        )
