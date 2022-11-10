# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from opensearchpy import OpenSearch

from opensearch_py_ml.ml_commons_integration.load.ml_common_load_client import (
    MLCommonLoadClient,
)
from opensearch_py_ml.ml_commons_integration.upload.ml_common_model_uploader import (
    MLCommonModelUploader,
)


class MLCommonClient:
    """
    A client that communicates to the ml-common plugin for OpenSearch. This client allows for uploading of trained
    machine learning models to an OpenSearch index.
    """

    def __init__(self, os_client: OpenSearch):
        self._client = os_client
        self._model_uploader = MLCommonModelUploader(os_client)
        self._load_client = MLCommonLoadClient(os_client)

    def upload_model(
        self,
        model_path: str,
        model_config_path: str,
        isVerbose: bool = False,
    ) -> None:
        """
        This method uploads model into opensearch cluster using ml-common plugin's api.

        first this method creates a model id to store model metadata and then breaks the model zip file into
        multiple chunks and then upload chunks into opensearch cluster.

        Parameters
        ----------
        model_path: string
                     path of the zip file of the model
        model_config_path: string
                     filepath of the model metadata. A json file of model metadata is expected
        isVerbose: boolean, default False
                     if isVerbose is true method will print more messages.

        Returns
        -------
        None
            Doesn't return anything.

        """

        self._model_uploader.upload_model(model_path, model_config_path, isVerbose)

    def load_model(self, model_name: str, version_number: int):  # type: ignore
        return self._load_client.load_model(model_name, version_number)
