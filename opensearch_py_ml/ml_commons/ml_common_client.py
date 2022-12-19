# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from opensearchpy import OpenSearch

from opensearch_py_ml.ml_commons.ml_common_utils import ML_BASE_URI
from opensearch_py_ml.ml_commons.model_uploader import ModelUploader


class MLCommonClient:
    """
    A client that communicates to the ml-common plugin for OpenSearch. This client allows for uploading of trained
    machine learning models to an OpenSearch index.
    """

    def __init__(self, os_client: OpenSearch):
        self._client = os_client
        self._model_uploader = ModelUploader(os_client)

    def upload_model(
        self,
        model_path: str,
        model_config_path: str,
        isVerbose: bool = False,
    ) -> str:
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
        model_id: string
            returns the model_id so that we can use this for further operation.

        """

        return self._model_uploader._upload_model(
            model_path, model_config_path, isVerbose
        )

    def load_model(self, model_id: str):  # type: ignore
        """
        This method loads model into opensearch cluster using ml-common plugin's load model api.

        Parameters
        ----------
        model_id: string
                     unique id of the model
        isVerbose: boolean, default False
                     if isVerbose is true method will print more messages.

        Returns
        -------
        object
            returns a json object, with task_id and status key.

        """
        MODEL_LOAD_API_ENDPOINT = f"models/{model_id}/_load"
        API_URL = f"{ML_BASE_URI}/{MODEL_LOAD_API_ENDPOINT}"

        return self._client.transport.perform_request(
            method="POST",
            url=API_URL,
        )
