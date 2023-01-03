# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.


from typing import List

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
        :param model_path: path of the zip file of the model
        :type model_path: string
        :param model_config_path: filepath of the model metadata. A json file of model metadata is expected
        :type model_config_path: string
        :param isVerbose: if isVerbose is true method will print more messages. default False
        :type isVerbose: boolean

        Returns
        -------
        :return: returns the model_id so that we can use this for further operation.
        :rtype: string

        """

        return self._model_uploader._upload_model(
            model_path, model_config_path, isVerbose
        )

    def load_model(self, model_id: str):  # type: ignore
        """
        This method loads model into opensearch cluster using ml-common plugin's load model api.

        Parameters
        ----------
        :param model_id: unique id of the model
        :type model_id: string

        Returns
        -------
        :return: returns a json object, with task_id and status key.
        :rtype: object

        """
        MODEL_LOAD_API_ENDPOINT = f"models/{model_id}/_load"
        API_URL = f"{ML_BASE_URI}/{MODEL_LOAD_API_ENDPOINT}"

        return self._client.transport.perform_request(
            method="POST",
            url=API_URL,
        )

    def get_task_info(self, task_id: str):  # type: ignore
        """
        This method return information about a task running into opensearch cluster (using ml commons api)
        when we load a model.

        Parameters
        ----------
        :param task_id: unique id of the task
        :type task_id: string

        Returns
        -------
        :return: returns a json object, with detailed information about the task
        :rtype: object
        """

        MODEL_TASK_API_ENDPOINT = f"tasks/{task_id}"
        API_URL = f"{ML_BASE_URI}/{MODEL_TASK_API_ENDPOINT}"

        return self._client.transport.perform_request(
            method="GET",
            url=API_URL,
        )

    def generate_embedding(self, model_id: str, sentences: List[str]):  # type: ignore
        """
        This method return embedding for given sentences (using ml commons _predict api)

        Parameters
        ----------
        :param model_id: unique id of the nlp model
        :type model_id: string
        :param sentences: List of sentences
        :type sentences: list of string

        Returns
        -------
        :return: returns a json object `inference_results` which is a list of embedding results of given sentences
            every item has 4 properties: name, data_type, shape, data (embedding value)
        :rtype: object
        """

        SENTENCE_EMBEDDING_API_ENDPOINT = f"_predict/text_embedding/{model_id}"
        API_URL = f"{ML_BASE_URI}/{SENTENCE_EMBEDDING_API_ENDPOINT}"

        API_BODY = {"text_docs": sentences, "target_response": ["sentence_embedding"]}

        return self._client.transport.perform_request(
            method="POST",
            url=API_URL,
            body=API_BODY,
        )
