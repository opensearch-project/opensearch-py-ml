# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.


import time
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
        multiple chunks and then upload chunks into opensearch cluster

        :param model_path: path of the zip file of the model
        :type model_path: string
        :param model_config_path: filepath of the model metadata. A json file of model metadata is expected
            Model metadata format example:
            {
                "name": "all-MiniLM-L6-v2",
                "version": 1,
                "model_format": "TORCH_SCRIPT",
                "model_config": {
                    "model_type": "bert",
                    "embedding_dimension": 384,
                    "framework_type": "sentence_transformers",
                },
            }

            refer to:
            https://opensearch.org/docs/latest/ml-commons-plugin/model-serving-framework/#upload-model-to-opensearch
        :type model_config_path: string
        :param isVerbose: if isVerbose is true method will print more messages. default False
        :type isVerbose: boolean
        :return: returns the model_id so that we can use this for further operation.
        :rtype: string
        """
        return self._model_uploader._upload_model(
            model_path, model_config_path, isVerbose
        )

    def load_model(self, model_id: str) -> object:
        """
        This method loads model into opensearch cluster using ml-common plugin's load model api

        :param model_id: unique id of the model
        :type model_id: string
        :return: returns a json object, with task_id and status key.
        :rtype: object
        """

        API_URL = f"{ML_BASE_URI}/models/{model_id}/_load"

        return self._client.transport.perform_request(
            method="POST",
            url=API_URL,
        )

    def get_task_info(self, task_id: str, wait_until_task_done: bool = False) -> object:
        """
        This method return information about a task running into opensearch cluster (using ml commons api)
        when we load a model

        :param task_id: unique id of the task
        :type task_id: string
        :param wait_until_task_done: a boolean indicator if we want to wait until a task done before
            returning the task related information
        :type task_id: bool
        :return: returns a json object, with detailed information about the task
        :rtype: object
        """
        if wait_until_task_done:
            end = time.time() + 120  # timeout seconds
            task_flag = False
            while not task_flag or time.time() < end:
                time.sleep(1)
                output = self._get_task_info(task_id)
                if (
                    output["state"] == "COMPLETED"
                    or output["state"] == "FAILED"
                    or output["state"] == "COMPLETED_WITH_ERROR"
                ):
                    task_flag = True
        return self._get_task_info(task_id)

    def _get_task_info(self, task_id: str):
        API_URL = f"{ML_BASE_URI}/tasks/{task_id}"

        return self._client.transport.perform_request(
            method="GET",
            url=API_URL,
        )

    def get_model_info(self, model_id: str) -> object:
        """
        This method return information about a model uploaded into opensearch cluster (using ml commons api)

        :param model_id: unique id of the model
        :type model_id: string
        :return: returns a json object, with detailed information about the model
        :rtype: object
        """

        API_URL = f"{ML_BASE_URI}/models/{model_id}"

        return self._client.transport.perform_request(
            method="GET",
            url=API_URL,
        )

    def generate_embedding(self, model_id: str, sentences: List[str]) -> object:
        """
        This method return embedding for given sentences (using ml commons _predict api)

        :param model_id: unique id of the nlp model
        :type model_id: string
        :param sentences: List of sentences
        :type sentences: list of string
        :return: returns a json object `inference_results` which is a list of embedding results of given sentences
            every item has 4 properties: name, data_type, shape, data (embedding value)
        :rtype: object
        """

        API_URL = f"{ML_BASE_URI}/_predict/text_embedding/{model_id}"

        API_BODY = {"text_docs": sentences, "target_response": ["sentence_embedding"]}

        return self._client.transport.perform_request(
            method="POST",
            url=API_URL,
            body=API_BODY,
        )

    def unload_model(self, model_id: str, node_ids: List[str] = []) -> object:
        """
        This method unloads a model from all the nodes or from the given list of nodes (using ml commons _unload api)

        :param model_id: unique id of the nlp model
        :type model_id: string
        :param node_ids: List of nodes
        :type node_ids: list of string
        :return: returns a json object with defining from which nodes the model has unloaded.
        :rtype: object
        """

        API_URL = f"{ML_BASE_URI}/models/{model_id}/_unload"

        API_BODY = {"node_ids": node_ids}
        if len(node_ids) > 0:
            return self._client.transport.perform_request(
                method="POST",
                url=API_URL,
                body=API_BODY,
            )
        else:
            return self._client.transport.perform_request(
                method="POST",
                url=API_URL,
            )

    def delete_model(self, model_id: str) -> object:
        """
        This method deletes a model from opensearch cluster (using ml commons api)

        :param model_id: unique id of the model
        :type model_id: string
        :return: returns a json object, with detailed information about the deleted model
        :rtype: object
        """

        API_URL = f"{ML_BASE_URI}/models/{model_id}"

        return self._client.transport.perform_request(
            method="DELETE",
            url=API_URL,
        )
    
    def search_model(self, algorithm_name: str = "") -> object:
        API_URL = f"{ML_BASE_URI}/models/_search"

        if len(algorithm_name) == 0:
            API_BODY = {"query": {"match_all": {}}}
        else:
            API_BODY = {
                "query": {
                    "term": {"algorithm": {"value": algorithm_name}}
                }
            }

        return self._client.transport.perform_request(
            method="POST",
            url=API_URL,
            body=API_BODY,
        )
