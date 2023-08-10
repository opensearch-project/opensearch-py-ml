# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.


import json
import time
from typing import Any, List, Union

from deprecated.sphinx import deprecated
from opensearchpy import OpenSearch

from opensearch_py_ml.ml_commons.ml_common_utils import (
    ML_BASE_URI,
    MODEL_FORMAT_FIELD,
    MODEL_GROUP_ID,
    MODEL_NAME_FIELD,
    MODEL_VERSION_FIELD,
    TIMEOUT,
)
from opensearch_py_ml.ml_commons.model_execute import ModelExecute
from opensearch_py_ml.ml_commons.model_uploader import ModelUploader


class MLCommonClient:
    """
    A client that communicates to the ml-common plugin for OpenSearch. This client allows for registration of trained
    machine learning models to an OpenSearch index.
    """

    def __init__(self, os_client: OpenSearch):
        self._client = os_client
        self._model_uploader = ModelUploader(os_client)
        self._model_execute = ModelExecute(os_client)

    def execute(self, algorithm_name: str, input_json: dict) -> dict:
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
        return self._model_execute._execute(algorithm_name, input_json)

    @deprecated(
        reason="Since OpenSearch 2.7.0, you can use register_model instead",
        version="2.7.0",
    )
    def upload_model(
        self,
        model_path: str,
        model_config_path: str,
        isVerbose: bool = False,
        load_model: bool = True,
        wait_until_loaded: bool = True,
    ) -> str:
        """
        This method registers the model in the opensearch cluster using ml-common plugin's api.
        First, this method creates a model id to store model metadata and then breaks the model zip file into
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
        :param load_model: Whether to deploy the model using uploaded model chunks
        :type load_model: bool
        :param wait_until_loaded: If load_model is true, whether to wait until the model is deployed
        :type wait_until_loaded: bool
        :return: returns the model_id so that we can use this for further operation.
        :rtype: string
        """
        model_id = self._model_uploader._register_model(
            model_path, model_config_path, isVerbose
        )

        # loading the model chunks from model index
        if load_model:
            self.load_model(model_id, wait_until_loaded=wait_until_loaded)

        return model_id

    def register_model(
        self,
        model_path: str,
        model_config_path: str,
        model_group_id: str = "",
        isVerbose: bool = False,
        deploy_model: bool = True,
        wait_until_deployed: bool = True,
    ) -> str:
        """
        This method registers the model in the opensearch cluster using ml-common plugin's api.
        First, this method creates a model id to store model metadata and then breaks the model zip file into
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
        :param model_group_id: Model group id
        :type model_group_id: string
        :param isVerbose: if isVerbose is true method will print more messages. default False
        :type isVerbose: boolean
        :param deploy_model: Whether to deploy the model using uploaded model chunks
        :type deploy_model: bool
        :param wait_until_deployed: If deploy_model is true, whether to wait until the model is deployed
        :type wait_until_deployed: bool
        :return: returns the model_id so that we can use this for further operation.
        :rtype: string
        """
        model_id = self._model_uploader._register_model(
            model_path, model_config_path, model_group_id, isVerbose
        )

        # loading the model chunks from model index
        if deploy_model:
            self.deploy_model(model_id, wait_until_deployed=wait_until_deployed)

        return model_id

    @deprecated(
        reason="Since OpenSearch 2.7.0, you can use register_pretrained_model instead",
        version="2.7.0",
    )
    def upload_pretrained_model(
        self,
        model_name: str,
        model_version: str,
        model_format: str,
        load_model: bool = True,
        wait_until_loaded: bool = True,
    ):
        """
        This method registers the pretrained model in the opensearch cluster using ml-common plugin's api.
        First, this method creates a model id to store model info and then deploys the model if load_model is True.
        The model has to be supported by ML Commons. Refer to https://opensearch.org/docs/latest/ml-commons-plugin/pretrained-models/.

        :param model_name: Name of the pretrained model
        :type model_name: string
        :param model_version: Version of the pretrained model
        :type model_version: string
        :param model_format: "TORCH_SCRIPT" or "ONNX"
        :type model_format: string
        :param load_model: Whether to deploy the model using uploaded model chunks
        :type load_model: bool
        :param wait_until_loaded: If load_model is true, whether to wait until the model is deployed
        :type wait_until_loaded: bool
        :return: returns the model_id so that we can use this for further operation
        :rtype: string
        """
        # creating model meta doc
        model_config_json = {
            MODEL_NAME_FIELD: model_name,
            "version": model_version,
            "model_format": model_format,
        }
        model_id = self._send_model_info(model_config_json)

        # loading the model chunks from model index
        if load_model:
            self.load_model(model_id, wait_until_loaded=wait_until_loaded)

        return model_id

    def register_pretrained_model(
        self,
        model_name: str,
        model_version: str,
        model_format: str,
        model_group_id: str = "",
        deploy_model: bool = True,
        wait_until_deployed: bool = True,
    ):
        """
        This method registers the pretrained model in the opensearch cluster using ml-common plugin's api.
        First, this method creates a model id to store model info and then deploys the model if deploy_model is True.
        The model has to be supported by ML Commons. Refer to https://opensearch.org/docs/latest/ml-commons-plugin/pretrained-models/.

        :param model_name: Name of the pretrained model
        :type model_name: string
        :param model_version: Version of the pretrained model
        :type model_version: string
        :param model_format: "TORCH_SCRIPT" or "ONNX"
        :type model_format: string
        :param model_group_id: Model group id
        :type model_group_id: string
        :param deploy_model: Whether to deploy the model using uploaded model chunks
        :type deploy_model: bool
        :param wait_until_deployed: If deploy_model is true, whether to wait until the model is deployed
        :type wait_until_deployed: bool
        :return: returns the model_id so that we can use this for further operation
        :rtype: string
        """
        # creating model meta doc
        model_config_json = {
            MODEL_NAME_FIELD: model_name,
            MODEL_VERSION_FIELD: model_version,
            MODEL_FORMAT_FIELD: model_format,
            MODEL_GROUP_ID: model_group_id,
        }
        model_id = self._send_model_info(model_config_json)

        print(model_id)

        # loading the model chunks from model index
        if deploy_model:
            self.deploy_model(model_id, wait_until_deployed=wait_until_deployed)

        return model_id

    def _send_model_info(self, model_meta_json: dict):
        """
        This method sends the pretrained model info to ML Commons' register api

        :param model_meta_json: a dictionary object with model configurations
        :type model_meta_json: dict
        :return: returns a unique id of the model
        :rtype: string
        """
        output: Union[bool, Any] = self._client.transport.perform_request(
            method="POST",
            url=f"{ML_BASE_URI}/models/_register",
            body=model_meta_json,
        )
        end = time.time() + TIMEOUT  # timeout seconds
        task_flag = False
        while not task_flag or time.time() < end:
            time.sleep(1)
            status = self._get_task_info(output["task_id"])
            if status["state"] != "CREATED":
                task_flag = True
        # TODO: need to add the test case later for this line
        if not task_flag:
            raise TimeoutError("Model registration timed out")
        if status["state"] == "FAILED":
            raise Exception(status["error"])
        print("Model was registered successfully. Model Id: ", status["model_id"])
        return status["model_id"]

    @deprecated(
        reason="Since OpenSearch 2.7.0, you can use deploy_model instead",
        version="2.7.0",
    )
    def load_model(self, model_id: str, wait_until_loaded: bool = True) -> object:
        """
        This method deploys a model in the opensearch cluster using ml-common plugin's deploy model api

        :param model_id: unique id of the model
        :type model_id: string
        :param wait_until_loaded: Whether to wait until the model is deployed
        :type wait_until_loaded: bool
        :return: returns a json object, with task_id and status key.
        :rtype: object
        """

        API_URL = f"{ML_BASE_URI}/models/{model_id}/_deploy"

        task_id = self._client.transport.perform_request(method="POST", url=API_URL)[
            "task_id"
        ]

        if wait_until_loaded:
            # Wait until deployed
            for i in range(TIMEOUT):
                ml_model_status = self.get_model_info(model_id)
                model_state = ml_model_status.get("model_state")
                if model_state in ["DEPLOYED", "PARTIALLY_DEPLOYED"]:
                    break
                time.sleep(1)

            # TODO: need to add the test case later for this line
            # Check the model status
            if model_state == "DEPLOYED":
                print("Model deployed successfully")
            elif model_state == "PARTIALLY_DEPLOYED":
                print("Model deployed only partially")
            # TODO: need to add the test case later for this line
            else:
                raise Exception("Model deployment failed")

        return self._get_task_info(task_id)

    def deploy_model(self, model_id: str, wait_until_deployed: bool = True) -> object:
        """
        This method deploys a model in the opensearch cluster using ml-common plugin's deploy model api

        :param model_id: unique id of the model
        :type model_id: string
        :param wait_until_deployed: Whether to wait until the model is deployed
        :type wait_until_deployed: bool
        :return: returns a json object, with task_id and status key.
        :rtype: object
        """

        API_URL = f"{ML_BASE_URI}/models/{model_id}/_deploy"

        task_id = self._client.transport.perform_request(method="POST", url=API_URL)[
            "task_id"
        ]

        if wait_until_deployed:
            # Wait until deployed
            for i in range(TIMEOUT):
                ml_model_status = self.get_model_info(model_id)
                model_state = ml_model_status.get("model_state")
                if model_state in ["DEPLOYED", "PARTIALLY_DEPLOYED"]:
                    break
                time.sleep(1)

            # TODO: need to add the test case later for this line
            # Check the model status
            if model_state == "DEPLOYED":
                print("Model deployed successfully")
            elif model_state == "PARTIALLY_DEPLOYED":
                print("Model deployed only partially")
            else:
                raise Exception("Model deployment failed")

        return self._get_task_info(task_id)

    def get_task_info(self, task_id: str, wait_until_task_done: bool = False) -> object:
        """
        This method return information about a task running into opensearch cluster (using ml commons api)
        when we deploy a model

        :param task_id: unique id of the task
        :type task_id: string
        :param wait_until_task_done: a boolean indicator if we want to wait until a task done before
            returning the task related information
        :type wait_until_task_done: bool
        :return: returns a json object, with detailed information about the task
        :rtype: object
        """
        if wait_until_task_done:
            end = time.time() + TIMEOUT  # timeout seconds
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

    def search_task(self, input_json) -> object:
        """
        This method searches a task from opensearch cluster (using ml commons api)
        :param json: json input for the search request
        :type json: string or dict
        :return: returns a json object, with detailed information about the searched task
        :rtype: object
        """

        API_URL = f"{ML_BASE_URI}/tasks/_search"

        if isinstance(input_json, str):
            try:
                json_obj = json.loads(input_json)
                if not isinstance(json_obj, dict):
                    return "Invalid JSON object passed as argument."
                API_BODY = json.dumps(json_obj)
            except json.JSONDecodeError:
                return "Invalid JSON string passed as argument."
        elif isinstance(input_json, dict):
            API_BODY = json.dumps(input_json)
        else:
            return "Invalid JSON object passed as argument."

        return self._client.transport.perform_request(
            method="GET",
            url=API_URL,
            body=API_BODY,
        )

    def search_model(self, input_json) -> object:
        """
        This method searches a task from opensearch cluster (using ml commons api)
        :param json: json input for the search request
        :type json: string or dict
        :return: returns a json object, with detailed information about the searched task
        :rtype: object
        """

        API_URL = f"{ML_BASE_URI}/models/_search"

        if isinstance(input_json, str):
            try:
                json_obj = json.loads(input_json)
                if not isinstance(json_obj, dict):
                    return "Invalid JSON object passed as argument."
                API_BODY = json.dumps(json_obj)
            except json.JSONDecodeError:
                return "Invalid JSON string passed as argument."
        elif isinstance(input_json, dict):
            API_BODY = json.dumps(input_json)
        else:
            return "Invalid JSON object passed as argument."

        return self._client.transport.perform_request(
            method="POST",
            url=API_URL,
            body=API_BODY,
        )

    def get_model_info(self, model_id: str) -> object:
        """
        This method return information about a model registered in the opensearch cluster (using ml commons api)

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

    @deprecated(
        reason="Since OpenSearch 2.7.0, you can use undeploy_model instead",
        version="2.7.0",
    )
    def unload_model(self, model_id: str, node_ids: List[str] = []) -> object:
        """
        This method undeploys a model from all the nodes or from the given list of nodes (using ml commons _undeploy api)

        :param model_id: unique id of the nlp model
        :type model_id: string
        :param node_ids: List of nodes
        :type node_ids: list of string
        :return: returns a json object with defining from which nodes the model was undeployed.
        :rtype: object
        """

        API_URL = f"{ML_BASE_URI}/models/{model_id}/_undeploy"

        # TODO: need to add the test case later for this line
        API_BODY = {}
        if len(node_ids) > 0:
            API_BODY["node_ids"] = node_ids

        return self._client.transport.perform_request(
            method="POST",
            url=API_URL,
            body=API_BODY,
        )

    def undeploy_model(self, model_id: str, node_ids: List[str] = []) -> object:
        """
        This method undeploys a model from all the nodes or from the given list of nodes (using ml commons _undeploy api)

        :param model_id: unique id of the nlp model
        :type model_id: string
        :param node_ids: List of nodes
        :type node_ids: list of string
        :return: returns a json object with defining from which nodes the model was undeployed.
        :rtype: object
        """

        API_URL = f"{ML_BASE_URI}/models/{model_id}/_undeploy"

        API_BODY = {}
        if len(node_ids) > 0:
            API_BODY["node_ids"] = node_ids

        return self._client.transport.perform_request(
            method="POST",
            url=API_URL,
            body=API_BODY,
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

    def delete_task(self, task_id: str) -> object:
        """
        This method deletes a task from opensearch cluster (using ml commons api)

        :param task_id: unique id of the task
        :type task_id: string
        :return: returns a json object, with detailed information about the deleted task
        :rtype: object
        """

        API_URL = f"{ML_BASE_URI}/tasks/{task_id}"

        return self._client.transport.perform_request(
            method="DELETE",
            url=API_URL,
        )
