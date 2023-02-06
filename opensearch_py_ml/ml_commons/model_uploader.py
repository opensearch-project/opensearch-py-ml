# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import hashlib
import json
import os
from math import ceil
from typing import Any, Iterable, Union

from opensearchpy import OpenSearch

from opensearch_py_ml.ml_commons.ml_common_utils import (
    BUF_SIZE,
    ML_BASE_URI,
    MODEL_MAX_SIZE,
    MODEL_UPLOAD_CHUNK_SIZE,
)


class ModelUploader:
    """
    Class for uploading model using ml-commons apis in opensearch cluster.
    """

    META_API_ENDPOINT = "models/meta"
    MODEL_NAME_FIELD = "name"
    MODEL_VERSION_FIELD = "version"
    MODEL_FORMAT_FIELD = "model_format"
    TOTAL_CHUNKS_FIELD = "total_chunks"
    MODEL_CONFIG_FIELD = "model_config"
    MODEL_TYPE = "model_type"
    EMBEDDING_DIMENSION = "embedding_dimension"
    FRAMEWORK_TYPE = "framework_type"
    MODEL_CONTENT_HASH_VALUE = "model_content_hash_value"

    def __init__(self, os_client: OpenSearch):
        self._client = os_client

    def _upload_model(
        self, model_path: str, model_meta_path: str, isVerbose: bool
    ) -> str:
        """
        This method uploads model into opensearch cluster using ml-common plugin's api.
        first this method creates a model id to store model metadata and then breaks the model zip file into
        multiple chunks and then upload chunks into cluster.

        :param model_path: path of the zip file of the model
        :type model_path: string
        :param model_meta_path:
            filepath of the model metadata. A json file of model metadata is expected
            Model metadata format example:
                {
                    "name": "all-MiniLM-L6-v2",
                    "version": 1,
                    "model_format": "TORCH_SCRIPT",
                    "model_config": {
                        "model_type": "bert",
                        "embedding_dimension": 384,
                        "framework_type": "sentence_transformers",
                        "all_config": '{"_name_or_path":"nreimers/MiniLM-L6-H384-uncased","architectures":["BertModel"],"attention_probs_dropout_prob":0.1,"gradient_checkpointing":false,"hidden_act":"gelu","hidden_dropout_prob":0.1,"hidden_size":384,"initializer_range":0.02,"intermediate_size":1536,"layer_norm_eps":1e-12,"max_position_embeddings":512,"model_type":"bert","num_attention_heads":12,"num_hidden_layers":6,"pad_token_id":0,"position_embedding_type":"absolute","transformers_version":"4.8.2","type_vocab_size":2,"use_cache":true,"vocab_size":30522}',
                    },
                }
            refer to:
                https://opensearch.org/docs/latest/ml-commons-plugin/model-serving-framework/#upload-model-to-opensearch
        :type model_meta_path: string
        :param isVerbose: if isVerbose is true method will print more messages
        :type isVerbose: bool
        :return: returns model id which is created by the model metadata
        :rtype: string
        """
        if os.stat(model_path).st_size > MODEL_MAX_SIZE:
            raise Exception("Model file size exceeds the limit of 4GB")

        total_num_chunks: int = ceil(
            os.stat(model_path).st_size / MODEL_UPLOAD_CHUNK_SIZE
        )

        # we are generating the sha1 hash for the model zip file
        hash_val_model_file = self._generate_hash(model_path)

        if isVerbose:
            print("Total number of chunks", total_num_chunks)
            print("Sha1 value of the model file: ", hash_val_model_file)

        model_meta_json_file = open(model_meta_path)

        model_meta_json: dict[str, Union[str, dict[str, str]]] = json.load(
            model_meta_json_file
        )
        model_meta_json[self.TOTAL_CHUNKS_FIELD] = total_num_chunks
        model_meta_json[self.MODEL_CONTENT_HASH_VALUE] = hash_val_model_file

        if self._check_mandatory_field(model_meta_json):
            meta_output: Union[bool, Any] = self._client.transport.perform_request(
                method="POST",
                url=f"{ML_BASE_URI}/{self.META_API_ENDPOINT}",
                body=model_meta_json,
            )
            print(
                "Model meta data was created successfully. Model Id: ",
                meta_output.get("model_id"),
            )

            # model meta doc is created successfully, and now we can upload model chunks to the model id
            if meta_output.get("status") == "CREATED" and meta_output.get("model_id"):
                model_id = meta_output.get("model_id")

                def model_file_chunk_generator() -> Iterable[str]:
                    with open(model_path, "rb") as f:
                        while True:
                            data = f.read(MODEL_UPLOAD_CHUNK_SIZE)
                            if not data:
                                break
                            yield data  # type: ignore # check if we actually need to do base64 encoding

                to_iterate_over = enumerate(model_file_chunk_generator())

                for i, chunk in to_iterate_over:
                    if isVerbose:
                        print(f"uploading chunk {i + 1} of {total_num_chunks}")
                    output = self._client.transport.perform_request(
                        method="POST",
                        url=f"{ML_BASE_URI}/models/{model_id}/chunk/{i}",
                        body=chunk,
                    )
                    if isVerbose:
                        print("Model id:", output)

                print("Model uploaded successfully")
                return model_id
            else:
                raise Exception(
                    "Model meta doc creation wasn't successful. Please check the errors"
                )

    def _check_mandatory_field(self, model_meta: dict) -> bool:
        """
        This method checks if model meta doc has all the required fields to create a model meta doc in opensearch

        Parameters
        ----------
        :param model_meta: content of the model meta file
        :type model_meta: dict

        Returns
        -------
        :return: if all the required fields are present returns True otherwise
                                            raise exception
        :rtype: bool
        """

        if model_meta:
            if not model_meta.get(self.MODEL_NAME_FIELD):
                raise ValueError(f"{self.MODEL_NAME_FIELD} can not be empty")
            if not model_meta.get(self.MODEL_VERSION_FIELD):
                raise ValueError(f"{self.MODEL_VERSION_FIELD} can not be empty")
            if not model_meta.get(self.MODEL_FORMAT_FIELD):
                raise ValueError(f"{self.MODEL_FORMAT_FIELD} can not be empty")
            if not model_meta.get(self.MODEL_CONTENT_HASH_VALUE):
                raise ValueError(f"{self.MODEL_CONTENT_HASH_VALUE} can not be empty")
            if not model_meta.get(self.TOTAL_CHUNKS_FIELD):
                raise ValueError(f"{self.TOTAL_CHUNKS_FIELD} can not be empty")
            if not model_meta.get(self.MODEL_CONFIG_FIELD):
                raise ValueError(f"{self.MODEL_CONFIG_FIELD} can not be empty")
            else:
                if not isinstance(model_meta.get(self.MODEL_CONFIG_FIELD), dict):
                    raise TypeError(
                        f"{self.MODEL_CONFIG_FIELD} is expecting to be an object"
                    )
                model_config = model_meta.get(self.MODEL_CONFIG_FIELD)
                if not model_config.get(self.MODEL_TYPE):
                    raise ValueError(f"{self.MODEL_TYPE} can not be empty")
                if not model_config.get(self.EMBEDDING_DIMENSION):
                    raise ValueError(f"{self.EMBEDDING_DIMENSION} can not be empty")
                if not model_config.get(self.FRAMEWORK_TYPE):
                    raise ValueError(f"{self.FRAMEWORK_TYPE} can not be empty")
            return True
        else:
            raise ValueError("Model metadata can't be empty")

    def _generate_hash(self, model_file_path: str) -> str:
        """
        Generate sha1 hash value for the model zip file.

        Parameters
        ----------
        :param model_file_path: file path of the model file
        :type model_file_path: string


        Returns
        -------
        :return: sha256 hash
        :rtype: string

        """

        sha256 = hashlib.sha256()
        with open(model_file_path, "rb") as file:
            while True:
                chunk = file.read(BUF_SIZE)
                if not chunk:
                    break
                sha256.update(chunk)
        sha256_value = sha256.hexdigest()
        return sha256_value
