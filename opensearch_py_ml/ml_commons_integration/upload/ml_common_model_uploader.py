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

from opensearch_py_ml.ml_commons_integration.ml_common_utils import (
    BUF_SIZE,
    ML_BASE_URI,
    MODEL_UPLOAD_CHUNK_SIZE,
)


class MLCommonModelUploader:
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

    def upload_model(
        self, model_path: str, model_meta_path: str, isVerbose: bool
    ) -> None:

        """
        This method uploads model into opensearch cluster using ml-common plugin's api.
        first this method creates a model id to store model metadata and then breaks the model zip file into
        multiple chunks and then upload chunks into cluster.

        @param model_path         string     path of the zip file of the model
        @param model_meta_path    string     filepath of the model metadata. A json file of model metadata is expected
        @param isVerbose          bool       if isVerbose is true method will print more messages.
        """

        total_num_chunks: int = ceil(
            os.stat(model_path).st_size / MODEL_UPLOAD_CHUNK_SIZE
        )

        # we are generating the sha1 hash for the model zip file
        hash_val_model_file = self.generate_hash(model_path)

        if isVerbose:
            print("Total number of chunks", total_num_chunks)
            print("Sha1 value of the model file: ", hash_val_model_file)

        model_meta_json_file = open(model_meta_path)

        model_meta_json: dict[str, Union[str, dict[str, str]]] = json.load(
            model_meta_json_file
        )
        model_meta_json[self.TOTAL_CHUNKS_FIELD] = total_num_chunks
        model_meta_json[self.MODEL_CONTENT_HASH_VALUE] = hash_val_model_file

        if self.check_mandatory_field(model_meta_json):
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
                        print(output)
                print("Model uploaded successfully")
            else:
                raise Exception(
                    "Model meta doc creation wasn't successful. Please check the errors"
                )

    def check_mandatory_field(self, model_meta: dict) -> bool:
        """
        This method checks if model meta doc has all the required fields to create a model meta doc in opensearch.

        @param model_meta         dict     content of the model meta file

        @return                   boolean  if all the required fields are present returns True otherwise
                                            raise exception
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

    def generate_hash(self, model_file_path: str) -> str:
        """
        Generate sha1 hash value for the model zip file.

        @param model_meta         dict     content of the model meta file

        @return                   boolean  if all the required fields are present returns True otherwise
                                            raise exception
        """

        sha1 = hashlib.sha1()
        with open(model_file_path, "rb") as file:
            while True:
                chunk = file.read(BUF_SIZE)
                if not chunk:
                    break
                sha1.update(chunk)
        sha1_value = sha1.hexdigest()
        return sha1_value
