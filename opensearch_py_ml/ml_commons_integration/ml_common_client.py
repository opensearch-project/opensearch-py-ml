"""
Copyright OpenSearch Contributors
SPDX-License-Identifier: Apache-2.0
 """

from opensearchpy import OpenSearch
from upload.ml_common_upload_client import MLCommonUploadClient, DEFAULT_ML_COMMON_UPLOAD_CHUNK_SIZE


class MLCommonClient:
    """
    A client that communicates to the ml-common plugin for OpenSearch. This client allows for uploading of trained
    machine learning models to an OpenSearch index.
    """

    def __init__(self, os_client: OpenSearch):
        self._client = os_client
        self._upload_client = MLCommonUploadClient(os_client)

    def put_model(self, model_path: str, chunk_size: int = DEFAULT_ML_COMMON_UPLOAD_CHUNK_SIZE):
        self._upload_client.put_model(model_path, chunk_size)