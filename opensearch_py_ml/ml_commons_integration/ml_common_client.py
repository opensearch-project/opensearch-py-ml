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
from opensearch_py_ml.ml_commons_integration.upload.ml_common_upload_client import (
    DEFAULT_ML_COMMON_UPLOAD_CHUNK_SIZE,
    MLCommonUploadClient,
)


class MLCommonClient:
    """
    A client that communicates to the ml-common plugin for OpenSearch. This client allows for uploading of trained
    machine learning models to an OpenSearch index.
    """

    def __init__(self, os_client: OpenSearch):
        self._client = os_client
        self._upload_client = MLCommonUploadClient(os_client)
        self._load_client = MLCommonLoadClient(os_client)

    def put_model(
        self,
        model_path: str,
        model_name: str,
        version_number: int,
        chunk_size: int = DEFAULT_ML_COMMON_UPLOAD_CHUNK_SIZE,
        verbose: bool = False,
    ) -> None:
        self._upload_client.put_model(
            model_path, model_name, version_number, chunk_size, verbose
        )

    def load_model(self, model_name: str, version_number: int):  # type: ignore
        return self._load_client.load_model(model_name, version_number)
