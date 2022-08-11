"""
Copyright OpenSearch Contributors
SPDX-License-Identifier: Apache-2.0
 """

DEFAULT_ML_COMMON_UPLOAD_CHUNK_SIZE = 4 * 1024 * 1024

from opensearchpy import OpenSearch

class MLCommonUploadClient:
    """
    Client for performing model upload tasks to ml-commons plugin for OpenSearch.
    """
    def __init__(self, os_client: OpenSearch):
        self._client = os_client

    def put_model(self, model_path: str, chunk_size: int = DEFAULT_ML_COMMON_UPLOAD_CHUNK_SIZE):
        pass