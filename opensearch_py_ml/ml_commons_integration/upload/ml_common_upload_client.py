"""
Copyright OpenSearch Contributors
SPDX-License-Identifier: Apache-2.0
 """

from opensearchpy import OpenSearch
import os
import math
from typing import Iterable
import base64
from opensearch_py_ml.ml_commons_integration.ml_common_utils import ML_BASE_URI

from tqdm.auto import tqdm # type: ignore

DEFAULT_ML_COMMON_UPLOAD_CHUNK_SIZE = 10_000_000 # 10MB


class MLCommonUploadClient:
    """
    Client for performing model upload tasks to ml-commons plugin for OpenSearch.
    """
    def __init__(self, os_client: OpenSearch):
        self._client = os_client

    def put_model(self,
                  model_path: str,
                  model_name: str,
                  version_number: int,
                  chunk_size: int = DEFAULT_ML_COMMON_UPLOAD_CHUNK_SIZE,
                  verbose: bool = False) -> None:
        total_model_size = os.stat(model_path).st_size

        def model_file_chunk_generator() -> Iterable[str]:
            with open(model_path, "rb") as f:
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    yield data # check if we actually need to do base64 encoding

        to_iterate_over = enumerate(model_file_chunk_generator())
        if verbose:
            to_iterate_over = tqdm(to_iterate_over)

        for i, chunk in to_iterate_over:
            self._client.transport.perform_request(
                method="POST",
                url=f"/{ML_BASE_URI}/custom_model/upload/{model_name}/{version_number}/{i}",
                body=chunk
            )