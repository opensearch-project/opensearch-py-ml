# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import os
from math import ceil
from typing import Iterable

from opensearchpy import OpenSearch

from opensearch_py_ml.ml_commons_integration.ml_common_utils import ML_BASE_URI

DEFAULT_ML_COMMON_UPLOAD_CHUNK_SIZE = 10_000_000  # 10MB


class MLCommonUploadClient:
    """
    Client for performing model upload tasks to ml-commons plugin for OpenSearch.
    """

    def __init__(self, os_client: OpenSearch):
        self._client = os_client

    def put_model(
        self,
        model_path: str,
        model_name: str,
        version_number: int,
        chunk_size: int = DEFAULT_ML_COMMON_UPLOAD_CHUNK_SIZE,
        verbose: bool = False,
    ) -> None:
        total_num_chunks = ceil(os.stat(model_path).st_size / chunk_size)

        def model_file_chunk_generator() -> Iterable[str]:
            with open(model_path, "rb") as f:
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    yield data  # type: ignore # check if we actually need to do base64 encoding

        to_iterate_over = enumerate(model_file_chunk_generator())

        for i, chunk in to_iterate_over:
            if verbose:
                print(f"uploading chunk {i + 1} of {total_num_chunks}")
            self._client.transport.perform_request(
                method="POST",
                url=f"{ML_BASE_URI}/custom_model/upload_chunk/{model_name}/{version_number}/{i}/{total_num_chunks}",
                body=chunk,
            )
