# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import hashlib

ML_BASE_URI = "/_plugins/_ml"
MODEL_CHUNK_MAX_SIZE = 10_000_000
MODEL_MAX_SIZE = 4_000_000_000
BUF_SIZE = 65536  # lets read stuff in 64kb chunks!
TIMEOUT = 120  # timeout for synchronous method calls in seconds
META_API_ENDPOINT = "models/meta"
MODEL_NAME_FIELD = "name"
MODEL_VERSION_FIELD = "version"
MODEL_FORMAT_FIELD = "model_format"
TOTAL_CHUNKS_FIELD = "total_chunks"
MODEL_CONFIG_FIELD = "model_config"
MODEL_CONTENT_SIZE_IN_BYTES_FIELD = "model_content_size_in_bytes"
MODEL_TYPE = "model_type"
EMBEDDING_DIMENSION = "embedding_dimension"
FRAMEWORK_TYPE = "framework_type"
MODEL_CONTENT_HASH_VALUE = "model_content_hash_value"
MODEL_GROUP_ID = "model_group_id"
MODEL_FUNCTION_NAME = "function_name"
MODEL_TASK_TYPE = "model_task_type"


def _generate_model_content_hash_value(model_file_path: str) -> str:
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
