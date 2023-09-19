# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import pytest
from opensearchpy.client import OpenSearch

from opensearch_py_ml.ml_commons.ml_common_utils import (
    _generate_model_content_hash_value,
)
from opensearch_py_ml.ml_commons.model_uploader import ModelUploader
from tests import FLIGHTS_SMALL_FILE_NAME

# Define test files and indices
OPENSEARCH_HOST = "https://localhost:9200"
# for automated CI, we need put instance as the node name is `instance`
# OPENSEARCH_HOST = "https://instance:9200"
OPENSEARCH_ADMIN_USER, OPENSEARCH_ADMIN_PASSWORD = "admin", "admin"


# Define client to use in tests
OPENSEARCH_TEST_CLIENT = OpenSearch(
    hosts=[OPENSEARCH_HOST],
    http_auth=(OPENSEARCH_ADMIN_USER, OPENSEARCH_ADMIN_PASSWORD),
    verify_certs=False,
)

model_uploader = ModelUploader(OPENSEARCH_TEST_CLIENT)


def test_check_mandatory_field():
    model_meta = {}
    with pytest.raises(ValueError, match="Model metadata can't be empty"):
        model_uploader._check_mandatory_field(model_meta)
    model_meta = {
        "name": "all-MiniLM-L6-v2",
        "version": 1,
        "model_format": "TORCH_SCRIPT",
        "model_content_hash_value": "9376c2ebd7c83f99ec2526323786c348d2382e6d86576f750c89ea544d6bbb14",
        "total_chunks": 9,
        "model_config": {
            "model_type": "bert",
            "embedding_dimension": 384,
            "framework_type": "sentence_transformers",
            "all_config": '{"_name_or_path":"nreimers/MiniLM-L6-H384-uncased","architectures":["BertModel"],"attention_probs_dropout_prob":0.1,"gradient_checkpointing":false,"hidden_act":"gelu","hidden_dropout_prob":0.1,"hidden_size":384,"initializer_range":0.02,"intermediate_size":1536,"layer_norm_eps":1e-12,"max_position_embeddings":512,"model_type":"bert","num_attention_heads":12,"num_hidden_layers":6,"pad_token_id":0,"position_embedding_type":"absolute","transformers_version":"4.8.2","type_vocab_size":2,"use_cache":true,"vocab_size":30522}',
        },
    }
    assert True == model_uploader._check_mandatory_field(model_meta)

    model_meta["model_config"]["framework_type"] = ""
    with pytest.raises(ValueError, match="framework_type can not be empty"):
        model_uploader._check_mandatory_field(model_meta)

    model_meta["model_config"]["embedding_dimension"] = ""
    with pytest.raises(ValueError, match="embedding_dimension can not be empty"):
        model_uploader._check_mandatory_field(model_meta)

    model_meta["model_config"]["model_type"] = ""
    with pytest.raises(ValueError, match="model_type can not be empty"):
        model_uploader._check_mandatory_field(model_meta)

    model_meta["model_config"] = "asd"
    with pytest.raises(TypeError, match="model_config is expecting to be an object"):
        model_uploader._check_mandatory_field(model_meta)

    model_meta["model_config"] = ""
    with pytest.raises(ValueError, match="model_config can not be empty"):
        model_uploader._check_mandatory_field(model_meta)

    model_meta["total_chunks"] = ""
    with pytest.raises(ValueError, match="total_chunks can not be empty"):
        model_uploader._check_mandatory_field(model_meta)

    model_meta["model_content_hash_value"] = ""
    with pytest.raises(ValueError, match="model_content_hash_value can not be empty"):
        model_uploader._check_mandatory_field(model_meta)

    model_meta["model_format"] = ""
    with pytest.raises(ValueError, match="model_format can not be empty"):
        model_uploader._check_mandatory_field(model_meta)

    model_meta["version"] = ""
    with pytest.raises(ValueError, match="version can not be empty"):
        model_uploader._check_mandatory_field(model_meta)

    model_meta["name"] = ""
    with pytest.raises(ValueError, match="name can not be empty"):
        model_uploader._check_mandatory_field(model_meta)


def test_generate_model_content_hash_value():
    assert (
        "18521f420cf85149025b75df329689c416be0ce3fc78b2afdfdf177654b77b34"
        == _generate_model_content_hash_value(FLIGHTS_SMALL_FILE_NAME)
    )
