# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.
import shutil
from pathlib import Path

import pytest

from opensearch_py_ml.ml_models import CrossEncoderModel
from tests.ml_models.test_sentencetransformermodel_pytest import (
    compare_model_config,
    compare_model_zip_file,
)

TEST_FOLDER = Path(__file__) / "tests" / "test_model_files"


@pytest.fixture(scope="function")
def tinybert() -> CrossEncoderModel:
    model = CrossEncoderModel("cross-encoder/ms-marco-TinyBERT-L-2-v2", overwrite=True)
    yield model
    shutil.rmtree(
        "/tmp/models/cross-encoder/ms-marco-TinyBert-L-2-v2", ignore_errors=True
    )


def test_pt_has_correct_files(tinybert):
    zip_path = tinybert.zip_model()
    config_path = tinybert.make_model_config_json()
    compare_model_zip_file(
        zip_file_path=zip_path,
        expected_filenames={"ms-marco-TinyBERT-L-2-v2.pt", "tokenizer.json", "LICENSE"},
        model_format="TORCH_SCRIPT",
    )
    compare_model_config(
        model_config_path=config_path,
        model_id="cross-encoder/ms-marco-TinyBERT-L-2-v2",
        model_format="TORCH_SCRIPT",
        expected_model_description={
            "model_type": "bert",
            "embedding_dimension": 1,
            "framework_type": "huggingface_transformers",
        },
    )


def test_onnx_has_correct_files(tinybert):
    zip_path = tinybert.zip_model(framework="onnx")
    config_path = tinybert.make_model_config_json()
    compare_model_zip_file(
        zip_file_path=zip_path,
        expected_filenames={
            "ms-marco-TinyBERT-L-2-v2.onnx",
            "tokenizer.json",
            "LICENSE",
        },
        model_format="ONNX",
    )
    compare_model_config(
        model_config_path=config_path,
        model_id="cross-encoder/ms-marco-TinyBERT-L-2-v2",
        model_format="ONNX",
        expected_model_description={
            "model_type": "bert",
            "embedding_dimension": 1,
            "framework_type": "huggingface_transformers",
        },
    )


def test_can_pick_names_for_files(tinybert):
    zip_path = tinybert.zip_model(
        framework="torch_script", zip_fname="funky-model-filename.zip"
    )
    config_path = tinybert.make_model_config_json(
        config_fname="funky-model-config.json"
    )
    assert (tinybert._folder_path / "funky-model-filename.zip").exists()
    compare_model_zip_file(
        zip_file_path=zip_path,
        expected_filenames={"ms-marco-TinyBERT-L-2-v2.pt", "tokenizer.json", "LICENSE"},
        model_format="TORCH_SCRIPT",
    )
    compare_model_config(
        model_config_path=config_path,
        model_id="cross-encoder/ms-marco-TinyBERT-L-2-v2",
        model_format="TORCH_SCRIPT",
        expected_model_description={
            "model_type": "bert",
            "embedding_dimension": 1,
            "framework_type": "huggingface_transformers",
        },
    )
    assert config_path.endswith("funky-model-config.json")
