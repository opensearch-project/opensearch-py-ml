# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import os
from copy import copy

import pytest

from opensearch_py_ml.ml_models import SparseTokenizeModel

from .test_sparseencondingmodel_pytest import (
    compare_model_config,
    compare_model_zip_file,
)


@pytest.fixture(scope="module")
def loaded_model(tmp_path_factory):
    model_folder = tmp_path_factory.mktemp("loaded_model") / "model_files"
    return SparseTokenizeModel(folder_path=str(model_folder))


@pytest.fixture
def model_folder(tmp_path):
    return str(tmp_path / "test_model_files")


@pytest.fixture
def test_model(model_folder, loaded_model):
    # Reuse downloaded weights without sharing output paths between tests.
    model = copy(loaded_model)
    model.folder_path = model_folder
    return model


def test_check_attribute(model_folder, test_model, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    try:
        check_attribute = getattr(test_model, "model_id", "folder_path")
    except AttributeError:
        check_attribute = False
    assert check_attribute

    assert test_model.folder_path == model_folder
    assert (
        test_model.model_id
        == "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"
    )

    default_folder = os.path.join(os.getcwd(), "opensearch_neural_sparse_model_files")

    test_model0 = SparseTokenizeModel()
    assert test_model0.folder_path == default_folder

    test_model1 = SparseTokenizeModel(
        folder_path=model_folder,
        model_id="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
    )
    assert (
        test_model1.model_id
        == "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"
    )


def test_folder_path(tmp_path):
    test_non_empty_path = tmp_path / "non_empty"
    test_non_empty_path.mkdir()
    (test_non_empty_path / "existing-file").touch()
    with pytest.raises(Exception) as exc_info:
        SparseTokenizeModel(folder_path=str(test_non_empty_path), overwrite=False)
    assert exc_info.type is Exception
    assert "The default folder path already exists" in exc_info.value.args[0]


def test_check_required_fields(test_model):
    # test without required_fields should raise TypeError
    with pytest.raises(TypeError):
        test_model.process_sparse_encoding()
    with pytest.raises(TypeError):
        test_model.save_as_pt()


def test_save_as_pt(test_model):
    try:
        test_model.save_as_pt(sentences=["today is sunny"])
    except Exception as exec:
        assert False, f"Tracing model in torchScript raised an exception {exec}"


def test_make_model_config_json_for_torch_script(model_folder):
    model_format = "TORCH_SCRIPT"
    expected_model_description = "This is a sparse encoding model for opensearch-neural-sparse-encoding-doc-v3-distill."
    model_id = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"
    test_model3 = SparseTokenizeModel(model_id=model_id, folder_path=model_folder)
    test_model3.save_as_pt(model_id=model_id, sentences=["today is sunny"])
    model_config_path_torch = test_model3.make_model_config_json(
        model_format="TORCH_SCRIPT", description=expected_model_description
    )

    compare_model_config(
        model_config_path_torch,
        model_id,
        model_format,
        expected_model_description=expected_model_description,
        function_name="SPARSE_TOKENIZE",
    )


def test_overwrite_description(model_folder):
    model_id = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"
    model_format = "TORCH_SCRIPT"
    expected_model_description = "Expected Description"

    test_model4 = SparseTokenizeModel(
        folder_path=model_folder,
        model_id=model_id,
    )

    test_model4.save_as_pt(model_id=model_id, sentences=["today is sunny"])
    model_config_path_torch = test_model4.make_model_config_json(
        model_format=model_format, description=expected_model_description
    )
    try:
        with open(model_config_path_torch) as json_file:
            model_config_data_torch = json.load(json_file)
    except Exception as exec:
        assert (
            False
        ), f"Creating model config file for tracing in {model_format} raised an exception {exec}"

    assert (
        "description" in model_config_data_torch
        and model_config_data_torch["description"] == expected_model_description
    ), "Cannot overwrite description in model config file"


def test_long_description(model_folder):
    model_id = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"
    model_format = "TORCH_SCRIPT"
    expected_model_description = (
        "This is a sparce encoding model: It generate lots of tokens with different weight "
        "which used to semantic search."
        " The model was specifically trained for the task of semantic search."
    )

    test_model5 = SparseTokenizeModel(
        folder_path=model_folder,
        model_id=model_id,
    )

    test_model5.save_as_pt(model_id=model_id, sentences=["today is sunny"])
    model_config_path_torch = test_model5.make_model_config_json(
        model_format=model_format, description=expected_model_description
    )
    try:
        with open(model_config_path_torch) as json_file:
            model_config_data_torch = json.load(json_file)
    except Exception as exec:
        assert (
            False
        ), f"Creating model config file for tracing in {model_format} raised an exception {exec}"

    assert (
        "description" in model_config_data_torch
        and model_config_data_torch["description"] == expected_model_description
    ), "Missing or Wrong model description in model config file when the description is longer than normally."


def test_save_as_pt_with_license(model_folder):
    model_id = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"
    model_format = "TORCH_SCRIPT"
    torch_script_zip_file_path = os.path.join(
        model_folder, "opensearch-neural-sparse-encoding-doc-v2-distill.zip"
    )
    torch_script_expected_filenames = {
        "idf.json",
        "tokenizer.json",
        "LICENSE",
    }

    test_model6 = SparseTokenizeModel(
        folder_path=model_folder,
        model_id=model_id,
    )

    test_model6.save_as_pt(
        model_id=model_id, sentences=["today is sunny"], add_apache_license=True
    )

    compare_model_zip_file(
        torch_script_zip_file_path, torch_script_expected_filenames, model_format
    )


def test_default_description(model_folder):
    model_id = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"
    model_format = "TORCH_SCRIPT"
    expected_model_description = "This is a neural sparse tokenizer model: It tokenize input sentence into tokens and assign pre-defined weight from IDF to each. It serves only in query."

    test_model7 = SparseTokenizeModel(
        folder_path=model_folder,
        model_id=model_id,
    )

    test_model7.save_as_pt(model_id=model_id, sentences=["today is sunny"])
    model_config_path_torch = test_model7.make_model_config_json(
        model_format=model_format
    )
    try:
        with open(model_config_path_torch) as json_file:
            model_config_data_torch = json.load(json_file)
    except Exception as exec:
        assert (
            False
        ), f"Creating model config file for tracing in {model_format} raised an exception {exec}"

    assert (
        "description" in model_config_data_torch
        and model_config_data_torch["description"] == expected_model_description
    ), "Missing or Wrong model description in model config file when the description is not given."


def test_process_sparse_encoding(model_folder):
    model_id = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"

    test_model8 = SparseTokenizeModel(
        folder_path=model_folder,
        model_id=model_id,
    )
    encoding_result = test_model8.process_sparse_encoding(["hello world", "hello"])
    assert encoding_result == [
        {"hello": 6.937756538391113, "world": 3.4208686351776123},
        {"hello": 6.937756538391113},
    ]


def test_save(model_folder):
    model_id = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"

    test_model9 = SparseTokenizeModel(
        folder_path=model_folder,
        model_id=model_id,
    )
    test_model9.save(model_folder)
    assert os.path.exists(os.path.join(model_folder, "tokenizer.json"))
    assert os.path.exists(os.path.join(model_folder, "idf.json"))
