# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import os
import shutil
from zipfile import ZipFile

import pytest
import torch

from opensearch_py_ml.ml_models import SemanticHighlighterModel

TEST_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath("__file__")), "tests", "test_model_files"
)


def clean_test_folder(TEST_FOLDER):
    if os.path.exists(TEST_FOLDER):
        for files in os.listdir(TEST_FOLDER):
            sub_path = os.path.join(TEST_FOLDER, files)
            if os.path.isfile(sub_path):
                os.remove(sub_path)
            else:
                try:
                    shutil.rmtree(sub_path)
                except OSError as err:
                    print(
                        "Fail to delete files, please delete all files in "
                        + str(TEST_FOLDER)
                        + " "
                        + str(err)
                    )

        shutil.rmtree(TEST_FOLDER)


def compare_model_config(
    model_config_path,
    model_id,
    model_format,
    expected_model_description=None,
):
    try:
        with open(model_config_path) as json_file:
            model_config_data = json.load(json_file)
    except Exception as exec:
        assert (
            False
        ), f"Creating model config file for tracing in {model_format} raised an exception {exec}"

    assert (
        "name" in model_config_data and model_config_data["name"] == model_id
    ), f"Missing or Wrong model name in {model_format} model config file"

    assert (
        "model_format" in model_config_data
        and model_config_data["model_format"] == model_format
    ), f"Missing or Wrong model_format in {model_format} model config file"

    if expected_model_description is not None:
        assert (
            "description" in model_config_data
            and model_config_data["description"] == expected_model_description
        ), f"Missing or Wrong model description in {model_format} model config file'"

    assert (
        "model_content_size_in_bytes" in model_config_data
    ), f"Missing 'model_content_size_in_bytes' in {model_format} model config file"

    assert (
        "model_content_hash_value" in model_config_data
    ), f"Missing 'model_content_hash_value' in {model_format} model config file"


def compare_model_zip_file(zip_file_path, expected_filenames, model_format):
    with ZipFile(zip_file_path, "r") as f:
        filenames = set(f.namelist())
        assert (
            filenames == expected_filenames
        ), f"The content in the {model_format} model zip file does not match the expected content: {filenames} != {expected_filenames}"


def prepare_example_inputs():
    # Create mock inputs that the model would expect
    return {
        "input_ids": torch.tensor([[101, 2054, 2003, 1037, 2307, 1012, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1]]),
        "token_type_ids": torch.tensor([[0, 0, 0, 0, 0, 0, 0]]),
        "sentence_ids": torch.tensor([[0, 0, 0, 0, 0, 0, -100]]),
    }


clean_test_folder(TEST_FOLDER)
test_model = SemanticHighlighterModel(folder_path=TEST_FOLDER)


def test_check_attribute():
    try:
        check_attribute = getattr(test_model, "model_id", "folder_path")
    except AttributeError:
        check_attribute = False
    assert check_attribute

    assert test_model.folder_path == TEST_FOLDER
    assert (
        test_model.model_id == "opensearch-project/opensearch-semantic-highlighter-v1"
    )

    # Skip this part as it would modify files outside the test directory
    # clean_test_folder(default_folder)
    # test_model0 = SemanticHighlighterModel()
    # assert test_model0.folder_path == default_folder
    # clean_test_folder(default_folder)


def test_folder_path():
    # The SemanticHighlighterModel doesn't enforce the check for existing folders
    # like other model classes do. Let's verify it doesn't raise an exception,
    # which is its current behavior.
    test_non_empty_path = os.path.join(
        os.path.dirname(os.path.abspath("__file__")), "tests"
    )
    try:
        model = SemanticHighlighterModel(
            folder_path=test_non_empty_path, overwrite=False
        )
        # Clean up if a folder was created
        if (
            os.path.exists(model.folder_path)
            and os.path.basename(model.folder_path) == "semantic-highlighter"
        ):
            shutil.rmtree(model.folder_path)
    except Exception as e:
        assert False, f"Unexpected exception was raised: {e}"


def test_check_required_fields():
    # test without required_fields should raise TypeError
    with pytest.raises(TypeError):
        test_model.save_as_pt()


def test_save_as_pt():
    example_inputs = prepare_example_inputs()
    try:
        test_model.save_as_pt(example_inputs=example_inputs)
    except Exception as exec:
        assert False, f"Tracing model in torchScript raised an exception {exec}"


def test_save_as_onnx():
    example_inputs = prepare_example_inputs()
    # The save_as_onnx method is expected to raise NotImplementedError
    with pytest.raises(NotImplementedError) as exc_info:
        test_model.save_as_onnx(example_inputs=example_inputs)
    assert "ONNX format is not supported for semantic highlighter models" in str(
        exc_info.value
    )


def test_make_model_config_json_for_torch_script():
    model_id = "opensearch-project/opensearch-semantic-highlighter-v1"
    model_format = "TORCH_SCRIPT"
    expected_model_description = "This is a semantic highlighter model that extracts relevant passages from documents."

    clean_test_folder(TEST_FOLDER)
    test_model1 = SemanticHighlighterModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    example_inputs = prepare_example_inputs()
    test_model1.save_as_pt(model_id=model_id, example_inputs=example_inputs)
    model_config_path_torch = test_model1.make_model_config_json(
        model_format="TORCH_SCRIPT", description=expected_model_description
    )

    compare_model_config(
        model_config_path_torch,
        model_id,
        model_format,
        expected_model_description=expected_model_description,
    )

    clean_test_folder(TEST_FOLDER)


def test_model_zip_content():
    model_id = "opensearch-project/opensearch-semantic-highlighter-v1"
    model_format = "TORCH_SCRIPT"
    torch_script_zip_file_path = os.path.join(
        TEST_FOLDER, "opensearch-semantic-highlighter-v1.zip"
    )
    torch_script_expected_filenames = {
        "opensearch-semantic-highlighter-v1.pt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "LICENSE",
    }

    clean_test_folder(TEST_FOLDER)
    test_model2 = SemanticHighlighterModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    example_inputs = prepare_example_inputs()
    test_model2.save_as_pt(
        model_id=model_id,
        example_inputs=example_inputs,
        add_apache_license=True,
    )

    compare_model_zip_file(
        torch_script_zip_file_path, torch_script_expected_filenames, model_format
    )

    clean_test_folder(TEST_FOLDER)


clean_test_folder(TEST_FOLDER)
