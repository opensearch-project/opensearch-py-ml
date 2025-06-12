# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import os
import shutil
from unittest.mock import MagicMock, patch
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


def create_mock_traced_model():
    """Create a mock traced model for testing"""
    mock_model = MagicMock()
    # Mock the model to return a tuple of tensors (typical model output)
    mock_output = (torch.tensor([[0.1, 0.2, 0.3]]), torch.tensor([[0.4, 0.5, 0.6]]))
    mock_model.return_value = mock_output
    return mock_model


def test_test_traced_model_cpu_success():
    """Test _test_traced_model with successful CPU inference"""
    clean_test_folder(TEST_FOLDER)
    test_model_cpu = SemanticHighlighterModel(folder_path=TEST_FOLDER)

    # Create mock inputs
    original_inputs = prepare_example_inputs()
    model_path = os.path.join(TEST_FOLDER, "test_model.pt")

    # Create a dummy model file for testing
    os.makedirs(TEST_FOLDER, exist_ok=True)
    with open(model_path, "wb") as f:
        f.write(b"dummy model file")

    # Mock torch.jit.load to return our mock model
    mock_model = create_mock_traced_model()

    with patch("torch.jit.load", return_value=mock_model):
        with patch("torch.cuda.is_available", return_value=False):
            # This should not raise any exceptions
            test_model_cpu._test_traced_model(mock_model, original_inputs, model_path)

    clean_test_folder(TEST_FOLDER)


def test_test_traced_model_gpu_success_and_output_comparison():
    """Test _test_traced_model with GPU success and output structure comparison"""
    clean_test_folder(TEST_FOLDER)
    test_model_gpu = SemanticHighlighterModel(folder_path=TEST_FOLDER)

    # Create mock inputs
    original_inputs = prepare_example_inputs()
    model_path = os.path.join(TEST_FOLDER, "test_model.pt")

    # Create a dummy model file for testing
    os.makedirs(TEST_FOLDER, exist_ok=True)
    with open(model_path, "wb") as f:
        f.write(b"dummy model file")

    # Mock CPU model returns 2 tensors
    mock_cpu_model = MagicMock()
    cpu_output = (torch.tensor([[0.1, 0.2]]), torch.tensor([[0.3, 0.4]]))
    mock_cpu_model.return_value = cpu_output

    # Mock GPU model returns 3 tensors (different structure to test comparison logic)
    mock_gpu_model = MagicMock()
    gpu_tensor1 = MagicMock()
    gpu_tensor1.cpu.return_value = torch.tensor([[0.1, 0.2]])
    gpu_tensor2 = MagicMock()
    gpu_tensor2.cpu.return_value = torch.tensor([[0.3, 0.4]])
    gpu_tensor3 = MagicMock()
    gpu_tensor3.cpu.return_value = torch.tensor([[0.5, 0.6]])
    gpu_output = (gpu_tensor1, gpu_tensor2, gpu_tensor3)
    mock_gpu_model.return_value = gpu_output

    def mock_jit_load(path, map_location):
        if map_location.type == "cpu":
            return mock_cpu_model
        else:
            return mock_gpu_model

    with patch("torch.jit.load", side_effect=mock_jit_load):
        with patch("torch.cuda.is_available", return_value=True):
            # This covers GPU success path, output comparison, and different structures message
            test_model_gpu._test_traced_model(
                mock_cpu_model, original_inputs, model_path
            )

    clean_test_folder(TEST_FOLDER)


def test_test_traced_model_failure_scenarios():
    """Test _test_traced_model with various failure scenarios"""
    clean_test_folder(TEST_FOLDER)

    # Test CPU inference failure
    test_model_cpu_fail = SemanticHighlighterModel(folder_path=TEST_FOLDER)
    original_inputs = prepare_example_inputs()
    model_path = os.path.join(TEST_FOLDER, "test_model.pt")

    os.makedirs(TEST_FOLDER, exist_ok=True)
    with open(model_path, "wb") as f:
        f.write(b"dummy model file")

    # Mock model that raises an exception for CPU inference
    mock_model = MagicMock()
    mock_model.side_effect = RuntimeError("CPU inference failed")

    with patch("torch.jit.load", return_value=mock_model):
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CPU inference failed"):
                test_model_cpu_fail._test_traced_model(
                    mock_model, original_inputs, model_path
                )

    # Test model loading failure
    test_model_load_fail = SemanticHighlighterModel(folder_path=TEST_FOLDER)
    with patch("torch.jit.load", side_effect=FileNotFoundError("Model file not found")):
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(FileNotFoundError, match="Model file not found"):
                test_model_load_fail._test_traced_model(
                    None, original_inputs, model_path
                )

    clean_test_folder(TEST_FOLDER)


def test_test_traced_model_gpu_failure_and_no_gpu():
    """Test _test_traced_model with GPU failure and no GPU scenarios"""
    clean_test_folder(TEST_FOLDER)
    test_model = SemanticHighlighterModel(folder_path=TEST_FOLDER)

    original_inputs = prepare_example_inputs()
    model_path = os.path.join(TEST_FOLDER, "test_model.pt")

    os.makedirs(TEST_FOLDER, exist_ok=True)
    with open(model_path, "wb") as f:
        f.write(b"dummy model file")

    # Test GPU failure scenario (should not raise exception)
    mock_cpu_model = create_mock_traced_model()
    mock_gpu_model = MagicMock()
    mock_gpu_model.side_effect = RuntimeError("GPU inference failed")

    def mock_jit_load(path, map_location):
        if map_location.type == "cpu":
            return mock_cpu_model
        else:
            return mock_gpu_model

    with patch("torch.jit.load", side_effect=mock_jit_load):
        with patch("torch.cuda.is_available", return_value=True):
            # GPU failure should be handled gracefully
            test_model._test_traced_model(mock_cpu_model, original_inputs, model_path)

    # Test no GPU available scenario
    with patch("torch.jit.load", return_value=mock_cpu_model):
        with patch("torch.cuda.is_available", return_value=False):
            # Should skip GPU testing
            test_model._test_traced_model(mock_cpu_model, original_inputs, model_path)

    clean_test_folder(TEST_FOLDER)


clean_test_folder(TEST_FOLDER)
