# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import os
import shutil
import time
from functools import wraps
from zipfile import ZipFile

import pytest
from torch import nn

from opensearch_py_ml.ml_models import SparseEncodingModel
from opensearch_py_ml.ml_models.sparse_encoding_model import sanitize_model_modules
from utils.model_uploader.autotracing_utils import init_sparse_model

TEST_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath("__file__")), "tests", "test_model_files"
)
TESTDATA_FILENAME = os.path.join(
    os.path.dirname(os.path.abspath("__file__")), "tests", "sample_zip.zip"
)
TESTDATA_UNZIP_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath("__file__")), "tests", "sample_zip"
)


def retry(n_retries=3, delay_sec=10, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, n_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == n_retries:
                        raise
                    print(
                        f"Attempt {attempt} failed: {e}. Retrying in {delay_sec}sec..."
                    )
                    time.sleep(delay_sec)
            return None

        return wrapper

    return decorator


@retry(n_retries=5, delay_sec=60, exceptions=(ConnectionError,))
def init_model():
    return SparseEncodingModel(
        folder_path=TEST_FOLDER,
        model_id="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
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


def check_value(expected, actual, delta):
    if isinstance(expected, float):
        assert abs(expected - actual) <= delta
    else:
        assert expected == actual


def compare_model_config(
    model_config_path,
    model_id,
    model_format,
    expected_model_description=None,
    function_name="SPARSE_ENCODING",
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

    assert (
        "function_name" in model_config_data
        and model_config_data["function_name"] == function_name
    ), f"Missing or Wrong function_name in {model_format} model config file"

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


clean_test_folder(TEST_FOLDER)
# test model with model id "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"
test_model = init_model()


def test_check_attribute():
    try:
        check_attribute = getattr(test_model, "model_id", "folder_path")
    except AttributeError:
        check_attribute = False
    assert check_attribute

    assert test_model.folder_path == TEST_FOLDER
    assert (
        test_model.model_id
        == "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"
    )

    default_folder = os.path.join(os.getcwd(), "opensearch_neural_sparse_model_files")

    clean_test_folder(default_folder)
    test_model0 = SparseEncodingModel(
        model_id="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"
    )
    assert test_model0.folder_path == default_folder
    clean_test_folder(default_folder)

    clean_test_folder(TEST_FOLDER)
    test_model1 = SparseEncodingModel(
        folder_path=TEST_FOLDER, model_id="sentence-transformers/all-MiniLM-L6-v2"
    )
    assert test_model1.model_id == "sentence-transformers/all-MiniLM-L6-v2"


def test_folder_path():
    with pytest.raises(Exception) as exc_info:
        test_non_empty_path = os.path.join(
            os.path.dirname(os.path.abspath("__file__")), "tests"
        )
        SparseEncodingModel(
            model_id="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
            folder_path=test_non_empty_path,
            overwrite=False,
        )
    assert exc_info.type is Exception
    assert "The default folder path already exists" in exc_info.value.args[0]


def test_check_required_fields():
    # test without required_fields should raise TypeError
    with pytest.raises(TypeError):
        test_model.process_sparse_encoding()
    with pytest.raises(TypeError):
        test_model.save_as_pt()


def test_save_as_pt():
    try:
        test_model.save_as_pt(sentences=["today is sunny"])
    except Exception as exec:
        assert False, f"Tracing model in torchScript raised an exception {exec}"


def test_make_model_config_json_for_torch_script():
    model_format = "TORCH_SCRIPT"
    expected_model_description = "This is a sparse encoding model for opensearch-neural-sparse-encoding-doc-v2-distill."
    model_id = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"
    clean_test_folder(TEST_FOLDER)
    test_model3 = SparseEncodingModel(
        model_id="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
        folder_path=TEST_FOLDER,
    )
    test_model3.save_as_pt(model_id=model_id, sentences=["today is sunny"])
    model_config_path_torch = test_model3.make_model_config_json(
        model_format="TORCH_SCRIPT", description=expected_model_description
    )

    compare_model_config(
        model_config_path_torch,
        model_id,
        model_format,
        expected_model_description=expected_model_description,
    )

    clean_test_folder(TEST_FOLDER)


def test_overwrite_description():
    model_id = "sentence-transformers/msmarco-distilbert-base-tas-b"
    model_format = "TORCH_SCRIPT"
    expected_model_description = "Expected Description"

    clean_test_folder(TEST_FOLDER)
    test_model4 = SparseEncodingModel(
        folder_path=TEST_FOLDER,
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

    clean_test_folder(TEST_FOLDER)


def test_long_description():
    model_id = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"
    model_format = "TORCH_SCRIPT"
    expected_model_description = (
        "This is a sparce encoding model: It generate lots of tokens with different weight "
        "which used to semantic search."
        " The model was specifically trained for the task of semantic search."
    )

    clean_test_folder(TEST_FOLDER)
    test_model5 = SparseEncodingModel(
        folder_path=TEST_FOLDER,
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

    clean_test_folder(TEST_FOLDER)


def test_save_as_pt_with_license():
    model_id = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"
    model_format = "TORCH_SCRIPT"
    torch_script_zip_file_path = os.path.join(
        TEST_FOLDER, "opensearch-neural-sparse-encoding-doc-v2-distill.zip"
    )
    torch_script_expected_filenames = {
        "opensearch-neural-sparse-encoding-doc-v2-distill.pt",
        "tokenizer.json",
        "LICENSE",
    }

    clean_test_folder(TEST_FOLDER)
    test_model6 = SparseEncodingModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    test_model6.save_as_pt(
        model_id=model_id, sentences=["today is sunny"], add_apache_license=True
    )

    compare_model_zip_file(
        torch_script_zip_file_path, torch_script_expected_filenames, model_format
    )

    clean_test_folder(TEST_FOLDER)


def test_default_description():
    model_id = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"
    model_format = "TORCH_SCRIPT"
    expected_model_description = "This is a neural sparse encoding model: It transfers text into sparse vector, and then extract nonzero index and value to entry and weights. It serves only in ingestion and customer should use tokenizer model in query."

    clean_test_folder(TEST_FOLDER)
    test_model7 = SparseEncodingModel(
        folder_path=TEST_FOLDER,
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

    clean_test_folder(TEST_FOLDER)


def test_process_sparse_encoding():
    model_id = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"

    test_model8 = SparseEncodingModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    encoding_result = test_model8.process_sparse_encoding(["hello world", "hello"])
    assert len(encoding_result[0]) == 298
    check_value(1.6551653146743774, encoding_result[0]["hello"], 0.001)
    assert len(encoding_result[1]) == 260
    check_value(1.9172965288162231, encoding_result[1]["hello"], 0.001)

    test_model8 = SparseEncodingModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
        sparse_prune_ratio=0.1,
        activation="l0",
    )
    encoding_result = test_model8.process_sparse_encoding(["hello world", "hello"])
    assert len(encoding_result[0]) == 77
    check_value(0.9765068888664246, encoding_result[0]["hello"], 0.001)
    assert len(encoding_result[1]) == 64
    check_value(1.0706572532653809, encoding_result[1]["hello"], 0.001)


def test_sanitize_module_name_and_trace():
    class WeirdSub(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input_ids=None, attention_mask=None):
            pass

    # simulate weird remote module path
    WeirdSub.__module__ = "remote.repo@bad:name/1"

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = WeirdSub()

        def forward(self, features: dict):
            pass

    model = Toy().eval()
    sanitize_model_modules(model)

    # After sanitize, module name should only contain [0-9A-Za-z_.]
    assert all(c.isalnum() or c in {"_", "."} for c in model.m.__class__.__module__)


def test_init_sparse_model_kwargs_passthrough():
    received = {}

    class FakeModel:
        def __init__(
            self,
            model_id,
            folder_path,
            overwrite,
            sparse_prune_ratio,
            activation,
            model_init_kwargs,
        ):
            received["model_id"] = model_id
            received["folder_path"] = folder_path
            received["overwrite"] = overwrite
            received["sparse_prune_ratio"] = sparse_prune_ratio
            received["activation"] = activation
            received["model_init_kwargs"] = model_init_kwargs

    model = init_sparse_model(
        FakeModel,
        model_id="foo/bar",
        folder_path="/tmp/xyz",
        sparse_prune_ratio=0.2,
        activation="l0",
        model_init_kwargs={"trust_remote_code": True, "revision": "dev"},
    )

    assert isinstance(model, FakeModel)
    assert received["model_id"] == "foo/bar"
    assert received["folder_path"] == "/tmp/xyz"
    assert received["overwrite"] is True
    assert received["sparse_prune_ratio"] == 0.2
    assert received["activation"] == "l0"
    assert received["model_init_kwargs"]["trust_remote_code"] is True


clean_test_folder(TEST_FOLDER)
clean_test_folder(TESTDATA_UNZIP_FOLDER)
