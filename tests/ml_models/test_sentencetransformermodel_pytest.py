# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import os
import shutil

import pytest

from opensearch_py_ml.ml_models import SentenceTransformerModel

TEST_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath("__file__")), "tests", "test_model_files"
)
TESTDATA_FILENAME = os.path.join(
    os.path.dirname(os.path.abspath("__file__")), "tests", "sample_zip.zip"
)
TESTDATA_UNZIP_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath("__file__")), "tests", "sample_zip"
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


clean_test_folder(TEST_FOLDER)
test_model = SentenceTransformerModel(folder_path=TEST_FOLDER)


def test_check_attribute():
    try:
        check_attribute = getattr(test_model, "model_id", "folder_path")
    except AttributeError:
        check_attribute = False
    assert check_attribute

    assert test_model.folder_path == TEST_FOLDER
    assert test_model.model_id == "sentence-transformers/msmarco-distilbert-base-tas-b"

    default_folder = os.path.join(os.getcwd(), "sentence_transformer_model_files")

    clean_test_folder(default_folder)
    test_model0 = SentenceTransformerModel()
    assert test_model0.folder_path == default_folder
    clean_test_folder(default_folder)

    clean_test_folder(TEST_FOLDER)
    test_model1 = SentenceTransformerModel(
        folder_path=TEST_FOLDER, model_id="sentence-transformers/all-MiniLM-L6-v2"
    )
    assert test_model1.model_id == "sentence-transformers/all-MiniLM-L6-v2"


def test_folder_path():
    with pytest.raises(Exception) as exc_info:
        test_non_empty_path = os.path.join(
            os.path.dirname(os.path.abspath("__file__")), "tests"
        )
        SentenceTransformerModel(folder_path=test_non_empty_path, overwrite=False)
    assert exc_info.type is Exception
    assert "The default folder path already exists" in exc_info.value.args[0]


def test_check_required_fields():
    # test without required_fields should raise TypeError
    with pytest.raises(TypeError):
        test_model.train()

    with pytest.raises(TypeError):
        test_model.load_training_data()

    with pytest.raises(TypeError):
        test_model.train_model()

    with pytest.raises(TypeError):
        test_model.read_queries()

    with pytest.raises(TypeError):
        test_model.save_as_pt()


def test_missing_files():
    with pytest.raises(FileNotFoundError):
        test_model.train(read_path="1234")

    with pytest.raises(FileNotFoundError):
        test_model.read_queries(read_path="1234")

    # test synthetic queries already exists in folder
    with pytest.raises(Exception) as exc_info:
        temp_path = os.path.join(
            os.path.dirname(os.path.abspath("__file__")),
            "tests",
            "test_SentenceTransformerModel",
        )
        clean_test_folder(temp_path)
        test_model2 = SentenceTransformerModel(folder_path=temp_path)
        test_model2.read_queries(TESTDATA_FILENAME)
        test_model2.read_queries(TESTDATA_FILENAME)
        clean_test_folder(temp_path)
    assert "folder is not empty" in str(exc_info.value)

    # test no tokenizer.json file
    with pytest.raises(Exception) as exc_info:
        test_model.zip_model(verbose=True)
    assert "Cannot find tokenizer.json file" in str(exc_info.value)

    # test no model file
    with pytest.raises(Exception) as exc_info:
        temp_path = os.path.join(
            os.path.dirname(os.path.abspath("__file__")),
            "tests",
            "test_SentenceTransformerModel",
        )
        clean_test_folder(temp_path)
        test_model3 = SentenceTransformerModel(folder_path=temp_path)
        test_model3.save_as_pt(sentences=["today is sunny"])
        os.remove(os.path.join(temp_path, "msmarco-distilbert-base-tas-b.pt"))
        test_model3.zip_model(verbose=True)
        clean_test_folder(temp_path)
    assert "Cannot find model in the model path" in str(exc_info.value)

    # test no config.json
    with pytest.raises(Exception) as exc_info:
        temp_path = os.path.join(
            os.path.dirname(os.path.abspath("__file__")),
            "tests",
            "test_SentenceTransformerModel",
        )
        clean_test_folder(temp_path)
        test_model4 = SentenceTransformerModel(folder_path=temp_path)
        test_model4.save_as_pt(sentences=["today is sunny"])
        os.remove(os.path.join(temp_path, "config.json"))
        test_model4.make_model_config_json()
        clean_test_folder(temp_path)
    assert "Cannot find config.json" in str(exc_info.value)


def test_save_as_pt():
    try:
        test_model.save_as_pt(sentences=["today is sunny"])
    except Exception as exec:
        assert False, f"Tracing model in torchScript raised an exception {exec}"


def test_save_as_onnx():
    try:
        test_model.save_as_onnx()
    except Exception as exec:
        assert False, f"Tracing model in ONNX raised an exception {exec}"


def test_make_model_config_json_for_torch_script():
    model_id = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    expected_model_config_data = {
        "embedding_dimension": 384,
        "pooling_mode": "MEAN",
        "normalize_result": True,
    }

    clean_test_folder(TEST_FOLDER)
    test_model5 = SentenceTransformerModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    test_model5.save_as_pt(model_id=model_id, sentences=["today is sunny"])
    model_config_path_torch = test_model5.make_model_config_json(
        model_format="TORCH_SCRIPT", verbose=True
    )
    try:
        with open(model_config_path_torch) as json_file:
            model_config_data_torch = json.load(json_file)
    except Exception as exec:
        assert (
            False
        ), f"Creating model config file for tracing in torch_script raised an exception {exec}"

    assert (
        "name" in model_config_data_torch
        and model_config_data_torch["name"] == model_id
    ), "Missing or Wrong model name in torch script model config file"
    assert (
        "model_format" in model_config_data_torch
        and model_config_data_torch["model_format"] == "TORCH_SCRIPT"
    ), "Missing or Wrong model_format in torch script model config file"
    assert (
        "description" in model_config_data_torch
        and model_config_data_torch["description"]
        == "This is a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space and was designed for semantic search. It has been trained on 215M  pairs from diverse sources."
    ), "Missing or Wrong model description in onnx model config file'"
    assert (
        "model_config" in model_config_data_torch
    ), "Missing 'model_config' in torch script model config file"

    for k, v in expected_model_config_data.items():
        assert (
            k in model_config_data_torch["model_config"]
            and model_config_data_torch["model_config"][k] == v
        ) or (
            k not in model_config_data_torch["model_config"]
            and k == "normalize_result"
            and not v
        )

    clean_test_folder(TEST_FOLDER)


def test_make_model_config_json_for_onnx():
    model_id = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    expected_model_config_data = {
        "embedding_dimension": 384,
        "pooling_mode": "MEAN",
        "normalize_result": False,
    }

    clean_test_folder(TEST_FOLDER)
    test_model6 = SentenceTransformerModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    test_model6.save_as_onnx(model_id=model_id)
    model_config_path_onnx = test_model6.make_model_config_json(model_format="ONNX")
    try:
        with open(model_config_path_onnx) as json_file:
            model_config_data_onnx = json.load(json_file)
    except Exception as exec:
        assert (
            False
        ), f"Creating model config file for tracing in onnx raised an exception {exec}"

    assert (
        "name" in model_config_data_onnx and model_config_data_onnx["name"] == model_id
    ), "Missing or Wrong model name in onnx model config file"
    assert (
        "model_format" in model_config_data_onnx
        and model_config_data_onnx["model_format"] == "ONNX"
    ), "Missing or Wrong model_format in onnx model config file"
    assert (
        "description" in model_config_data_onnx
        and model_config_data_onnx["description"]
        == "This is a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search."
    ), "Missing or Wrong model description in onnx model config file"
    assert (
        "model_config" in model_config_data_onnx
    ), "Missing 'model_config' in onnx model config file"

    for k, v in expected_model_config_data.items():
        assert (
            k in model_config_data_onnx["model_config"]
            and model_config_data_onnx["model_config"][k] == v
        ) or (
            k not in model_config_data_onnx["model_config"]
            and k == "normalize_result"
            and not v
        )

    clean_test_folder(TEST_FOLDER)


def test_overwrite_fields_in_model_config():
    model_id = "sentence-transformers/all-distilroberta-v1"
    expected_model_config_data = {
        "embedding_dimension": 768,
        "pooling_mode": "MEAN",
        "normalize_result": True,
    }

    overwritten_model_config_data = {
        "embedding_dimension": 128,
        "pooling_mode": "MAX",
        "normalize_result": False,
    }

    clean_test_folder(TEST_FOLDER)
    test_model7 = SentenceTransformerModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    test_model7.save_as_pt(model_id=model_id, sentences=["today is sunny"])
    model_config_path_torch = test_model7.make_model_config_json(
        model_format="TORCH_SCRIPT"
    )

    try:
        with open(model_config_path_torch) as json_file:
            model_config_data_torch = json.load(json_file)
    except Exception as exec:
        assert (
            False
        ), f"Creating model config file for tracing in torch_script raised an exception {exec}"

    assert (
        "name" in model_config_data_torch
        and model_config_data_torch["name"] == model_id
    ), "Missing or Wrong model name in torch script model config file"
    assert (
        "model_format" in model_config_data_torch
        and model_config_data_torch["model_format"] == "TORCH_SCRIPT"
    ), "Missing or Wrong model_format in onnx model config file"
    assert (
        "model_config" in model_config_data_torch
    ), "Missing 'model_config' in torch script model config file"

    for k, v in expected_model_config_data.items():
        assert (
            k in model_config_data_torch["model_config"]
            and model_config_data_torch["model_config"][k] == v
        ) or (
            k not in model_config_data_torch["model_config"]
            and k == "normalize_result"
            and not v
        )

    clean_test_folder(TEST_FOLDER)
    test_model8 = SentenceTransformerModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    test_model8.save_as_pt(model_id=model_id, sentences=["today is sunny"])
    model_config_path_torch = test_model8.make_model_config_json(
        model_format="TORCH_SCRIPT",
        embedding_dimension=overwritten_model_config_data["embedding_dimension"],
        pooling_mode=overwritten_model_config_data["pooling_mode"],
        normalize_result=overwritten_model_config_data["normalize_result"],
    )

    try:
        with open(model_config_path_torch) as json_file:
            model_config_data_torch = json.load(json_file)
    except Exception as exec:
        assert (
            False
        ), f"Creating model config file for tracing in torch_script raised an exception {exec}"

    assert (
        "name" in model_config_data_torch
        and model_config_data_torch["name"] == model_id
    ), "Missing or Wrong model name in torch script model config file"
    assert (
        "model_format" in model_config_data_torch
        and model_config_data_torch["model_format"] == "TORCH_SCRIPT"
    ), "Missing or Wrong model_format in torch script model config file"
    assert (
        "model_config" in model_config_data_torch
    ), "Missing 'model_config' in torch script model config file"

    for k, v in overwritten_model_config_data.items():
        assert (
            k in model_config_data_torch["model_config"]
            and model_config_data_torch["model_config"][k] == v
        ) or (
            k not in model_config_data_torch["model_config"]
            and k == "normalize_result"
            and not v
        )

    clean_test_folder(TEST_FOLDER)

    
def test_missing_readme_md_file():
    model_id = "sentence-transformers/msmarco-distilbert-base-tas-b"
    clean_test_folder(TEST_FOLDER)
    test_model9 = SentenceTransformerModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    test_model9.save_as_pt(model_id=model_id, sentences=["today is sunny"])
    temp_path = os.path.join(
        TEST_FOLDER,
        "README.md",
    )
    os.remove(temp_path)
    model_config_path_torch = test_model9.make_model_config_json(
        model_format="TORCH_SCRIPT"
    )
    try:
        with open(model_config_path_torch) as json_file:
            model_config_data_torch = json.load(json_file)
    except Exception as exec:
        assert (
            False
        ), f"Creating model config file for tracing in torch_script raised an exception {exec}"

    assert (
        "description" in model_config_data_torch
        and model_config_data_torch["description"]
        == "This is a sentence-transformers model: It maps sentences & paragraphs to a 768 dimensional dense vector space."
    ), "Should use default model description when README.md file is missing"

    clean_test_folder(TEST_FOLDER)


def test_missing_expected_description_in_readme_file():
    model_id = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    clean_test_folder(TEST_FOLDER)
    test_model10 = SentenceTransformerModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    test_model10.save_as_pt(model_id=model_id, sentences=["today is sunny"])
    temp_path = os.path.join(
        TEST_FOLDER,
        "README.md",
    )
    with open(temp_path, "w") as f:
        f.write("No model description here")
    model_config_path_torch = test_model10.make_model_config_json(
        model_format="TORCH_SCRIPT"
    )
    try:
        with open(model_config_path_torch) as json_file:
            model_config_data_torch = json.load(json_file)
    except Exception as exec:
        assert (
            False
        ), f"Creating model config file for tracing in torch_script raised an exception {exec}"

    assert (
        "description" in model_config_data_torch
        and model_config_data_torch["description"]
        == "This is a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space."
    ), "Should use default model description when description is missing from README.md"

    clean_test_folder(TEST_FOLDER)


def test_overwrite_description():
    model_id = "sentence-transformers/msmarco-distilbert-base-tas-b"
    clean_test_folder(TEST_FOLDER)
    test_model11 = SentenceTransformerModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    test_model11.save_as_pt(model_id=model_id, sentences=["today is sunny"])
    model_config_path_torch = test_model11.make_model_config_json(
        model_format="TORCH_SCRIPT", description="Expected Description"
    )
    try:
        with open(model_config_path_torch) as json_file:
            model_config_data_torch = json.load(json_file)
    except Exception as exec:
        assert (
            False
        ), f"Creating model config file for tracing in torch_script raised an exception {exec}"

    assert (
        "description" in model_config_data_torch
        and model_config_data_torch["description"] == "Expected Description"
    ), "Cannot overwrite description in model config file"

    clean_test_folder(TEST_FOLDER)


def test_long_description():
    model_id = "sentence-transformers/gtr-t5-base"
    clean_test_folder(TEST_FOLDER)
    test_model12 = SentenceTransformerModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    test_model12.save_as_pt(model_id=model_id, sentences=["today is sunny"])
    model_config_path_torch = test_model12.make_model_config_json(
        model_format="TORCH_SCRIPT"
    )
    try:
        with open(model_config_path_torch) as json_file:
            model_config_data_torch = json.load(json_file)
    except Exception as exec:
        assert (
            False
        ), f"Creating model config file for tracing in torch_script raised an exception {exec}"

    assert (
        "description" in model_config_data_torch
        and model_config_data_torch["description"]
        == "This is a sentence-transformers model: It maps sentences & paragraphs to a 768 dimensional dense vector space. The model was specifically trained for the task of sematic search."
    ), "Missing or Wrong model description in torch_script model config file"

    clean_test_folder(TEST_FOLDER)


def test_truncation_parameter():
    model_id = "sentence-transformers/msmarco-distilbert-base-tas-b"
    MAX_LENGTH_TASB = 512

    clean_test_folder(TEST_FOLDER)
    test_model13 = SentenceTransformerModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    test_model13.save_as_pt(model_id=model_id, sentences=["today is sunny"])

    tokenizer_json_file_path = os.path.join(TEST_FOLDER, "tokenizer.json")
    try:
        with open(tokenizer_json_file_path, "r") as json_file:
            tokenizer_json = json.load(json_file)
    except Exception as exec:
        assert (
            False
        ), f"Creating tokenizer.json file for tracing raised an exception {exec}"

    assert tokenizer_json[
        "truncation"
    ], "truncation parameter in tokenizer.json is null"

    assert (
        tokenizer_json["truncation"]["max_length"] == MAX_LENGTH_TASB
    ), "max_length is not properly set"

    clean_test_folder(TEST_FOLDER)

clean_test_folder(TEST_FOLDER)
clean_test_folder(TESTDATA_UNZIP_FOLDER)
