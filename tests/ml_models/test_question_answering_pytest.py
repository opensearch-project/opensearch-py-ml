# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.
# How to run: pytest tests/ml_models/test_question_answering_pytest.py

import json
import os
import shutil
from zipfile import ZipFile

import pytest

from opensearch_py_ml.ml_models import QuestionAnsweringModel

# default parameters
default_model_id = "distilbert-base-cased-distilled-squad"
default_model_description = (
    "This is a question-answering model: it provides answers to a question and context."
)

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


def compare_model_config(
    model_config_path,
    model_id,
    model_format,
    expected_model_description=None,
    expected_model_config_data=None,
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

    if expected_model_config_data is not None:
        assert (
            "model_config" in model_config_data
        ), f"Missing 'model_config' in {model_format} model config file"

        if expected_model_config_data is not None:
            for k, v in expected_model_config_data.items():
                assert (
                    k in model_config_data["model_config"]
                    and model_config_data["model_config"][k] == v
                ) or (
                    k not in model_config_data["model_config"]
                    and k == "normalize_result"
                    and not v
                )

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


# New
clean_test_folder(TEST_FOLDER)
test_model = QuestionAnsweringModel(folder_path=TEST_FOLDER)


def test_check_attribute():
    test_model = QuestionAnsweringModel(folder_path=TEST_FOLDER)
    try:
        check_attribute = getattr(test_model, "model_id", "folder_path")
    except AttributeError:
        check_attribute = False
    assert check_attribute

    assert test_model.folder_path == TEST_FOLDER
    assert test_model.model_id == default_model_id

    default_folder = os.path.join(os.getcwd(), "question_answering_model_files")

    clean_test_folder(default_folder)
    test_model0 = QuestionAnsweringModel()
    assert test_model0.folder_path == default_folder
    clean_test_folder(default_folder)

    clean_test_folder(TEST_FOLDER)
    our_model_id = "distilbert-base-cased-distilled-squad"
    test_model1 = QuestionAnsweringModel(folder_path=TEST_FOLDER, model_id=our_model_id)
    assert test_model1.model_id == our_model_id


def test_folder_path():
    with pytest.raises(Exception) as exc_info:
        test_non_empty_path = os.path.join(
            os.path.dirname(os.path.abspath("__file__")), "tests"
        )
        QuestionAnsweringModel(folder_path=test_non_empty_path, overwrite=False)
    assert exc_info.type is Exception
    assert "The default folder path already exists" in exc_info.value.args[0]


# New tests for save_as_pt and save_as_onnx

test_cases = [
    {"question": "Who was Jim Henson?", "context": "Jim Henson was a nice puppet"},
    {
        "question": "Where do I live?",
        "context": "My name is Sarah and I live in London",
    },
    {
        "question": "What's my name?",
        "context": "My name is Clara and I live in Berkeley.",
    },
    {
        "question": "Which name is also used to describe the Amazon rainforest in English?",
        "context": "The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain 'Amazonas' in their names. The Amazon represents over half of the planet's remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species.",
    },
]


def get_official_answer(test_cases):
    # Obtain pytorch's official model
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    official_model = AutoModelForQuestionAnswering.from_pretrained(
        "distilbert-base-cased-distilled-squad"
    )

    results = []

    for case in test_cases:
        question, context = case["question"], case["context"]
        inputs = tokenizer(question, context, return_tensors="pt")
        with torch.no_grad():
            outputs = official_model(**inputs)
        answer_start_index = torch.argmax(outputs.start_logits, dim=-1).item()
        answer_end_index = torch.argmax(outputs.end_logits, dim=-1).item()
        results.append([answer_start_index, answer_end_index])

    return results


def get_pt_answer(test_cases, folder_path, model_id):
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    traced_model = torch.jit.load(f"{folder_path}/{model_id}.pt")

    results = []

    for case in test_cases:
        question, context = case["question"], case["context"]
        inputs = tokenizer(question, context, return_tensors="pt")

        with torch.no_grad():
            outputs = traced_model(**inputs)
        answer_start_index = torch.argmax(outputs["start_logits"], dim=-1).item()
        answer_end_index = torch.argmax(outputs["end_logits"], dim=-1).item()
        results.append([answer_start_index, answer_end_index])

    return results


def get_onnx_answer(test_cases, folder_path, model_id):
    import numpy as np
    from onnxruntime import InferenceSession
    from transformers import AutoTokenizer

    session = InferenceSession(f"{folder_path}/{model_id}.onnx")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    results = []

    for case in test_cases:
        question, context = case["question"], case["context"]
        inputs = tokenizer(question, context, return_tensors="pt")

        inputs = tokenizer(question, context, return_tensors="np")
        outputs = session.run(
            output_names=["start_logits", "end_logits"], input_feed=dict(inputs)
        )

        answer_start_index = np.argmax(outputs[0], axis=-1).item()
        answer_end_index = np.argmax(outputs[1], axis=-1).item()
        results.append([answer_start_index, answer_end_index])

    return results


def test_pt_answer():
    test_model = QuestionAnsweringModel(folder_path=TEST_FOLDER, overwrite=True)
    test_model.save_as_pt(default_model_id)
    pt_results = get_pt_answer(test_cases, TEST_FOLDER, default_model_id)
    official_results = get_official_answer(test_cases)
    for i in range(len(pt_results)):
        assert (
            pt_results[i] == official_results[i]
        ), f"Failed at index {i}: pt_results[{i}] ({pt_results[i]}) != official_results[{i}] ({official_results[i]})"

    clean_test_folder(TEST_FOLDER)
    clean_test_folder(TESTDATA_UNZIP_FOLDER)


def test_onnx_answer():
    test_model = QuestionAnsweringModel(folder_path=TEST_FOLDER, overwrite=True)
    test_model.save_as_onnx(default_model_id)
    onnx_results = get_onnx_answer(test_cases, TEST_FOLDER, default_model_id)
    official_results = get_official_answer(test_cases)
    for i in range(len(onnx_results)):
        assert (
            onnx_results[i] == official_results[i]
        ), f"Failed at index {i}: onnx_results[{i}] ({onnx_results[i]}) != official_results[{i}] ({official_results[i]})"

    clean_test_folder(TEST_FOLDER)
    clean_test_folder(TESTDATA_UNZIP_FOLDER)


def test_make_model_config_json_for_torch_script():
    model_id = default_model_id
    model_format = "TORCH_SCRIPT"
    expected_model_description = default_model_description
    expected_model_config_data = {
        "embedding_dimension": 768,
        "pooling_mode": "CLS",
        "normalize_result": False,
    }

    clean_test_folder(TEST_FOLDER)
    test_model5 = QuestionAnsweringModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    test_model5.save_as_pt(model_id=model_id, sentences=["today is sunny"])
    model_config_path_torch = test_model5.make_model_config_json(
        model_format="TORCH_SCRIPT", verbose=True
    )

    compare_model_config(
        model_config_path_torch,
        model_id,
        model_format,
        expected_model_description=expected_model_description,
        expected_model_config_data=expected_model_config_data,
    )

    clean_test_folder(TEST_FOLDER)


def test_make_model_config_json_for_onnx():
    model_id = default_model_id
    model_format = "ONNX"
    expected_model_description = default_model_description
    expected_model_config_data = {
        "embedding_dimension": 768,
        "pooling_mode": "CLS",
        "normalize_result": False,
    }

    clean_test_folder(TEST_FOLDER)
    test_model6 = QuestionAnsweringModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    test_model6.save_as_onnx(model_id=model_id)
    model_config_path_onnx = test_model6.make_model_config_json(model_format="ONNX")

    compare_model_config(
        model_config_path_onnx,
        model_id,
        model_format,
        expected_model_description=expected_model_description,
        expected_model_config_data=expected_model_config_data,
    )

    clean_test_folder(TEST_FOLDER)


def test_overwrite_fields_in_model_config():
    model_id = default_model_id
    model_format = "TORCH_SCRIPT"

    overwritten_model_config_data = {
        "embedding_dimension": 128,
        "pooling_mode": "MAX",
        "normalize_result": False,
    }

    clean_test_folder(TEST_FOLDER)
    test_model8 = QuestionAnsweringModel(
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

    compare_model_config(
        model_config_path_torch,
        model_id,
        model_format,
        expected_model_description=None,
        expected_model_config_data=overwritten_model_config_data,
    )

    clean_test_folder(TEST_FOLDER)


def test_missing_expected_description_in_readme_file():
    model_id = default_model_id
    model_format = "TORCH_SCRIPT"
    expected_model_description = default_model_description

    clean_test_folder(TEST_FOLDER)
    test_model10 = QuestionAnsweringModel(
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
    ), "Should use default model description when description is missing from README.md"

    clean_test_folder(TEST_FOLDER)


def test_overwrite_description():
    model_id = default_model_id
    model_format = "TORCH_SCRIPT"
    expected_model_description = "Expected Description"

    clean_test_folder(TEST_FOLDER)
    test_model11 = QuestionAnsweringModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    test_model11.save_as_pt(model_id=model_id, sentences=["today is sunny"])
    model_config_path_torch = test_model11.make_model_config_json(
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


def test_truncation_parameter():
    model_id = default_model_id
    MAX_LENGTH_TASB = 512

    clean_test_folder(TEST_FOLDER)
    test_model13 = QuestionAnsweringModel(
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


def test_save_as_pt_with_license():
    model_id = "distilbert-base-cased-distilled-squad"
    model_format = "TORCH_SCRIPT"
    torch_script_zip_file_path = os.path.join(
        TEST_FOLDER, "distilbert-base-cased-distilled-squad.zip"
    )
    torch_script_expected_filenames = {
        "distilbert-base-cased-distilled-squad.pt",
        "tokenizer.json",
        "LICENSE",
    }

    clean_test_folder(TEST_FOLDER)
    test_model15 = QuestionAnsweringModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    test_model15.save_as_pt(
        model_id=model_id,
        sentences=["today is sunny"],
        add_apache_license=True,
    )

    compare_model_zip_file(
        torch_script_zip_file_path, torch_script_expected_filenames, model_format
    )

    clean_test_folder(TEST_FOLDER)


def test_save_as_onnx_with_license():
    model_id = "distilbert-base-cased-distilled-squad"
    model_format = "ONNX"
    onnx_zip_file_path = os.path.join(
        TEST_FOLDER, "distilbert-base-cased-distilled-squad.zip"
    )
    onnx_expected_filenames = {
        "distilbert-base-cased-distilled-squad.onnx",
        "tokenizer.json",
        "LICENSE",
    }

    clean_test_folder(TEST_FOLDER)
    test_model16 = QuestionAnsweringModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    test_model16.save_as_onnx(model_id=model_id, add_apache_license=True)

    compare_model_zip_file(onnx_zip_file_path, onnx_expected_filenames, model_format)

    clean_test_folder(TEST_FOLDER)


clean_test_folder(TEST_FOLDER)
clean_test_folder(TESTDATA_UNZIP_FOLDER)
