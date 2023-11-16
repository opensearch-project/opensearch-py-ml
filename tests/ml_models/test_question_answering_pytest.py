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


default_model_id = "distilbert-base-cased-distilled-squad"
clean_test_folder(TEST_FOLDER)
test_model = QuestionAnsweringModel(folder_path=TEST_FOLDER)


def test_check_attribute():
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
    test_model1 = QuestionAnsweringModel(
        folder_path=TEST_FOLDER, model_id=our_model_id
    )
    assert test_model1.model_id == our_model_id


def test_folder_path():
    with pytest.raises(Exception) as exc_info:
        test_non_empty_path = os.path.join(
            os.path.dirname(os.path.abspath("__file__")), "tests"
        )
        QuestionAnsweringModel(folder_path=test_non_empty_path, overwrite=False)
    assert exc_info.type is Exception
    assert "The default folder path already exists" in exc_info.value.args[0]



# New unit test to compare output of our model with official model
# def get_output_similarity():
#     questions = ["Who was Jim Henson?", "Where do I live?", "What's my name?", "Which name is also used to describe the Amazon rainforest in English?"]
#     contexts = ["Jim Henson was a nice puppet", "My name is Sarah and I live in London", "My name is Clara and I live in Berkeley.", "The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain 'Amazonas' in their names. The Amazon represents over half of the planet's remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species."]

#     # Obtain pytorch's official model
#     from transformers import AutoTokenizer, AutoModelForQuestionAnswering
#     import torch
#     tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
#     official_model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')

#     # Official model's answer
#     def official_model_answer(question, context):
#         inputs = tokenizer(question, context, return_tensors="pt")
#         with torch.no_grad():
#             outputs = official_model(**inputs)
#         answer_start_index = torch.argmax(outputs.start_logits, dim=-1).item()
#         answer_end_index = torch.argmax(outputs.end_logits, dim=-1).item()
#         predict_answer_tokens = inputs['input_ids'][0, answer_start_index : answer_end_index + 1]
#         official_answer = tokenizer.decode(predict_answer_tokens)
#         return official_answer

#     # Test onnx's answer
#     def test_onnx():
#         from transformers import AutoTokenizer
#         from onnxruntime import InferenceSession
#         import numpy as np
#         session = InferenceSession(f"{folder_path}/{model_id}.onnx")

#         for i in range(len(questions)):
#             question = questions[i]
#             context = contexts[i]
#             inputs = tokenizer(question, context, return_tensors="pt")
#             print(f"=== test {i}, question: {question}, context: {context}")

#             inputs = tokenizer(question, context, return_tensors="np")
#             outputs = session.run(output_names=["start_logits", "end_logits"], input_feed=dict(inputs))

#             answer_start_index = np.argmax(outputs[0], axis=-1).item()
#             answer_end_index = np.argmax(outputs[1], axis=-1).item()
#             predict_answer_tokens = inputs['input_ids'][0, answer_start_index : answer_end_index + 1]
#             answer = tokenizer.decode(predict_answer_tokens)
#             official_answer = official_model_answer(question, context)

#             assert answer == official_answer, f"FAILED: traced model answer [{answer}] != [{official_answer}]"
            
#             print(f"    Official answer: {official_answer}")
#             print(f"    Our answer: {answer}")
    
#     # Test torchscript model's answer
#     def test_pt():
#         traced_model = torch.jit.load(f"{folder_path}/{model_id}.pt")

#         for i in range(len(questions)):
#             question = questions[i]
#             context = contexts[i]
#             inputs = tokenizer(question, context, return_tensors="pt")
#             print(f"=== test {i}, question: {question}, context: {context}")

#             with torch.no_grad():
#                 outputs = traced_model(**inputs)
#             answer_start_index = torch.argmax(outputs["start_logits"], dim=-1).item()
#             answer_end_index = torch.argmax(outputs["end_logits"], dim=-1).item()
#             predict_answer_tokens = inputs['input_ids'][0, answer_start_index : answer_end_index + 1]
#             answer = tokenizer.decode(predict_answer_tokens)

#             print(f"    Official answer: {official_model_answer(question, context)}")
#             print(f"    Our answer: {answer}")

clean_test_folder(TEST_FOLDER)
clean_test_folder(TESTDATA_UNZIP_FOLDER)