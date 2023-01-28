# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

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
        test_model.zip_model()
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
        test_model3.zip_model()
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


clean_test_folder(TEST_FOLDER)
clean_test_folder(TESTDATA_UNZIP_FOLDER)
