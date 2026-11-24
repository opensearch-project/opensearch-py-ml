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
    model_format = "TORCH_SCRIPT"
    expected_model_description = "This is a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space and was designed for semantic search. It has been trained on 215M  pairs from diverse sources."
    expected_model_config_data = {
        "embedding_dimension": 384,
        "pooling_mode": "MEAN",
        "normalize_result": True,
        "additional_config": {"space_type": "l2"},
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

    compare_model_config(
        model_config_path_torch,
        model_id,
        model_format,
        expected_model_description=expected_model_description,
        expected_model_config_data=expected_model_config_data,
    )

    clean_test_folder(TEST_FOLDER)


def test_make_model_config_json_for_onnx():
    model_id = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    model_format = "ONNX"
    expected_model_description = "This is a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search."
    expected_model_config_data = {
        "embedding_dimension": 384,
        "pooling_mode": "MEAN",
        "normalize_result": False,
        "additional_config": {"space_type": "cosinesimil"},
    }

    clean_test_folder(TEST_FOLDER)
    test_model6 = SentenceTransformerModel(
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
    model_id = "sentence-transformers/all-distilroberta-v1"
    model_format = "TORCH_SCRIPT"
    expected_model_config_data = {
        "embedding_dimension": 768,
        "pooling_mode": "MEAN",
        "normalize_result": True,
        "additional_config": {"space_type": "l2"},
    }

    overwritten_model_config_data = {
        "embedding_dimension": 128,
        "pooling_mode": "MAX",
        "normalize_result": False,
        "additional_config": {"space_type": "cosinesimil"},
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

    compare_model_config(
        model_config_path_torch,
        model_id,
        model_format,
        expected_model_description=None,
        expected_model_config_data=expected_model_config_data,
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
        additional_config=overwritten_model_config_data["additional_config"],
    )

    compare_model_config(
        model_config_path_torch,
        model_id,
        model_format,
        expected_model_description=None,
        expected_model_config_data=overwritten_model_config_data,
    )

    clean_test_folder(TEST_FOLDER)


def test_missing_readme_md_file():
    model_id = "sentence-transformers/msmarco-distilbert-base-tas-b"
    model_format = "TORCH_SCRIPT"
    expected_model_description = "This is a sentence-transformers model: It maps sentences & paragraphs to a 768 dimensional dense vector space."

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
    ), "Should use default model description when README.md file is missing"

    clean_test_folder(TEST_FOLDER)


def test_missing_expected_description_in_readme_file():
    model_id = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    model_format = "TORCH_SCRIPT"
    expected_model_description = "This is a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space."

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
    model_id = "sentence-transformers/msmarco-distilbert-base-tas-b"
    model_format = "TORCH_SCRIPT"
    expected_model_description = "Expected Description"

    clean_test_folder(TEST_FOLDER)
    test_model11 = SentenceTransformerModel(
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


def test_long_description():
    model_id = "sentence-transformers/gtr-t5-base"
    model_format = "TORCH_SCRIPT"
    expected_model_description = "This is a sentence-transformers model: It maps sentences & paragraphs to a 768 dimensional dense vector space. The model was specifically trained for the task of semantic search."

    clean_test_folder(TEST_FOLDER)
    test_model12 = SentenceTransformerModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    test_model12.save_as_pt(model_id=model_id, sentences=["today is sunny"])
    model_config_path_torch = test_model12.make_model_config_json(
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
    ), "Missing or Wrong model description in model config file when the description is longer than normally."

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


def test_undefined_model_max_length_in_tokenizer_for_torch_script():
    # Model of which tokenizer has undefined model max length.
    model_id = "intfloat/e5-small-v2"
    expected_max_length = 512

    clean_test_folder(TEST_FOLDER)
    test_model14 = SentenceTransformerModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    test_model14.save_as_pt(model_id=model_id, sentences=["today is sunny"])

    tokenizer_json_file_path = os.path.join(TEST_FOLDER, "tokenizer.json")
    try:
        with open(tokenizer_json_file_path, "r") as json_file:
            tokenizer_json = json.load(json_file)
    except Exception as exec:
        assert (
            False
        ), f"Creating tokenizer.json file for tracing raised an exception {exec}"

    assert (
        tokenizer_json["truncation"]["max_length"] == expected_max_length
    ), "max_length is not properly set"

    clean_test_folder(TEST_FOLDER)


def test_undefined_model_max_length_in_tokenizer_for_onnx():
    # Model of which tokenizer has undefined model max length.
    model_id = "intfloat/e5-small-v2"
    expected_max_length = 512

    clean_test_folder(TEST_FOLDER)
    test_model14 = SentenceTransformerModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    test_model14.save_as_onnx(model_id=model_id)

    tokenizer_json_file_path = os.path.join(TEST_FOLDER, "tokenizer.json")
    try:
        with open(tokenizer_json_file_path, "r") as json_file:
            tokenizer_json = json.load(json_file)
    except Exception as exec:
        assert (
            False
        ), f"Creating tokenizer.json file for tracing raised an exception {exec}"

    assert (
        tokenizer_json["truncation"]["max_length"] == expected_max_length
    ), "max_length is not properly set"

    clean_test_folder(TEST_FOLDER)


def test_save_as_pt_with_license():
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    model_format = "TORCH_SCRIPT"
    torch_script_zip_file_path = os.path.join(TEST_FOLDER, "all-MiniLM-L6-v2.zip")
    torch_script_expected_filenames = {
        "all-MiniLM-L6-v2.pt",
        "tokenizer.json",
        "LICENSE",
    }

    clean_test_folder(TEST_FOLDER)
    test_model15 = SentenceTransformerModel(
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
    model_id = "sentence-transformers/all-distilroberta-v1"
    model_format = "ONNX"
    onnx_zip_file_path = os.path.join(TEST_FOLDER, "all-distilroberta-v1.zip")
    onnx_expected_filenames = {"all-distilroberta-v1.onnx", "tokenizer.json", "LICENSE"}

    clean_test_folder(TEST_FOLDER)
    test_model16 = SentenceTransformerModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    test_model16.save_as_onnx(model_id=model_id, add_apache_license=True)

    compare_model_zip_file(onnx_zip_file_path, onnx_expected_filenames, model_format)

    clean_test_folder(TEST_FOLDER)


def test_zip_model_with_license():
    model_id = "sentence-transformers/msmarco-distilbert-base-tas-b"
    model_format = "TORCH_SCRIPT"
    zip_file_path = os.path.join(TEST_FOLDER, "msmarco-distilbert-base-tas-b.zip")
    expected_filenames_wo_license = {
        "msmarco-distilbert-base-tas-b.pt",
        "tokenizer.json",
    }
    expected_filenames_with_license = {
        "msmarco-distilbert-base-tas-b.pt",
        "tokenizer.json",
        "LICENSE",
    }

    clean_test_folder(TEST_FOLDER)
    test_model17 = SentenceTransformerModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    test_model17.save_as_pt(model_id=model_id, sentences=["today is sunny"])
    compare_model_zip_file(zip_file_path, expected_filenames_wo_license, model_format)

    test_model17.zip_model(add_apache_license=True)
    compare_model_zip_file(zip_file_path, expected_filenames_with_license, model_format)

    clean_test_folder(TEST_FOLDER)


ONNX_TEST_MODEL_ID = "sentence-transformers/paraphrase-MiniLM-L3-v2"
ONNX_TEST_SENTENCES = [
    "First test sentence",
    "This is another sentence used for testing model embedding outputs.",
]


def _load_onnx_graph(onnx_path):
    import onnx

    return onnx.load(onnx_path, load_external_data=False)


def _run_onnx_like_ml_commons(onnx_path, tokenizer_path, model_config, sentences):
    """Run the exported model the way ml-commons does (inputs, pooling, normalization)."""
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = {i.name for i in session.get_inputs()}
    embeddings = []
    for sentence in sentences:
        encoding = tokenizer.encode(sentence)
        feed = {
            "input_ids": np.array([encoding.ids], dtype=np.int64),
            "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
        }
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.array([encoding.type_ids], dtype=np.int64)
        output = session.run(None, feed)[0][0]
        mask = np.array(encoding.attention_mask, dtype=np.float32)[:, None]
        pooling_mode = model_config["pooling_mode"]
        if pooling_mode == "MEAN":
            output = (output * mask).sum(axis=0) / mask.sum()
        elif pooling_mode == "MEAN_SQRT_LEN":
            output = (output * mask).sum(axis=0) / np.sqrt(mask.sum())
        elif pooling_mode == "WEIGHTED_MEAN":
            weights = mask * np.arange(1, len(mask) + 1, dtype=np.float32)[:, None]
            output = (output * weights).sum(axis=0) / weights.sum()
        elif pooling_mode == "LAST_TOKEN":
            output = output[int(np.nonzero(mask[:, 0])[0][-1])]
        elif pooling_mode == "CLS":
            output = output[0]
        if model_config.get("normalize_result"):
            output = output / np.linalg.norm(output)
        embeddings.append(output)
    return np.array(embeddings)


def _assert_matches_sentence_transformers(model_id_or_path, embeddings, prompt=None):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    expected = SentenceTransformer(model_id_or_path, device="cpu").encode(
        ONNX_TEST_SENTENCES, prompt=prompt
    )
    for actual, ref in zip(embeddings, expected):
        cosine = float(
            np.dot(actual, ref) / (np.linalg.norm(actual) * np.linalg.norm(ref))
        )
        assert cosine > 0.9999, f"ONNX output differs from original model: {cosine}"


def test_save_as_onnx_is_compatible_with_ml_commons():
    # plain Transformer -> Pooling pipeline (bert): single file, IR <= 9, token embeddings out,
    # token_type_ids in, ml-commons applies MEAN pooling
    clean_test_folder(TEST_FOLDER)
    test_model = SentenceTransformerModel(
        folder_path=TEST_FOLDER, model_id=ONNX_TEST_MODEL_ID
    )
    zip_file_path = test_model.save_as_onnx(model_id=ONNX_TEST_MODEL_ID)
    model_config_path = test_model.make_model_config_json(model_format="ONNX")

    with ZipFile(zip_file_path, "r") as f:
        assert set(f.namelist()) == {"paraphrase-MiniLM-L3-v2.onnx", "tokenizer.json"}

    onnx_path = os.path.join(TEST_FOLDER, "onnx", "paraphrase-MiniLM-L3-v2.onnx")
    graph = _load_onnx_graph(onnx_path)
    assert graph.ir_version <= SentenceTransformerModel.ONNX_MAX_IR_VERSION
    assert [i.name for i in graph.graph.input] == [
        "input_ids",
        "attention_mask",
        "token_type_ids",
    ]
    assert [o.name for o in graph.graph.output] == ["token_embeddings"]

    with open(model_config_path) as f:
        model_config = json.load(f)["model_config"]
    assert model_config["pooling_mode"] == "MEAN"

    pytest.importorskip("onnxruntime")
    embeddings = _run_onnx_like_ml_commons(
        onnx_path,
        os.path.join(TEST_FOLDER, "tokenizer.json"),
        model_config,
        ONNX_TEST_SENTENCES,
    )
    _assert_matches_sentence_transformers(ONNX_TEST_MODEL_ID, embeddings)
    clean_test_folder(TEST_FOLDER)


def test_save_as_onnx_pre_pooled_model_with_dense_layer():
    # a Dense module after pooling can't be applied by ml-commons, so the export outputs the final
    # embedding and the config uses pooling_mode NONE without normalization
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import (
        Dense,
        Normalize,
        Pooling,
        Transformer,
    )

    clean_test_folder(TEST_FOLDER)
    local_model_path = os.path.join(TEST_FOLDER, "dense-source-model")
    transformer = Transformer(ONNX_TEST_MODEL_ID)
    pooling = Pooling(transformer.get_embedding_dimension(), pooling_mode="mean")
    dense = Dense(transformer.get_embedding_dimension(), 64)
    SentenceTransformer(
        modules=[transformer, pooling, dense, Normalize()], device="cpu"
    ).save(local_model_path)

    output_folder = os.path.join(TEST_FOLDER, "dense-output")
    test_model = SentenceTransformerModel(
        folder_path=output_folder, model_id=local_model_path
    )
    zip_file_path = test_model.save_as_onnx(model_id=local_model_path)
    model_config_path = test_model.make_model_config_json(
        model_format="ONNX", pooling_mode="MEAN"
    )

    with open(model_config_path) as f:
        model_config = json.load(f)["model_config"]
    assert model_config["pooling_mode"] == "NONE"
    assert model_config["normalize_result"] is False
    assert model_config["embedding_dimension"] == 64

    onnx_path = os.path.join(output_folder, "onnx", "dense-source-model.onnx")
    graph = _load_onnx_graph(onnx_path)
    assert graph.ir_version <= SentenceTransformerModel.ONNX_MAX_IR_VERSION
    assert [o.name for o in graph.graph.output] == ["sentence_embedding"]
    with ZipFile(zip_file_path, "r") as f:
        assert set(f.namelist()) == {"dense-source-model.onnx", "tokenizer.json"}

    pytest.importorskip("onnxruntime")
    embeddings = _run_onnx_like_ml_commons(
        onnx_path,
        os.path.join(output_folder, "tokenizer.json"),
        model_config,
        ONNX_TEST_SENTENCES,
    )
    _assert_matches_sentence_transformers(local_model_path, embeddings)
    clean_test_folder(TEST_FOLDER)


def _save_local_sentence_transformer(
    path, pooling_mode="mean", dense=False, include_prompt=True
):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import (
        Dense,
        Normalize,
        Pooling,
        Transformer,
    )

    transformer = Transformer(ONNX_TEST_MODEL_ID)
    modules = [
        transformer,
        Pooling(
            transformer.get_embedding_dimension(),
            pooling_mode=pooling_mode,
            include_prompt=include_prompt,
        ),
    ]
    if dense:
        modules.append(Dense(transformer.get_embedding_dimension(), 64))
    modules.append(Normalize())
    SentenceTransformer(modules=modules, device="cpu").save(path)
    return path


@pytest.mark.parametrize(
    "pooling_mode, expected_ml_commons_mode",
    [
        ("mean_sqrt_len_tokens", "MEAN_SQRT_LEN"),
        ("weightedmean", "WEIGHTED_MEAN"),
        ("lasttoken", "LAST_TOKEN"),
        ("cls", "CLS"),
    ],
)
def test_save_as_onnx_maps_pooling_modes_to_ml_commons(
    pooling_mode, expected_ml_commons_mode
):
    # pooling modes ml-commons supports keep the per-token export and get ml-commons' name
    clean_test_folder(TEST_FOLDER)
    local_model_path = _save_local_sentence_transformer(
        os.path.join(TEST_FOLDER, "pooling-model"), pooling_mode=pooling_mode
    )
    output_folder = os.path.join(TEST_FOLDER, "pooling-output")
    test_model = SentenceTransformerModel(
        folder_path=output_folder, model_id=local_model_path
    )
    test_model.save_as_onnx(model_id=local_model_path)
    with open(test_model.make_model_config_json(model_format="ONNX")) as f:
        model_config = json.load(f)["model_config"]
    assert model_config["pooling_mode"] == expected_ml_commons_mode
    assert model_config["normalize_result"] is True

    onnx_path = os.path.join(output_folder, "onnx", "pooling-model.onnx")
    assert [o.name for o in _load_onnx_graph(onnx_path).graph.output] == [
        "token_embeddings"
    ]

    pytest.importorskip("onnxruntime")
    embeddings = _run_onnx_like_ml_commons(
        onnx_path,
        os.path.join(output_folder, "tokenizer.json"),
        model_config,
        ONNX_TEST_SENTENCES,
    )
    _assert_matches_sentence_transformers(local_model_path, embeddings)
    clean_test_folder(TEST_FOLDER)


def test_save_as_onnx_max_pooling_is_pre_pooled():
    # ml-commons' ONNX translator mis-normalizes MAX pooling ([1, dim] shape), so the export
    # outputs the final embedding with pooling_mode NONE instead
    clean_test_folder(TEST_FOLDER)
    local_model_path = _save_local_sentence_transformer(
        os.path.join(TEST_FOLDER, "max-model"), pooling_mode="max"
    )
    output_folder = os.path.join(TEST_FOLDER, "max-output")
    test_model = SentenceTransformerModel(
        folder_path=output_folder, model_id=local_model_path
    )
    test_model.save_as_onnx(model_id=local_model_path)
    with open(test_model.make_model_config_json(model_format="ONNX")) as f:
        model_config = json.load(f)["model_config"]
    assert model_config["pooling_mode"] == "NONE"
    assert model_config["normalize_result"] is False

    onnx_path = os.path.join(output_folder, "onnx", "max-model.onnx")
    assert [o.name for o in _load_onnx_graph(onnx_path).graph.output] == [
        "sentence_embedding"
    ]

    pytest.importorskip("onnxruntime")
    embeddings = _run_onnx_like_ml_commons(
        onnx_path,
        os.path.join(output_folder, "tokenizer.json"),
        model_config,
        ONNX_TEST_SENTENCES,
    )
    _assert_matches_sentence_transformers(local_model_path, embeddings)
    clean_test_folder(TEST_FOLDER)


def test_make_model_config_json_with_query_prefix():
    clean_test_folder(TEST_FOLDER)
    test_model = SentenceTransformerModel(
        folder_path=TEST_FOLDER, model_id=ONNX_TEST_MODEL_ID
    )
    test_model.save_as_onnx(model_id=ONNX_TEST_MODEL_ID)
    query_prefix = (
        "Instruct: Given a web search query, retrieve relevant passages\nQuery:"
    )
    with open(
        test_model.make_model_config_json(
            model_format="ONNX", query_prefix=query_prefix
        )
    ) as f:
        model_config = json.load(f)["model_config"]
    assert model_config["query_prefix"] == query_prefix

    # no prefix field unless given
    with open(test_model.make_model_config_json(model_format="ONNX")) as f:
        model_config = json.load(f)["model_config"]
    assert "query_prefix" not in model_config
    clean_test_folder(TEST_FOLDER)


def test_query_prefix_matches_sentence_transformers_prompt():
    # with include_prompt=True, prepending query_prefix (as ml-commons does) equals
    # sentence-transformers' prompt encoding
    clean_test_folder(TEST_FOLDER)
    query_prefix = (
        "Instruct: Given a web search query, retrieve relevant passages\nQuery:"
    )
    local_model_path = _save_local_sentence_transformer(
        os.path.join(TEST_FOLDER, "prompt-model"), include_prompt=True
    )
    output_folder = os.path.join(TEST_FOLDER, "prompt-output")
    test_model = SentenceTransformerModel(
        folder_path=output_folder, model_id=local_model_path
    )
    test_model.save_as_onnx(model_id=local_model_path)
    with open(
        test_model.make_model_config_json(
            model_format="ONNX", query_prefix=query_prefix
        )
    ) as f:
        model_config = json.load(f)["model_config"]
    assert model_config["query_prefix"] == query_prefix

    pytest.importorskip("onnxruntime")
    embeddings = _run_onnx_like_ml_commons(
        os.path.join(output_folder, "onnx", "prompt-model.onnx"),
        os.path.join(output_folder, "tokenizer.json"),
        model_config,
        [query_prefix + sentence for sentence in ONNX_TEST_SENTENCES],
    )
    _assert_matches_sentence_transformers(
        local_model_path, embeddings, prompt=query_prefix
    )
    clean_test_folder(TEST_FOLDER)


def test_query_prefix_rejected_when_pooling_excludes_prompt():
    # ml-commons can't exclude prompt tokens from pooling, so this combination is rejected
    clean_test_folder(TEST_FOLDER)
    local_model_path = _save_local_sentence_transformer(
        os.path.join(TEST_FOLDER, "no-prompt-model"), include_prompt=False
    )
    output_folder = os.path.join(TEST_FOLDER, "no-prompt-output")
    test_model = SentenceTransformerModel(
        folder_path=output_folder, model_id=local_model_path
    )
    test_model.save_as_onnx(model_id=local_model_path)
    with pytest.raises(ValueError, match="include_prompt=False"):
        test_model.make_model_config_json(model_format="ONNX", query_prefix="query: ")
    # without a prefix the model is still supported
    test_model.make_model_config_json(model_format="ONNX")
    clean_test_folder(TEST_FOLDER)


def test_save_as_onnx_zips_external_data_for_large_models(monkeypatch):
    # models over the single-file limit keep weights in <model>.onnx.data, which must be zipped
    monkeypatch.setattr(SentenceTransformerModel, "ONNX_SINGLE_FILE_LIMIT_BYTES", 0)
    clean_test_folder(TEST_FOLDER)
    test_model = SentenceTransformerModel(
        folder_path=TEST_FOLDER, model_id=ONNX_TEST_MODEL_ID
    )
    zip_file_path = test_model.save_as_onnx(model_id=ONNX_TEST_MODEL_ID)

    with ZipFile(zip_file_path, "r") as f:
        assert set(f.namelist()) == {
            "paraphrase-MiniLM-L3-v2.onnx",
            "paraphrase-MiniLM-L3-v2.onnx.data",
            "tokenizer.json",
        }
    graph = _load_onnx_graph(
        os.path.join(TEST_FOLDER, "onnx", "paraphrase-MiniLM-L3-v2.onnx")
    )
    assert graph.ir_version <= SentenceTransformerModel.ONNX_MAX_IR_VERSION

    pytest.importorskip("onnxruntime")
    with open(test_model.make_model_config_json(model_format="ONNX")) as f:
        model_config = json.load(f)["model_config"]
    embeddings = _run_onnx_like_ml_commons(
        os.path.join(TEST_FOLDER, "onnx", "paraphrase-MiniLM-L3-v2.onnx"),
        os.path.join(TEST_FOLDER, "tokenizer.json"),
        model_config,
        ONNX_TEST_SENTENCES,
    )
    _assert_matches_sentence_transformers(ONNX_TEST_MODEL_ID, embeddings)
    clean_test_folder(TEST_FOLDER)


clean_test_folder(TEST_FOLDER)
clean_test_folder(TESTDATA_UNZIP_FOLDER)
