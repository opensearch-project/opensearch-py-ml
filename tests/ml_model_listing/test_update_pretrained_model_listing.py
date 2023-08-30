# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

# We need to append UTILS_MODEL_UPLOADER_DIR path so that we can import
# functions from update_pretrained_model_listing.py
# since this python script is not in the root directory.

import json
import os
import shutil
import sys

import pytest

THIS_DIR = os.path.dirname(__file__)
UTILS_MODEL_UPLOADER_DIR = os.path.join(THIS_DIR, "../../utils/model_uploader")
sys.path.append(UTILS_MODEL_UPLOADER_DIR)

SAMPLE_FOLDER = os.path.join(THIS_DIR, "samples")
CONFIG_PATHS_TXT_FILENAME = "config_paths.txt"
CONFIG_FOLDERNAME = "config_folder"
EXCLUDED_MODELS_TXT_FILENAME = "excluded_models.txt"
SAMPLE_PRETRAINED_MODEL_LISTING = os.path.join(
    SAMPLE_FOLDER, "pretrained_model_listing.json"
)
SAMPLE_FOLDER_COPY = os.path.join(THIS_DIR, "samples_copy")
SAMPLE_MISSING_CONFIG_SUBFOLDERNAME = "ml-models/huggingface/sentence-transformers"
TEST_FILE = os.path.join(THIS_DIR, "test_pretrained_model_listing.json")

from update_pretrained_model_listing import main as update_pretrained_model_listing_main


def clean_test_file():
    if os.path.isfile(TEST_FILE):
        os.remove(TEST_FILE)


def copy_samples_folder():
    shutil.copytree(SAMPLE_FOLDER, SAMPLE_FOLDER_COPY)


def clean_samples_folder_copy():
    if os.path.exists(SAMPLE_FOLDER_COPY):
        for files in os.listdir(SAMPLE_FOLDER_COPY):
            sub_path = os.path.join(SAMPLE_FOLDER_COPY, files)
            if os.path.isfile(sub_path):
                os.remove(sub_path)
            else:
                try:
                    shutil.rmtree(sub_path)
                except OSError as err:
                    print(
                        "Fail to delete files, please delete all files in "
                        + str(SAMPLE_FOLDER_COPY)
                        + " "
                        + str(err)
                    )

        shutil.rmtree(SAMPLE_FOLDER_COPY)


clean_samples_folder_copy()
clean_test_file()


def test_create_new_pretrained_model_listing():
    clean_test_file()
    try:
        update_pretrained_model_listing_main(
            [
                os.path.join(SAMPLE_FOLDER, CONFIG_PATHS_TXT_FILENAME),
                os.path.join(SAMPLE_FOLDER, CONFIG_FOLDERNAME),
                "--pretrained_model_listing_json_filepath",
                TEST_FILE,
                "--excluded_models_txt_filepath",
                os.path.join(SAMPLE_FOLDER, EXCLUDED_MODELS_TXT_FILENAME),
            ]
        )
    except Exception as e:
        assert False, print(f"Failed while creating new pretrained model listing: {e}")

    try:
        with open(SAMPLE_PRETRAINED_MODEL_LISTING, "r") as f:
            sample_pretrained_model_listing = json.load(f)
    except Exception as e:
        assert False, print(
            f"Cannot open {SAMPLE_PRETRAINED_MODEL_LISTING} to use it for verification: {e}"
        )

    try:
        with open(TEST_FILE, "r") as f:
            test_pretrained_model_listing = json.load(f)
    except Exception as e:
        assert False, print(f"Cannot open {TEST_FILE} to verify its content: {e}")

    assert test_pretrained_model_listing == sample_pretrained_model_listing, print(
        "Incorrect pretrained model listing"
    )

    clean_test_file()


def test_missing_config_file():
    clean_test_file()
    clean_samples_folder_copy()

    copy_samples_folder()
    shutil.rmtree(
        os.path.join(
            SAMPLE_FOLDER_COPY, CONFIG_FOLDERNAME, SAMPLE_MISSING_CONFIG_SUBFOLDERNAME
        )
    )

    with pytest.raises(Exception) as exc_info:
        update_pretrained_model_listing_main(
            [
                os.path.join(SAMPLE_FOLDER_COPY, CONFIG_PATHS_TXT_FILENAME),
                os.path.join(SAMPLE_FOLDER_COPY, CONFIG_FOLDERNAME),
                "--pretrained_model_listing_json_filepath",
                TEST_FILE,
                "--excluded_models_txt_filepath",
                os.path.join(SAMPLE_FOLDER, EXCLUDED_MODELS_TXT_FILENAME),
            ]
        )
    assert exc_info.type is Exception
    assert "Cannot open" in str(exc_info.value)

    clean_test_file()
    clean_samples_folder_copy()


clean_samples_folder_copy()
clean_test_file()
