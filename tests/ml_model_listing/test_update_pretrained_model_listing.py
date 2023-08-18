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
SAMPLE_PRETRAINED_MODEL_LISTING = os.path.join(
    SAMPLE_FOLDER, "pretrained_model_listing.json"
)
SAMPLE_FOLDER_COPY = os.path.join(THIS_DIR, "samples_copy")
SAMPLE_SUBFOLDERNAME = "sentence-transformers"
TEST_FILE = os.path.join(THIS_DIR, "test_pretrained_model_listing.json")

from update_pretrained_model_listing import create_new_pretrained_model_listing


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
        create_new_pretrained_model_listing(
            os.path.join(SAMPLE_FOLDER, CONFIG_PATHS_TXT_FILENAME),
            os.path.join(SAMPLE_FOLDER, CONFIG_FOLDERNAME),
            pretrained_model_listing_json_filepath=TEST_FILE,
        )
    except Exception as e:
        assert False, print(f"Failed while creating new pretrained model listing: {e}")

    try:
        with open(SAMPLE_PRETRAINED_MODEL_LISTING, "r") as f:
            expected_pretrained_model_listing = json.load(f)
    except Exception as e:
        assert False, print(
            f"Cannot open {SAMPLE_PRETRAINED_MODEL_LISTING} to use it for verification: {e}"
        )

    try:
        with open(TEST_FILE, "r") as f:
            test_pretrained_model_listing = json.load(f)
    except Exception as e:
        assert False, print(f"Cannot open {TEST_FILE} to verify its content: {e}")

    assert test_pretrained_model_listing == expected_pretrained_model_listing, print(
        "Incorrect pretrained model listing"
    )

    clean_test_file()


def test_missing_config_file():
    clean_test_file()
    clean_samples_folder_copy()

    # Delete a subfolder that contains config files
    copy_samples_folder()
    shutil.rmtree(
        os.path.join(
            SAMPLE_FOLDER_COPY, CONFIG_FOLDERNAME, SAMPLE_SUBFOLDERNAME
        )
    )

    # Expect create_new_pretrained_model_listing to raise error
    with pytest.raises(Exception) as exc_info:
        create_new_pretrained_model_listing(
            os.path.join(SAMPLE_FOLDER_COPY, CONFIG_PATHS_TXT_FILENAME),
            os.path.join(SAMPLE_FOLDER_COPY, CONFIG_FOLDERNAME),
            pretrained_model_listing_json_filepath=TEST_FILE,
        )
    assert exc_info.type is Exception
    assert "Cannot open" in str(exc_info.value)

    clean_test_file()
    clean_samples_folder_copy()
    

def test_missing_model_description():
    clean_test_file()
    clean_samples_folder_copy()

    copy_samples_folder()
    path_to_sample_file = os.path.join(
        SAMPLE_FOLDER_COPY, CONFIG_FOLDERNAME, sample_file_path
    )
    
    # Remove description from the following model description
    sample_model_id = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    sample_model_name_in_config = f"huggingface/{sample_model_id}"
    sample_version = "1.0.1"
    sample_format = "torch_script"
    sample_file_path = f"{sample_model_id}/{sample_version}/{sample_format}/config.json"
    
    with open(path_to_sample_file, 'r') as f:
        sample_config = json.load(f)
    sample_config.pop('description')
    with open(path_to_sample_file, 'w') as f:
        json.dump(sample_config, f)
        
    # Expect create_new_pretrained_model_listing not to raise error
    try:
        create_new_pretrained_model_listing(
            os.path.join(SAMPLE_FOLDER_COPY, CONFIG_PATHS_TXT_FILENAME),
            os.path.join(SAMPLE_FOLDER_COPY, CONFIG_FOLDERNAME),
            pretrained_model_listing_json_filepath=TEST_FILE,
        )
    except Exception as e:
        assert False, print(f"Failed while creating new pretrained model listing: {e}")

    # Create expected_pretrained_model_listing
    path_to_pretrained_model_listing = os.path.join(
        SAMPLE_FOLDER_COPY, "pretrained_model_listing.json"
    )
    try:
        with open(path_to_pretrained_model_listing, "r") as f:
            expected_pretrained_model_listing = json.load(f)
    except Exception as e:
        assert False, print(
            f"Cannot open {path_to_pretrained_model_listing} to use it for verification: {e}"
        )
    for dict_obj in expected_pretrained_model_listing:
        if dict_obj["name"] == sample_model_name_in_config:
            dict_obj["versions"][sample_version].pop('description')
            break
    
    # Compare the generated test_pretrained_model_listing with expected_pretrained_model_listing
    try:
        with open(TEST_FILE, "r") as f:
            test_pretrained_model_listing = json.load(f)
    except Exception as e:
        assert False, print(f"Cannot open {TEST_FILE} to verify its content: {e}")

    assert test_pretrained_model_listing == expected_pretrained_model_listing, print(
        "Incorrect pretrained model listing given that description is missing in some model"
    )
    
    clean_test_file()
    clean_samples_folder_copy()
        

clean_samples_folder_copy()
clean_test_file()
