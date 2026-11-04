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
SAMPLE_MISSING_CONFIG_SUBFOLDERNAME = "ml-models/huggingface/sentence-transformers"

from update_pretrained_model_listing import main as update_pretrained_model_listing_main


def test_create_new_pretrained_model_listing(tmp_path):
    test_file = str(tmp_path / "test_pretrained_model_listing.json")
    try:
        update_pretrained_model_listing_main(
            [
                os.path.join(SAMPLE_FOLDER, CONFIG_PATHS_TXT_FILENAME),
                os.path.join(SAMPLE_FOLDER, CONFIG_FOLDERNAME),
                "--pretrained_model_listing_json_filepath",
                test_file,
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
        with open(test_file, "r") as f:
            test_pretrained_model_listing = json.load(f)
    except Exception as e:
        assert False, print(f"Cannot open {test_file} to verify its content: {e}")

    assert test_pretrained_model_listing == sample_pretrained_model_listing, print(
        "Incorrect pretrained model listing"
    )


def test_missing_config_file(tmp_path):
    test_file = str(tmp_path / "test_pretrained_model_listing.json")
    samples_folder_copy = tmp_path / "samples_copy"
    shutil.copytree(SAMPLE_FOLDER, samples_folder_copy)
    shutil.rmtree(
        os.path.join(
            samples_folder_copy, CONFIG_FOLDERNAME, SAMPLE_MISSING_CONFIG_SUBFOLDERNAME
        )
    )

    with pytest.raises(Exception) as exc_info:
        update_pretrained_model_listing_main(
            [
                os.path.join(samples_folder_copy, CONFIG_PATHS_TXT_FILENAME),
                os.path.join(samples_folder_copy, CONFIG_FOLDERNAME),
                "--pretrained_model_listing_json_filepath",
                test_file,
            ]
        )
    assert exc_info.type is Exception
    assert "Cannot open" in str(exc_info.value)
