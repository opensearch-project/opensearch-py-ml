# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

# This program is run by "Model Listing Uploading" workflow
# (See model_listing_uploader.yml) to update pretrained_model_listing.json.

import argparse
import json
import os
from typing import Optional

JSON_FILENAME = "pretrained_model_listing.json"
JSON_DIRNAME = "utils/model_uploader/model_listing"
PRETRAINED_MODEL_LISTING_JSON_FILEPATH = os.path.join(JSON_DIRNAME, JSON_FILENAME)
PREFIX_SENTENCE_TRANSFORMER_FILEPATH = "ml-models/huggingface/sentence-transformers"
TORCH_SCRIPT_FORMAT = "TORCH_SCRIPT"
ONNX_FORMAT = "ONNX"
TEMP_MODEL_PATH = "temp_model_path"
TEST_SENTENCES = [
    "First test sentence",
    "Second test sentence",
]


def get_sentence_transformer_model_description(
    config_folder_name: str, config_filepath: str
) -> Optional[str]:
    """
    Get description of the pretrained sentence transformer model from config file

    :param config_folder_name: Name of the local folder that stores config files (e.g. 'config_folder')
    :type config_folder_name: string
    :param config_filepath: Path to local config file
    (e.g. 'sentence-transformers/all-MiniLM-L12-v2/2.0.0/onnx/config.json')
    :type config_filepath: string
    :return: Description of the model
    :rtype: string or None
    """
    filepath = os.path.join(config_folder_name, config_filepath)
    try:
        with open(filepath, "r") as f:
            model_config = json.load(f)
    except Exception as e:
        print(f"Cannot open {filepath} to get model description: {e}")
        return None
    if "description" in model_config:
        return model_config["description"]
    else:
        return None


def create_new_pretrained_model_listing(
    config_paths_txt_filename: str, config_foldername: str
):
    """
    Create a new pretrained model listing and store it at PRETRAINED_MODEL_LISTING_JSON_FILEPATH
    based on current models in config_paths_txt_filename and their config files in config_foldername

    :param config_paths_txt_filename: Name of the txt file that stores paths to config file
    in the ml-models/huggingface/ folder of the S3 bucket
    :type config_paths_txt_filename: string
    :param config_foldername: Name of the local folder that stores config files
    :type config_foldername: string
    :return: No return value expected
    :rtype: None
    """
    print("\n=== Begin running update_pretrained_model_listing.py ===")
    print(f"--- Reading {config_paths_txt_filename} ---")
    with open(config_paths_txt_filename, "r") as f:
        config_paths_lst = f.read().split()

    print("---  Creating New Model Listing --- ")
    new_model_listing_dict = {}
    for config_filepath in config_paths_lst:
        if config_filepath.startswith(PREFIX_SENTENCE_TRANSFORMER_FILEPATH):
            # (e.g. 'ml-models/huggingface/sentence-transformers/all-MiniLM-L12-v2/2.0.0/onnx/config.json')
            model_parts = config_filepath.split("/")
            model_name = "/".join(model_parts[1:4])
            model_version = model_parts[4]
            model_format = model_parts[5]
            local_config_filepath = "/".join(model_parts[2:])
            if model_name not in new_model_listing_dict:
                new_model_listing_dict[model_name] = {
                    "name": model_name,
                    "versions": {},
                }
            versions_content = new_model_listing_dict[model_name]["versions"]
            if model_version not in versions_content:
                versions_content[model_version] = {
                    "format": [],
                }
            versions_content[model_version]["format"].append(model_format)
            if "description" not in versions_content[model_version]:
                description = get_sentence_transformer_model_description(
                    config_foldername, local_config_filepath
                )
                if description is not None:
                    versions_content[model_version]["description"] = description

    new_model_listing_lst = list(new_model_listing_dict.values())
    new_model_listing_lst = sorted(new_model_listing_lst, key=lambda d: d["name"])
    for model_dict in new_model_listing_lst:
        model_dict["versions"] = dict(sorted(model_dict["versions"].items()))

    print(
        f"---  Dumping New Model Listing in {PRETRAINED_MODEL_LISTING_JSON_FILEPATH} --- "
    )
    if not os.path.isdir(JSON_DIRNAME):
        os.makedirs(JSON_DIRNAME)
    with open(PRETRAINED_MODEL_LISTING_JSON_FILEPATH, "w") as f:
        json.dump(new_model_listing_lst, f, indent=2)
    print("\n=== Finished running update_pretrained_model_listing.py ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config_paths_txt_filename",
        type=str,
        help="Name of the file that stores config paths in S3",
    )
    parser.add_argument(
        "config_foldername",
        type=str,
        help="Name of the local folder that stores copies of config files from S3",
    )
    
    args = parser.parse_args()

    if not args.config_paths_txt_filename.endswith(
        ".txt"
    ):
        assert False, "Invalid arguments"

    create_new_pretrained_model_listing(
        args.config_paths_txt_filename,
        args.config_foldername,
    )
