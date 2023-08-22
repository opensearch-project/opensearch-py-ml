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
import sys
from typing import Optional

JSON_FILENAME = "pretrained_model_listing.json"
JSON_DIRNAME = "utils/model_uploader/model_listing"
PRETRAINED_MODEL_LISTING_JSON_FILEPATH = os.path.join(JSON_DIRNAME, JSON_FILENAME)


def get_sentence_transformer_model_description(
    config_folderpath: str, config_filepath: str
) -> Optional[str]:
    """
    Get description of the pretrained sentence transformer model from config file

    :param config_folderpath: Path to the folder that stores copies of config files from S3 (e.g. 'config_folder')
    :type config_folderpath: string
    :param config_filepath: Path to local config file
    (e.g. 'sentence-transformers/all-MiniLM-L12-v2/2.0.0/onnx/config.json')
    :type config_filepath: string
    :return: Description of the model
    :rtype: string or None
    """
    filepath = os.path.join(config_folderpath, config_filepath)
    try:
        with open(filepath, "r") as f:
            model_config = json.load(f)
    except Exception as e:
        raise Exception(f"Cannot open {filepath} to get model description: {e}")
    if "description" in model_config:
        return model_config["description"]
    else:
        return None


def create_new_pretrained_model_listing(
    config_paths_txt_filepath: str,
    config_folderpath: str,
    pretrained_model_listing_json_filepath: str = PRETRAINED_MODEL_LISTING_JSON_FILEPATH,
):
    """
    Create a new pretrained model listing and store it at pretrained_model_listing_json_filepath
    based on current models in config_paths_txt_filepath and their config files in config_folderpath

    :param config_paths_txt_filepath: Path to the txt file that stores a list of config paths from S3
    in the ml-models/huggingface/ folder of the S3 bucket
    :type config_paths_txt_filepath: string
    :param config_folderpath: Path to the folder that stores copies of config files from S3
    :type config_folderpath: string
    :return: No return value expected
    :param pretrained_model_listing_json_filepath: Path to the json file that stores new model listing
    :rtype: None
    """
    print("\n=== Begin running update_pretrained_model_listing.py ===")
    print(f"--- Reading {config_paths_txt_filepath} ---")
    with open(config_paths_txt_filepath, "r") as f:
        config_paths_lst = f.read().split()

    print("\n---  Creating New Model Listing --- ")
    new_model_listing_dict = {}
    for config_filepath in config_paths_lst:
        # (e.g. 'ml-models/huggingface/sentence-transformers/all-MiniLM-L12-v2/2.0.0/onnx/config.json')
        model_parts = config_filepath.split("/")
        model_name = "/".join(model_parts[1:4])
        model_version = model_parts[4]
        model_format = model_parts[5]
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
                config_folderpath, config_filepath
            )
            if description is not None:
                versions_content[model_version]["description"] = description

    new_model_listing_lst = list(new_model_listing_dict.values())
    new_model_listing_lst = sorted(new_model_listing_lst, key=lambda d: d["name"])
    for model_dict in new_model_listing_lst:
        model_dict["versions"] = dict(sorted(model_dict["versions"].items()))

    print(
        f"\n---  Dumping New Model Listing in {pretrained_model_listing_json_filepath} --- "
    )
    if not os.path.isdir(JSON_DIRNAME):
        os.makedirs(JSON_DIRNAME)
    with open(pretrained_model_listing_json_filepath, "w") as f:
        json.dump(new_model_listing_lst, f, indent=2)
    print("\n=== Finished running update_pretrained_model_listing.py ===")


def main(args):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config_paths_txt_filepath",
        type=str,
        help="Path to the txt file that stores a list of config paths from S3",
    )
    parser.add_argument(
        "config_folderpath",
        type=str,
        help="Path to the folder that stores copies of config files from S3",
    )
    parser.add_argument(
        "-fp",
        "--pretrained_model_listing_json_filepath",
        type=str,
        default=PRETRAINED_MODEL_LISTING_JSON_FILEPATH,
        help="Path to the json file that stores new model listing",
    )

    parsed_args = parser.parse_args(args)

    if not parsed_args.config_paths_txt_filepath.endswith(".txt"):
        raise Exception(
            "Invalid argument: config_paths_txt_filepath should be .txt file"
        )

    create_new_pretrained_model_listing(
        parsed_args.config_paths_txt_filepath,
        parsed_args.config_folderpath,
        parsed_args.pretrained_model_listing_json_filepath,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
