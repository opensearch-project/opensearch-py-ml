# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import argparse
import json
import os
import shutil
import sys

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(THIS_DIR, "../..")
sys.path.append(ROOT_DIR)

from opensearch_py_ml.ml_models.sentencetransformermodel import SentenceTransformerModel

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


def get_sentence_transformer_model_description(model_name, model_format) -> str:
    """
    Get description of the pretrained sentence transformer model

    :param model_name: Model name of the pretrained model
    (e.g. huggingface/sentence-transformers/msmarco-distilbert-base-tas-b)
    :type model_name: string
    :param model_format: Model format of the pretrained model (TORCH_SCRIPT/ONNX)
    :type model_format: string
    :return: Description of the model
    :rtype: string
    """
    model_id = model_name[len("huggingface/") :]
    pre_trained_model = SentenceTransformerModel(
        model_id=model_id, folder_path=TEMP_MODEL_PATH
    )
    if model_format == TORCH_SCRIPT_FORMAT:
        pre_trained_model.save_as_pt(model_id=model_id, sentences=TEST_SENTENCES)
    else:
        pre_trained_model.save_as_onnx(model_id=model_id)

    description = None
    readme_file_path = os.path.join(TEMP_MODEL_PATH, "README.md")
    if os.path.exists(readme_file_path):
        try:
            description = pre_trained_model.get_model_description_from_md_file(
                readme_file_path
            )
        except Exception as e:
            print(f"Cannot get model description from README.md file: {e}")
    try:
        shutil.rmtree(TEMP_MODEL_PATH)
    except Exception as e:
        assert False, f"Raised Exception while deleting {TEMP_MODEL_PATH}: {e}"
    return description


def create_new_pretrained_model_listing(models_txt_filename, old_json_filename):
    """
    Create a new pretrained model listing and store it at PRETRAINED_MODEL_LISTING_JSON_FILEPATH
    based on current models in models_txt_filename and the old pretrained model
    listing in old_json_filename

    :param models_txt_filename: Name of the txt file that stores string of
    directories in the ml-models/huggingface/ folder of the S3 bucket
    :type models_txt_filename: string
    :param old_json_filename: Name of the json file that contains the old pretrained
    model listing
    :type old_json_filename: string
    :return: No return value expected
    :rtype: None
    """
    print("\n=== Begin running update_pretrained_model_listing.py ===")
    print(f"--- Reading {models_txt_filename} ---")
    with open(models_txt_filename, "r") as f:
        model_lst = f.read().split()
        model_lst = list(
            set(
                [
                    model_filepath[: model_filepath.rfind("/")]
                    for model_filepath in model_lst
                ]
            )
        )

    print(f"--- Reading {old_json_filename} --- ")
    with open(old_json_filename, "r") as f:
        old_model_listing_lst = json.load(f)

    old_model_listing_dict = {
        model_data["name"]: model_data for model_data in old_model_listing_lst
    }

    print("---  Creating New Model Listing --- ")
    new_model_listing_dict = {}
    for model_filepath in model_lst:
        if model_filepath.startswith(PREFIX_SENTENCE_TRANSFORMER_FILEPATH):
            # (e.g. ml-models/huggingface/sentence-transformers/msmarco-distilbert-base-tas-b/1.0.1/torch_script)
            model_parts = model_filepath.split("/")
            model_name = "/".join(model_parts[1:4])
            model_version = model_parts[4]
            model_format = model_parts[5]
            if model_name not in new_model_listing_dict:
                new_model_listing_dict[model_name] = {
                    "name": model_name,
                    "version": [],
                    "format": [],
                }
            model_content = new_model_listing_dict[model_name]
            if model_version not in model_content["version"]:
                model_content["version"].append(model_version)
            if model_format not in model_content["format"]:
                model_content["format"].append(model_format)
            if model_name in old_model_listing_dict:
                if "description" in old_model_listing_dict[model_name]:
                    model_content["description"] = old_model_listing_dict[model_name][
                        "description"
                    ]
            else:
                try:
                    description = get_sentence_transformer_model_description(
                        model_name, model_format
                    )
                except Exception as e:
                    description = None
                    print(f"Cannot get sentence transformer model description: {e}")
                if description is not None:
                    model_content["description"] = description

    new_model_listing_lst = list(new_model_listing_dict.values())

    print(f"---  Dumping New Model Listing in {PRETRAINED_MODEL_LISTING_JSON_FILEPATH} --- ")
    if not os.path.isdir(JSON_DIRNAME):
        os.makedirs(JSON_DIRNAME)
    with open(PRETRAINED_MODEL_LISTING_JSON_FILEPATH, "w") as f:
        json.dump(new_model_listing_lst, f, indent=1)
    print("\n=== Finished running update_pretrained_model_listing.py ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models_txt_filename",
        type=str,
        help="Name of the file that stores model names",
    )
    parser.add_argument(
        "old_json_filename",
        type=str,
        help="Name of the file that stores the old version of the listing of pretrained models",
    )

    args = parser.parse_args()

    if not args.models_txt_filename.endswith(
        ".txt"
    ) or not args.old_json_filename.endswith(".json"):
        assert False, "Invalid arguments"

    create_new_pretrained_model_listing(
        args.models_txt_filename, args.old_json_filename
    )
