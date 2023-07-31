# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import argparse
import json
import re
import shutil

from mdutils.fileutils import MarkDownFile
from sentence_transformers import SentenceTransformer

PREFIX_SENTENCE_TRANSFORMER_FILEPATH = "ml-models/huggingface/sentence-transformers"
TEMP_MODEL_PATH = "temp_model_path"


def get_description_from_md_file(model_id: str) -> str:
    """
    Get description of the model from README.md file
    after the model is saved in local directory

    :param model_id: Model ID of the pretrained model
    (e.g. sentence-transformers/msmarco-distilbert-base-tas-b)
    :type model_id: string
    :return: Description of the model
    :rtype: string
    """
    readme_data = MarkDownFile.read_file(TEMP_MODEL_PATH + "/" + "README.md")
    start_str = f"# {model_id}"
    start = readme_data.find(start_str) + len(start_str) + 1
    end = readme_data.find("## ", start)
    if start == -1 or end == -1:
        assert False, "Cannot get description from model's README.md"

    description = readme_data[start:end].strip()
    description = re.sub(r"\(.*?\)", "", description)
    description = re.sub(r"[\[\]]", "", description)
    return description


def get_sentence_transformer_model_description(model_name) -> str:
    """
    Get description of the pretrained sentence transformer model

    :param model_name: Model name of the pretrained model
    (e.g. huggingface/sentence-transformers/msmarco-distilbert-base-tas-b)
    :type model_name: string
    :return: Description of the model
    :rtype: string
    """
    model_id = model_name[len("huggingface/") :]
    pretrained_model = SentenceTransformer(model_id)
    pretrained_model.save(path=TEMP_MODEL_PATH)
    description = get_description_from_md_file(model_id)
    try:
        shutil.rmtree(TEMP_MODEL_PATH)
    except Exception as e:
        assert False, f"Raised Exception while deleting {TEMP_MODEL_PATH}: {e}"
    return description


def create_new_pretrained_model_listing(
    models_txt_filename, old_json_filename, new_json_filename
):
    """
    Create a new pretrained model listing and store it at new_json_filename
    based on current models in models_txt_filename and the old pretrained model
    listing in old_json_filename

    :param models_txt_filename: Name of the txt file that stores string of
    directories in the ml-models/huggingface/ folder of the S3 bucket
    :type models_txt_filename: string
    :param old_json_filename: Name of the json file that contains the old pretrained
    model listing
    :type old_json_filename: string
    :param new_json_filename: Name of the json file that the new model listing will
    be stored at
    :type new_json_filename: string
    :return: No return value expected
    :rtype: None
    """
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

    with open(old_json_filename, "r") as f:
        old_model_listing_lst = json.load(f)

    old_model_listing_dict = {
        model_data["name"]: model_data for model_data in old_model_listing_lst
    }

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
                    "description": "",
                }
            model_content = new_model_listing_dict[model_name]
            if model_version not in model_content["version"]:
                model_content["version"].append(model_version)
            if model_format not in model_content["format"]:
                model_content["format"].append(model_format)
            if model_name in old_model_listing_dict:
                model_content["description"] = old_model_listing_dict[model_name][
                    "description"
                ]
            else:
                model_content[
                    "description"
                ] = get_sentence_transformer_model_description(model_name)

    new_model_listing_lst = list(new_model_listing_dict.values())
    with open(new_json_filename, "w") as f:
        json.dump(new_model_listing_lst, f, indent=1)


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
    parser.add_argument(
        "new_json_filename",
        type=str,
        help="Name of the file that stores the new version of the listing of pretrained models",
    )
    args = parser.parse_args()

    if (
        not args.models_txt_filename.endswith(".txt")
        or not args.old_json_filename.endswith(".json")
        or not args.new_json_filename.endswith(".json")
    ):
        assert False, "Invalid arguments"

    create_new_pretrained_model_listing(
        args.models_txt_filename, args.old_json_filename, args.new_json_filename
    )
