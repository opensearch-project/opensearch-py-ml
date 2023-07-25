# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

# This program is run by "Model Auto-tracing & Uploading" workflow
# (See model_uploader.yml) to verify if the model already exists in
# model hub before continuing the workflow.

import argparse
import re

MODEL_ID_START_PATTERN = "sentence-transformers/"
VERSION_PATTERN = r"^([1-9]\d*|0)(\.(([1-9]\d*)|0)){0,3}$"


def verify_inputs(model_id: str, model_version: str) -> None:
    """
    Verify the format of model_id and model_version

    :param model_id: Model ID of the pretrained model
    :type model_id: string
    :param model_version: Version of the pretrained model for registration
    :type model_version: string
    """
    if not model_id.startswith(MODEL_ID_START_PATTERN):
        assert False, f"Invalid Model ID: {model_id}"
    if re.fullmatch(VERSION_PATTERN, model_version) is None:
        assert False, f"Invalid Model Version: {model_version}"


def get_model_file_path(model_folder: str, model_id: str, model_version: str, model_format: str) -> str:
    """
    Construct the expected model file path on model hub

    :param model_folder: Model folder for uploading
    :type model_folder: string
    :param model_id: Model ID of the pretrained model
    :type model_id: string
    :param model_version: Version of the pretrained model for registration
    :type model_version: string
    :param model_format: Model format ("TORCH_SCRIPT" or "ONNX")
    :type model_format: string
    """
    model_name = str(model_id.split("/")[-1])
    model_format = model_format.lower()
    model_dirname = f"{model_folder}{model_name}/{model_version}/{model_format}"
    model_filename = (
        f"sentence-transformers_{model_name}-{model_version}-{model_format}.zip"
    )
    model_file_path = model_dirname + "/" + model_filename
    return model_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_folder",
        type=str,
        help="Model folder for uploading (e.g. ml-models/huggingface/sentence-transformers/)",
    )
    parser.add_argument(
        "model_id",
        type=str,
        help="Model ID for auto-tracing and uploading (e.g. sentence-transformers/msmarco-distilbert-base-tas-b)",
    )
    parser.add_argument(
        "model_version", type=str, help="Model version number (e.g. 1.0.1)"
    )
    parser.add_argument(
        "model_format",
        choices=["TORCH_SCRIPT", "ONNX"],
        help="Model format for auto-tracing",
    )

    args = parser.parse_args()
    verify_inputs(args.model_id, args.model_version)
    model_file_path = get_model_file_path(
        args.model_folder, args.model_id, args.model_version, args.model_format
    )

    # Print the model file path so that the workflow can store it in the variable (See model_uploader.yml)
    print(model_file_path)
