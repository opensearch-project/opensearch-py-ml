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
import json
import re
from pathlib import Path

VERSION_PATTERN = r"^([1-9]\d*|0)(\.(([1-9]\d*)|0)){0,3}$"
UPLOAD_PREFIX_KEY = "upload_prefix"
MODEL_NAME_KEY = "model_name"


def verify_inputs(model_id: str, model_version: str) -> None:
    """
    Verify the format of model_id and model_version

    :param model_id: Model ID of the pretrained model
    :type model_id: string
    :param model_version: Version of the pretrained model for registration
    :type model_version: string
    :return: No return value expected
    :rtype: None
    """
    # Skip validation for metrics correlation
    if model_id == "metrics_correlation":
        return

    assert model_id.count("/") == 1, f"Invalid Model ID: {model_id}"
    assert (
        re.fullmatch(VERSION_PATTERN, model_version) is not None
    ), f"Invalid Model Version: {model_version}"


def get_model_file_path(
    model_folder: str,
    model_id: str,
    model_version: str,
    model_format: str,
    custom_params: dict = {},
) -> str:
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
    :param custom_params: Custom params for model folder and name
    :type custom_params: dict
    :return: Expected model file path on model hub
    :rtype: string
    """
    if model_id == "metrics_correlation":
        model_type = "amazon"
        model_file_path = str(
            Path(model_folder)
            / model_version
            / model_format.lower()
            / f"{model_type}_{model_id}-{model_version}-{model_format.lower()}.zip"
        )
        return model_file_path
    else:
        model_type, model_name = model_id.split("/")
    if UPLOAD_PREFIX_KEY in custom_params:
        model_type = custom_params[UPLOAD_PREFIX_KEY]
    if MODEL_NAME_KEY in custom_params:
        model_name = custom_params[MODEL_NAME_KEY]
    model_format = model_format.lower()
    model_filename = f"{model_type}_{model_name}-{model_version}-{model_format}.zip"
    # robust to inputs, no matter a component endswith "/" or not
    model_file_path = str(
        Path(model_folder) / model_name / model_version / model_format / model_filename
    )
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
    parser.add_argument(
        "custom_params", type=str, help="Custom parameters in json string"
    )

    args = parser.parse_args()
    verify_inputs(args.model_id, args.model_version)
    model_file_path = get_model_file_path(
        args.model_folder,
        args.model_id,
        args.model_version,
        args.model_format,
        json.loads(args.custom_params),
    )

    # Print the model file path so that the workflow can store it in the variable (See model_uploader.yml)
    print(model_file_path)
