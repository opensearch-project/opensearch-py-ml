# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""
Metrics Correlation Auto-tracing Utility for OpenSearch

This module provides functionality for auto-tracing metrics correlation model
for deployment in OpenSearch. It handles creating example inputs, tracing the model,
packaging it for upload.
"""

import os
import sys

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(THIS_DIR, "../..")
sys.path.append(ROOT_DIR)

import argparse
import json
import warnings
from typing import Optional

import torch

from opensearch_py_ml.ml_models import MCorr
from utils.model_uploader.autotracing_utils import (
    METRICS_CORRELATION_ALGORITHM,
    TORCH_SCRIPT_FORMAT,
    autotracing_warning_filters,
    prepare_files_for_uploading,
    preview_model_config,
    store_description_variable,
    store_license_verified_variable,
)


def main(
    model_id: str,
    model_version: str,
    tracing_format: str,
    model_description: Optional[str] = None,
    model_name: Optional[str] = None,
    upload_prefix: Optional[str] = None,
) -> None:
    """
    Perform model auto-tracing and prepare files for uploading to OpenSearch model hub.

    This function handles the complete workflow for preparing a metrics correlation model:
    1. Generate example inputs for tracing
    2. Trace the model to TorchScript format
    3. Create model configuration
    4. Prepare files for uploading to the model repository

    Parameters
    ----------
    model_id : str
        Model ID of the pretrained Hugging Face model to trace
    model_version : str
        Version number to assign to the traced model for registration
    tracing_format : str
        Model format for tracing (only "TORCH_SCRIPT" is supported for semantic highlighter)
    model_description : str, optional
        Custom description for the model in the config file
    model_name : str, optional
        Custom name for the model in the config file and when registered
    upload_prefix : str, optional
        Path prefix for the uploaded model files
    """
    print(
        f"""
    === Begin running metrics_correlation_autotracing.py ===
    Model ID: {model_id}
    Model Version: {model_version}
    Tracing Format: {tracing_format}
    Model Description: {model_description if model_description is not None else 'N/A'}
    Model Name: {model_name if model_name is not None else 'N/A'}
    Upload Prefix: {upload_prefix if upload_prefix is not None else 'N/A'}
    ==========================================
    """
    )

    # Metrics correlation only supports TorchScript
    assert (
        tracing_format == TORCH_SCRIPT_FORMAT
    ), f"Metrics correlation only supports {TORCH_SCRIPT_FORMAT}"

    # License verification - internal Amazon model
    store_license_verified_variable(True)

    # Initialize the metrics correlation model
    print("--- Begin tracing metrics correlation model ---")
    mcorr_model = MCorr()
    mcorr_model.eval()

    # Create sample input for tracing
    sample_metrics = torch.normal(mean=0, std=1, size=[5, 128])
    print(f"Sample input shape: {sample_metrics.shape}")

    # Trace the model using torch.jit.script
    print("--- Tracing model to TorchScript ---")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traced_model = torch.jit.script(mcorr_model)

    # Test the traced model
    print("--- Testing traced model ---")
    with torch.no_grad():
        original_output = mcorr_model(sample_metrics)
        traced_output = traced_model(sample_metrics)
        print(f"Original model output: {len(original_output)} events")
        print(f"Traced model output: {len(traced_output)} events")

    # Save the traced model
    model_file_path = "metrics_correlation.pt"
    torch.jit.save(traced_model, model_file_path)
    print(f"Traced model saved to: {model_file_path}")

    # Store model description
    if model_description is None:
        model_description = "A metrics correlation model that detects anomalous events in time series data by identifying when multiple metrics simultaneously display unusual behavior."

    # Create model config
    print("--- Creating model configuration ---")
    config = {
        "name": model_name or model_id.split("/")[-1],
        "version": model_version,
        "model_format": "TORCH_SCRIPT",
        "algorithm": METRICS_CORRELATION_ALGORITHM,
        "description": model_description,
    }

    # Save config file
    config_file_path = "ml-commons_model_config.json"
    with open(config_file_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config file saved to: {config_file_path}")

    # Preview and prepare files for uploading
    preview_model_config(tracing_format, config_file_path)

    mcorr_model_path, mcorr_model_config_path = prepare_files_for_uploading(
        model_id=model_id,
        model_version=model_version,
        model_format=tracing_format,
        src_model_path=model_file_path,
        src_model_config_path=config_file_path,
        upload_prefix=None,
        model_name=model_name,
    )

    # Store description
    store_description_variable(mcorr_model_config_path)

    print("=== Metrics correlation auto-tracing completed ===")


if __name__ == "__main__":
    autotracing_warning_filters()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_id",
        type=str,
        help="Model ID for auto-tracing and uploading",
    )
    parser.add_argument(
        "model_version",
        type=str,
        help="Model version number (e.g. 1.0.1)",
    )
    parser.add_argument(
        "tracing_format",
        choices=["TORCH_SCRIPT"],
        help="Model format for auto-tracing (only TORCH_SCRIPT supported)",
    )
    parser.add_argument(
        "-up",
        "--upload_prefix",
        type=str,
        nargs="?",
        default=None,
        help="Model customize path prefix for upload",
    )
    parser.add_argument(
        "-mn",
        "--model_name",
        type=str,
        nargs="?",
        default=None,
        help="Model customize name for upload",
    )
    parser.add_argument(
        "-md",
        "--model_description",
        type=str,
        nargs="?",
        default=None,
        const=None,
        help="Model description if you want to overwrite the default description",
    )

    # Parse arguments and handle empty string values
    args = parser.parse_args()
    for arg in vars(args):
        value = getattr(args, arg)
        if isinstance(value, str) and value.strip() == "":
            setattr(args, arg, None)

    # Run the main function with parsed arguments
    main(
        args.model_id,
        args.model_version,
        args.tracing_format,
        args.model_description,
        args.model_name,
        args.upload_prefix,
    )
