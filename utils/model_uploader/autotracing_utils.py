# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.
import json
import os
import shutil
import warnings
from typing import Type, TypeVar

from huggingface_hub import HfApi

from opensearch_py_ml.ml_commons import MLCommonClient

# We need to append ROOT_DIR path so that we can import
# OPENSEARCH_TEST_CLIENT and opensearch_py_ml since this
# python script is not in the root directory.


BOTH_FORMAT = "BOTH"
TORCH_SCRIPT_FORMAT = "TORCH_SCRIPT"
ONNX_FORMAT = "ONNX"

DENSE_MODEL_ALGORITHM = "TEXT_EMBEDDING"
SPARSE_ALGORITHM = "SPARSE_ENCODING"
SPARSE_TOKENIZER_ALGORITHM = "SPARSE_TOKENIZE"
TEMP_MODEL_PATH = "temp_model_path"
TORCHSCRIPT_FOLDER_PATH = "model-torchscript/"
ONNX_FOLDER_PATH = "model-onnx/"
UPLOAD_FOLDER_PATH = "upload/"
MODEL_CONFIG_FILE_NAME = "ml-commons_model_config.json"
OUTPUT_DIR = "trace_output/"
LICENSE_VAR_FILE = "apache_verified.txt"
DESCRIPTION_VAR_FILE = "description.txt"
RTOL_TEST = 1e-03
ATOL_TEST = 1e-05


def register_and_deploy_model(
    ml_client: "MLCommonClient",
    model_format: str,
    model_path: str,
    model_config_path: str,
):
    """

    Args:
        ml_client: The ml client to register and deploy model
        model_format: The format of the model, one of [TORCH_SCRIPT,ONNX]
        model_path: The path of the model
        model_config_path: The path of the model config

    Returns:
        model_id The model_id of the registered model in OpenSearch

    """
    try:
        model_id = ml_client.register_model(
            model_path=model_path,
            model_config_path=model_config_path,
            deploy_model=True,
            isVerbose=True,
        )
        print(f"\n{model_format}_model_id:", model_id)
        assert model_id != "" or model_id is not None
        return model_id
    except Exception as e:
        assert (
            False
        ), f"Raised Exception in {model_format} model registration/deployment: {e}"


def check_model_status(
    ml_client: "MLCommonClient", model_id: str, model_format: str, model_algorithm: str
):
    """
    Check the status of the model.

    Args:
        ml_client:  Ml client to register and deploy model
        model_id:  The model_id of the registered model in OpenSearch
        model_format: The format of the model, one of [TORCH_SCRIPT,ONNX]

    Returns:

    """
    try:
        ml_model_status = ml_client.get_model_info(model_id)
        print("\nModel Status:")
        print(ml_model_status)
        assert ml_model_status.get("model_state") == "DEPLOYED"
        assert ml_model_status.get("model_format") == model_format
        assert ml_model_status.get("algorithm") == model_algorithm
    except Exception as e:
        assert False, f"Raised Exception in getting {model_format} model info: {e}"


def undeploy_model(ml_client: "MLCommonClient", model_id: str, model_format: str):
    """
    Undeploy the model from OpenSearch cluster.

    Args:
        ml_client:  Ml client to register and deploy model
        model_id:  The model_id of the registered model in OpenSearch
        model_format:  The format of the model, one of [TORCH_SCRIPT,ONNX]

    Returns:

    """
    try:
        ml_client.undeploy_model(model_id)
        ml_model_status = ml_client.get_model_info(model_id)
        assert ml_model_status.get("model_state") == "UNDEPLOYED"
    except Exception as e:
        assert False, f"Raised Exception in {model_format} model undeployment: {e}"


def delete_model(ml_client: "MLCommonClient", model_id: str, model_format: str):
    """
    Delete the model from OpenSearch cluster.

    Args:
        ml_client:  Ml client to register and deploy model
        model_id:  The model_id of the registered model in OpenSearch
        model_format:  The format of the model, one of [TORCH_SCRIPT,ONNX]

    Returns:

    """
    try:
        delete_model_obj = ml_client.delete_model(model_id)
        assert delete_model_obj.get("result") == "deleted"
    except Exception as e:
        assert False, f"Raised Exception in deleting {model_format} model: {e}"


def autotracing_warning_filters():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    warnings.filterwarnings("ignore", message="TracerWarning: torch.tensor")
    warnings.filterwarnings(
        "ignore", message="using SSL with verify_certs=False is insecure."
    )


def store_description_variable(config_path_for_checking_description: str) -> None:
    """
    Store model description in OUTPUT_DIR/DESCRIPTION_VAR_FILE
    to be used to generate issue body for manual approval

    :param config_path_for_checking_description: Path to config json file
    :type config_path_for_checking_description: str
    :return: No return value expected
    :rtype: None
    """
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        description_var_filepath = OUTPUT_DIR + "/" + DESCRIPTION_VAR_FILE
        with open(config_path_for_checking_description, "r") as f:
            config_dict = json.load(f)
            description = (
                config_dict["description"] if "description" in config_dict else "-"
            )
        print(f"Storing the following description at {description_var_filepath}")
        print(description)
        with open(description_var_filepath, "w") as f:
            f.write(description)
    except Exception as e:
        print(
            f"Cannot store description ({description}) in {description_var_filepath}: {e}"
        )


def store_license_verified_variable(license_verified: bool) -> None:
    """
    Store whether the model is licensed under Apache 2.0 in OUTPUT_DIR/LICENSE_VAR_FILE
    to be used to generate issue body for manual approval

    :param license_verified: Whether the model is licensed under Apache 2.0
    :type model_path: bool
    :return: No return value expected
    :rtype: None
    """
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        license_var_filepath = OUTPUT_DIR + "/" + LICENSE_VAR_FILE
        with open(license_var_filepath, "w") as f:
            f.write(str(license_verified))
    except Exception as e:
        print(
            f"Cannot store license_verified ({license_verified}) in {license_var_filepath}: {e}"
        )


def preview_model_config(model_format: str, model_config_path: str) -> None:
    print(f"\n+++++ {model_format} Model Config +++++\n")
    with open(model_config_path, "r") as f:
        model_config = json.load(f)
        print(json.dumps(model_config, indent=4))
    print("\n+++++++++++++++++++++++++++++++++++++++\n")


class ModelTraceError(Exception):
    """Custom exception for errors during the model tracing process."""

    def __init__(self, stage: str, model_format: str, original_exception: Exception):
        super().__init__(
            f"Error during {stage} for {model_format} model: {original_exception}"
        )
        self.stage = stage
        self.model_format = model_format
        self.original_exception = original_exception


T = TypeVar("T")


def init_sparse_model(
    model_class: Type[T], model_id, folder_path, sparse_prune_ratio=0, activation=None
) -> T:
    try:
        pre_trained_model = model_class(
            model_id=model_id,
            folder_path=folder_path,
            overwrite=True,
            sparse_prune_ratio=sparse_prune_ratio,
            activation=activation,
        )
    except Exception as e:
        raise ModelTraceError("initiating a sparse encoding model class object", e)
    return pre_trained_model


def prepare_files_for_uploading(
    model_id: str,
    model_version: str,
    model_format: str,
    src_model_path: str,
    src_model_config_path: str,
    upload_prefix: str = None,
) -> tuple[str, str]:
    """
    Prepare files for uploading by storing them in UPLOAD_FOLDER_PATH

    :param model_id: Model ID of the pretrained model
    :type model_id: string
    :param model_version: Version of the pretrained model for registration
    :type model_version: string
    :param model_format: Model format ("TORCH_SCRIPT" or "ONNX")
    :type model_format: string
    :param src_model_path: Path to model files for uploading
    :type src_model_path: string
    :param src_model_config_path: Path to model config files for uploading
    :type src_model_config_path: string
    :return: Tuple of dst_model_path (path to model zip file) and dst_model_config_path
    (path to model config json file) in the UPLOAD_FOLDER_PATH
    :rtype: Tuple[str, str]
    """
    model_type, model_name = (
        model_id.split("/")
        if upload_prefix is None
        else (upload_prefix, model_id.split("/")[-1])
    )
    model_format = model_format.lower()
    folder_to_delete = (
        TORCHSCRIPT_FOLDER_PATH if model_format == "torch_script" else ONNX_FOLDER_PATH
    )

    # Store to be uploaded files in UPLOAD_FOLDER_PATH
    try:
        dst_model_dir = (
            f"{UPLOAD_FOLDER_PATH}{model_name}/{model_version}/{model_format}"
        )
        os.makedirs(dst_model_dir, exist_ok=True)
        dst_model_filename = (
            f"{model_type}_{model_name}-{model_version}-{model_format}.zip"
        )
        dst_model_path = dst_model_dir + "/" + dst_model_filename
        shutil.copy(src_model_path, dst_model_path)
        print(f"\nCopied {src_model_path} to {dst_model_path}")

        dst_model_config_dir = (
            f"{UPLOAD_FOLDER_PATH}{model_name}/{model_version}/{model_format}"
        )
        os.makedirs(dst_model_config_dir, exist_ok=True)
        dst_model_config_filename = "config.json"
        dst_model_config_path = dst_model_config_dir + "/" + dst_model_config_filename
        shutil.copy(src_model_config_path, dst_model_config_path)
        print(f"Copied {src_model_config_path} to {dst_model_config_path}")
    except Exception as e:
        assert (
            False
        ), f"Raised Exception during preparing {model_format} files for uploading: {e}"

    # Delete model folder downloaded from HuggingFace during model tracing
    try:
        shutil.rmtree(folder_to_delete)
    except Exception as e:
        assert False, f"Raised Exception while deleting {folder_to_delete}: {e}"

    return dst_model_path, dst_model_config_path


def verify_license_by_hfapi(model_id: str):
    api = HfApi()
    model_info = api.model_info(model_id)
    model_license = model_info.cardData.get("license", "License information not found.")
    if model_license == "apache-2.0":
        return True
    else:
        return False
