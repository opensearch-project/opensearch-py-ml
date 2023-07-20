# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

# This program is run by "Model Auto-tracing & Uploading" workflow
# (See model_uploader.yml) to perform model auto-tracing and prepare
# files for uploading to OpenSearch model hub.

import argparse
import os
import shutil
import sys
import warnings
from typing import List, Optional, Tuple
from zipfile import ZipFile

import numpy as np
from numpy.typing import DTypeLike
from sentence_transformers import SentenceTransformer

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(THIS_DIR, "../..")
sys.path.append(ROOT_DIR) # Required for importing OPENSEARCH_TEST_CLIENT

LICENSE_PATH = "LICENSE"
from opensearch_py_ml.ml_commons import MLCommonClient
from opensearch_py_ml.ml_models.sentencetransformermodel import SentenceTransformerModel
from tests import OPENSEARCH_TEST_CLIENT

BOTH_FORMAT = "BOTH"
TORCH_SCRIPT_FORMAT = "TORCH_SCRIPT"
ONNX_FORMAT = "ONNX"

ORIGINAL_FOLDER_PATH = "sentence-transformers-original/"
TORCHSCRIPT_FOLDER_PATH = "sentence-transformers-torchscript/"
ONNX_FOLDER_PATH = "sentence-transformers-onnx/"
UPLOAD_FOLDER_PATH = "upload/"
MODEL_CONFIG_FILE_NAME = "ml-commons_model_config.json"
TEST_SENTENCES = [
    "First test sentence",
    "This is a very long sentence used for testing model embedding outputs.",
]
RTOL_TEST = 1e-03
ATOL_TEST = 1e-05
ML_BASE_URI = "/_plugins/_ml"


def trace_sentence_transformer_model(
    model_id: str,
    model_version: str,
    model_format: str,
    embedding_dimension: Optional[int] = None,
    pooling_mode: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Trace the pretrained sentence transformer model, create a model config file,
    and return a path to the model file and a path to the model config file required for model registration

    :param model_id: Model ID of the pretrained model
    :type model_id: string
    :param model_version: Version of the pretrained model for registration
    :type model_version: string
    :param model_format: Model format ("TORCH_SCRIPT" or "ONNX")
    :type model_format: string
    :param embedding_dimension: Embedding dimension input
    :type embedding_dimension: int
    :param pooling_mode: Pooling mode input ("CLS", "MEAN", "MAX", "MEAN_SQRT_LEN" or None)
    :type pooling_mode: string
    """
    folder_path = (
        TORCHSCRIPT_FOLDER_PATH
        if model_format == TORCH_SCRIPT_FORMAT
        else ONNX_FOLDER_PATH
    )

    # 1.) Initiate a sentence transformer model class object
    pre_trained_model = None
    try:
        pre_trained_model = SentenceTransformerModel(
            model_id=model_id, folder_path=folder_path, overwrite=True
        )
    except Exception as e:
        assert (
            False
        ), f"Raised Exception in tracing {model_format} model\
                             during initiating a sentence transformer model class object: {e}"

    # 2.) Save the model in the specified format
    model_path = None
    try:
        if model_format == TORCH_SCRIPT_FORMAT:
            model_path = pre_trained_model.save_as_pt(
                model_id=model_id, sentences=TEST_SENTENCES
            )
        else:
            model_path = pre_trained_model.save_as_onnx(model_id=model_id)
    except Exception as e:
        assert False, f"Raised Exception during saving model as {model_format}: {e}"

    # 3.) Create a model config json file
    try:
        pre_trained_model.make_model_config_json(
            version_number=model_version,
            model_format=model_format,
            embedding_dimension=embedding_dimension,
            pooling_mode=pooling_mode,
        )
    except Exception as e:
        assert (
            False
        ), f"Raised Exception during making model config file for {model_format} model: {e}"

    # 4.) Return model_path & model_config_path for model registration
    model_config_path = folder_path + MODEL_CONFIG_FILE_NAME
    return model_path, model_config_path


def register_and_deploy_sentence_transformer_model(
    ml_client: "MLCommonClient",
    model_path: str,
    model_config_path: str,
    model_format: str,
) -> List["DTypeLike"]:
    """
    Register the pretrained sentence transformer model by using the model file and the model config file,
    deploy the model to generate embeddings for the TEST_SENTENCES,
    and return the embeddings for model verification

    :param ml_client: A client that communicates to the ml-common plugin for OpenSearch
    :type ml_client: MLCommonClient
    :param model_path: Path to model file
    :type model_path: string
    :param model_config_path: Path to model config file
    :type model_config_path: string
    :param model_format: Model format ("TORCH_SCRIPT" or "ONNX")
    :type model_format: string
    """
    embedding_data = None

    # 1.) Register & Deploy the model
    model_id = ""
    task_id = ""
    try:
        model_id = ml_client.register_model(
            model_path=model_path,
            model_config_path=model_config_path,
            deploy_model=True,
            isVerbose=True,
        )
        print()
        print(f"{model_format}_model_id:", model_id)
        assert model_id != "" or model_id is not None
    except Exception as e:
        assert False, f"Raised Exception in {model_format} model registration/deployment: {e}"

    # 2.) Check model status
    try:
        ml_model_status = ml_client.get_model_info(model_id)
        print()
        print("Model Status:")
        print(ml_model_status)
        assert ml_model_status.get("model_format") == model_format
        assert ml_model_status.get("algorithm") == "TEXT_EMBEDDING"
    except Exception as e:
        assert False, f"Raised Exception in getting {model_format} model info: {e}"

    # 3.) Generate embeddings
    try:
        embedding_output = ml_client.generate_embedding(model_id, TEST_SENTENCES)
        assert len(embedding_output.get("inference_results")) == 2
        embedding_data = [
            embedding_output["inference_results"][i]["output"][0]["data"]
            for i in range(len(TEST_SENTENCES))
        ]
    except Exception as e:
        assert (
            False
        ), f"Raised Exception in generating sentence embedding with {model_format} model: {e}"

    # 4.) Undeploy the model
    try:
        ml_client.undeploy_model(model_id)
        ml_model_status = ml_client.get_model_info(model_id)
        assert ml_model_status.get("model_state") != "UNDEPLOY_FAILED"
    except Exception as e:
        assert False, f"Raised Exception in {model_format} model undeployment: {e}"

    # 5.) Delete the model
    try:
        delete_model_obj = ml_client.delete_model(model_id)
        assert delete_model_obj.get("result") == "deleted"
    except Exception as e:
        assert False, f"Raised Exception in deleting {model_format} model: {e}"

    # 6.) Return embedding outputs for model verification
    return embedding_data


def verify_embedding_data(
    original_embedding_data: List["DTypeLike"],
    tracing_embedding_data: List["DTypeLike"],
    model_format: str,
) -> None:
    """
    Verify the embeddings generated by the traced model with that of original model

    :param original_embedding_data: Embedding outputs of TEST_SENTENCES generated by the original model
    :type original_embedding_data: List['DTypeLike']
    :param tracing_embedding_data: Embedding outputs of TEST_SENTENCES generated by the traced model
    :type tracing_embedding_data: List['DTypeLike']
    :param model_format: Model format ("TORCH_SCRIPT" or "ONNX")
    :type model_format: string
    """
    failed_cases = []
    for i in range(len(TEST_SENTENCES)):
        try:
            np.testing.assert_allclose(
                original_embedding_data[i],
                tracing_embedding_data[i],
                rtol=RTOL_TEST,
                atol=ATOL_TEST,
            )
        except Exception as e:
            failed_cases.append((TEST_SENTENCES[i], e))

    if len(failed_cases):
        print()
        print(
            f"Original embeddings DOES NOT matches {model_format} embeddings in the following case(s):"
        )
        for sentence, e in failed_cases:
            print(sentence)
            print(e)
        assert False, f"Failed while verifying embeddings of {model_format} model"
    else:
        print()
        print(f"Original embeddings matches {model_format} embeddings")


def prepare_files_for_uploading(
    model_id: str,
    model_version: str,
    model_format: str,
    src_model_path: str,
    src_model_config_path: str,
) -> None:
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
    """
    model_name = str(model_id.split("/")[-1])
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
            f"sentence-transformers_{model_name}-{model_version}-{model_format}.zip"
        )
        dst_model_path = dst_model_dir + "/" + dst_model_filename
        with ZipFile(src_model_path, "a") as zipObj:
            zipObj.write(filename=LICENSE_PATH, arcname="LICENSE")
        shutil.copy(src_model_path, dst_model_path)
        print()
        print(f"Copied {src_model_path} to {dst_model_path}")

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


def main(
    model_id: str,
    model_version: str,
    tracing_format: str,
    embedding_dimension: Optional[int] = None,
    pooling_mode: Optional[str] = None,
) -> None:
    """
    Perform model auto-tracing and prepare files for uploading to OpenSearch model hub

    :param model_id: Model ID of the pretrained model
    :type model_id: string
    :param model_version: Version of the pretrained model for registration
    :type model_version: string
    :param tracing_format: Tracing format ("TORCH_SCRIPT", "ONNX", or "BOTH")
    :type tracing_format: string
    :param embedding_dimension: Embedding dimension input
    :type embedding_dimension: int
    :param pooling_mode: Pooling mode input ("CLS", "MEAN", "MAX", "MEAN_SQRT_LEN" or None)
    :type pooling_mode: string
    """

    print()
    print("=== Begin running model_autotracing.py ===")
    print("Model ID: ", model_id)
    print("Model Version: ", model_version)
    print("Tracing Format: ", tracing_format)
    print("Embedding Dimension: ", embedding_dimension)
    print("Pooling Mode: ", pooling_mode)
    print("==========================================")

    ml_client = MLCommonClient(OPENSEARCH_TEST_CLIENT)

    pre_trained_model = SentenceTransformer(model_id)
    original_embedding_data = list(
        pre_trained_model.encode(TEST_SENTENCES, convert_to_numpy=True)
    )

    if tracing_format in [TORCH_SCRIPT_FORMAT, BOTH_FORMAT]:
        print("--- Begin tracing a model in TORCH_SCRIPT ---")
        (
            torchscript_model_path,
            torchscript_model_config_path,
        ) = trace_sentence_transformer_model(
            model_id,
            model_version,
            TORCH_SCRIPT_FORMAT,
            embedding_dimension,
            pooling_mode,
        )
        torch_embedding_data = register_and_deploy_sentence_transformer_model(
            ml_client,
            torchscript_model_path,
            torchscript_model_config_path,
            TORCH_SCRIPT_FORMAT,
        )
        verify_embedding_data(
            original_embedding_data, torch_embedding_data, TORCH_SCRIPT_FORMAT
        )
        prepare_files_for_uploading(
            model_id,
            model_version,
            TORCH_SCRIPT_FORMAT,
            torchscript_model_path,
            torchscript_model_config_path,
        )
        print("--- Finished tracing a model in TORCH_SCRIPT ---")

    if tracing_format in [ONNX_FORMAT, BOTH_FORMAT]:
        print("--- Begin tracing a model in ONNX ---")
        onnx_model_path, onnx_model_config_path = trace_sentence_transformer_model(
            model_id,
            model_version,
            ONNX_FORMAT,
            embedding_dimension,
            pooling_mode,
        )
        onnx_embedding_data = register_and_deploy_sentence_transformer_model(
            ml_client, onnx_model_path, onnx_model_config_path, ONNX_FORMAT
        )

        verify_embedding_data(original_embedding_data, onnx_embedding_data, ONNX_FORMAT)
        prepare_files_for_uploading(
            model_id,
            model_version,
            ONNX_FORMAT,
            onnx_model_path,
            onnx_model_config_path,
        )
        print("--- Finished tracing a model in ONNX ---")

    print()
    print("=== Finished running model_autotracing.py ===")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    warnings.filterwarnings("ignore", message="TracerWarning: torch.tensor")
    warnings.filterwarnings(
        "ignore", message="using SSL with verify_certs=False is insecure."
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_id",
        type=str,
        help="Model ID for auto-tracing and uploading (e.g. sentence-transformers/msmarco-distilbert-base-tas-b)",
    )
    parser.add_argument(
        "model_version", type=str, help="Model version number (e.g. 1.0.1)"
    )
    parser.add_argument(
        "tracing_format",
        choices=["BOTH", "TORCH_SCRIPT", "ONNX"],
        help="Model format for auto-tracing",
    )
    parser.add_argument(
        "-ed",
        "--embedding_dimension",
        type=int,
        nargs="?",
        default=None,
        const=None,
        help="Embedding dimension of the model to use if it does not exist in original config.json",
    )
    parser.add_argument(
        "-pm",
        "--pooling_mode",
        type=str,
        nargs="?",
        default=None,
        const=None,
        choices=["CLS", "MEAN", "MAX", "MEAN_SQRT_LEN"],
        help="Pooling mode if it does not exist in original config.json",
    )
    args = parser.parse_args()

    main(
        args.model_id,
        args.model_version,
        args.tracing_format,
        args.embedding_dimension,
        args.pooling_mode,
    )
