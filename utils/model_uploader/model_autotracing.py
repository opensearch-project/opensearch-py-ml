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
from typing import List, Optional, Tuple

import numpy as np
from mdutils.fileutils import MarkDownFile
from numpy.typing import DTypeLike
from sentence_transformers import SentenceTransformer

# We need to append ROOT_DIR path so that we can import
# OPENSEARCH_TEST_CLIENT and opensearch_py_ml since this
# python script is not in the root directory.
THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(THIS_DIR, "../..")
sys.path.append(ROOT_DIR)

from opensearch_py_ml.ml_commons import MLCommonClient
from opensearch_py_ml.ml_models.sentencetransformermodel import SentenceTransformerModel
from tests import OPENSEARCH_TEST_CLIENT
from utils.model_uploader.autotracing_utils import (
    ATOL_TEST,
    BOTH_FORMAT,
    DENSE_MODEL_ALGORITHM,
    ONNX_FOLDER_PATH,
    ONNX_FORMAT,
    RTOL_TEST,
    TEMP_MODEL_PATH,
    TORCH_SCRIPT_FORMAT,
    TORCHSCRIPT_FOLDER_PATH,
    autotracing_warning_filters,
    check_model_status,
    prepare_files_for_uploading,
    preview_model_config,
    register_and_deploy_model,
    store_description_variable,
)
from utils.model_uploader.sparse_model_autotracing import (
    store_license_verified_variable,
)

TEST_SENTENCES = [
    "First test sentence",
    "This is another sentence used for testing model embedding outputs.",
    "OpenSearch is a scalable, flexible, and extensible open-source software suite for search, analytics, "
    "and observability applications licensed under Apache 2.0. Powered by Apache Lucene and driven by the OpenSearch "
    "Project community, OpenSearch offers a vendor-agnostic toolset you can use to build secure, high-performance, "
    "cost-efficient applications. Use OpenSearch as an end-to-end solution or connect it with your preferred "
    "open-source tools or partner projects.",
]


def verify_license_in_md_file() -> bool:
    """
    Verify that the model is licensed under Apache 2.0
    by looking at metadata in README.md file of the model

    TODO: Support other open source licenses in future

    :return: Whether the model is licensed under Apache 2.0
    :rtype: Bool
    """
    try:
        readme_data = MarkDownFile.read_file(TEMP_MODEL_PATH + "/" + "README.md")
    except Exception as e:
        print(f"Cannot verify the license: {e}")
        return False

    start = readme_data.find("---")
    end = readme_data.find("---", start + 3)
    if start == -1 or end == -1:
        return False
    metadata_info = readme_data[start + 3 : end]
    if "apache-2.0" in metadata_info.lower():
        print("\nFound apache-2.0 license at " + TEMP_MODEL_PATH + "/README.md")
        return True
    else:
        print("\nDid not find apache-2.0 license at " + TEMP_MODEL_PATH + "/README.md")
        return False


def trace_sentence_transformer_model(
    model_id: str,
    model_version: str,
    model_format: str,
    embedding_dimension: Optional[int] = None,
    pooling_mode: Optional[str] = None,
    model_description: Optional[str] = None,
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
    :param model_description: Model description input
    :type model_description: string
    :return: Tuple of model_path (path to model zip file) and model_config_path (path to model config json file)
    :rtype: Tuple[str, str]
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
                model_id=model_id,
                sentences=TEST_SENTENCES,
                add_apache_license=True,
            )
        else:
            model_path = pre_trained_model.save_as_onnx(
                model_id=model_id, add_apache_license=True
            )
    except Exception as e:
        assert False, f"Raised Exception during saving model as {model_format}: {e}"

    # 3.) Create a model config json file
    model_config_path = None
    try:
        model_config_path = pre_trained_model.make_model_config_json(
            version_number=model_version,
            model_format=model_format,
            embedding_dimension=embedding_dimension,
            pooling_mode=pooling_mode,
            description=model_description,
        )
    except Exception as e:
        assert (
            False
        ), f"Raised Exception during making model config file for {model_format} model: {e}"

    # 4.) Preview model config
    preview_model_config(model_format, model_config_path)

    # 5.) Return model_path & model_config_path for model registration
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
    :return: List of embedding data for TEST_SENTENCES
    :rtype: List["DTypeLike"]
    """
    embedding_data = None

    # 1.) Register & Deploy the model
    model_id = register_and_deploy_model(
        ml_client, model_format, model_path, model_config_path
    )
    # 2.) Check model status
    check_model_status(ml_client, model_id, model_format, DENSE_MODEL_ALGORITHM)
    # 3.) Generate embeddings
    try:
        embedding_output = ml_client.generate_embedding(model_id, TEST_SENTENCES)
        assert len(embedding_output.get("inference_results")) == len(TEST_SENTENCES)
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
        assert ml_model_status.get("model_state") == "UNDEPLOYED"
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
) -> bool:
    """
    Verify the embeddings generated by the traced model with those of original model

    :param original_embedding_data: Embedding outputs of TEST_SENTENCES generated by the original model
    :type original_embedding_data: List['DTypeLike']
    :param tracing_embedding_data: Embedding outputs of TEST_SENTENCES generated by the traced model
    :type tracing_embedding_data: List['DTypeLike']
    :return: Whether the embeddings generated by the traced model match with those of original model
    :rtype: bool
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
        print(
            "\nOriginal embeddings DOES NOT matches the embeddings in the following case(s):"
        )
        for sentence, e in failed_cases:
            print(sentence)
            print(e)
        return False
    else:
        return True


def main(
    model_id: str,
    model_version: str,
    tracing_format: str,
    embedding_dimension: Optional[int] = None,
    pooling_mode: Optional[str] = None,
    model_description: Optional[str] = None,
    upload_prefix: Optional[str] = None,
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
    :param model_description: Model description input
    :type model_description: string
    :return: No return value expected
    :rtype: None
    """
    print(
        f"""
    === Begin running model_autotracing.py ===
    Model ID: {model_id}
    Model Version: {model_version}
    Tracing Format: {tracing_format}
    Embedding Dimension: {embedding_dimension if embedding_dimension is not None else 'N/A'}
    Pooling Mode: {pooling_mode if pooling_mode is not None else 'N/A'}
    Model Description: {model_description if model_description is not None else 'N/A'}
    ==========================================
    """
    )

    ml_client = MLCommonClient(OPENSEARCH_TEST_CLIENT)

    pre_trained_model = SentenceTransformer(model_id)
    original_embedding_data = list(
        pre_trained_model.encode(TEST_SENTENCES, convert_to_numpy=True)
    )

    pre_trained_model.save(path=TEMP_MODEL_PATH)
    license_verified = verify_license_in_md_file()
    try:
        shutil.rmtree(TEMP_MODEL_PATH)
    except Exception as e:
        assert False, f"Raised Exception while deleting {TEMP_MODEL_PATH}: {e}"

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
            model_description,
        )

        torchscript_embedding_data = register_and_deploy_sentence_transformer_model(
            ml_client,
            torchscript_model_path,
            torchscript_model_config_path,
            TORCH_SCRIPT_FORMAT,
        )
        pass_test = verify_embedding_data(
            original_embedding_data, torchscript_embedding_data
        )
        assert (
            pass_test
        ), f"Failed while verifying embeddings of {model_id} model in TORCH_SCRIPT format"

        (
            torchscript_dst_model_path,
            torchscript_dst_model_config_path,
        ) = prepare_files_for_uploading(
            model_id,
            model_version,
            TORCH_SCRIPT_FORMAT,
            torchscript_model_path,
            torchscript_model_config_path,
            upload_prefix,
        )

        config_path_for_checking_description = torchscript_dst_model_config_path
        print("--- Finished tracing a model in TORCH_SCRIPT ---")

    if tracing_format in [ONNX_FORMAT, BOTH_FORMAT]:
        print("--- Begin tracing a model in ONNX ---")
        (
            onnx_model_path,
            onnx_model_config_path,
        ) = trace_sentence_transformer_model(
            model_id,
            model_version,
            ONNX_FORMAT,
            embedding_dimension,
            pooling_mode,
            model_description,
        )

        onnx_embedding_data = register_and_deploy_sentence_transformer_model(
            ml_client, onnx_model_path, onnx_model_config_path, ONNX_FORMAT
        )

        pass_test = verify_embedding_data(original_embedding_data, onnx_embedding_data)
        assert (
            pass_test
        ), f"Failed while verifying embeddings of {model_id} model in ONNX format"

        onnx_dst_model_path, onnx_dst_model_config_path = prepare_files_for_uploading(
            model_id,
            model_version,
            ONNX_FORMAT,
            onnx_model_path,
            onnx_model_config_path,
        )

        config_path_for_checking_description = onnx_dst_model_config_path
        print("--- Finished tracing a model in ONNX ---")

    store_license_verified_variable(license_verified)
    store_description_variable(config_path_for_checking_description)

    print("\n=== Finished running model_autotracing.py ===")


if __name__ == "__main__":
    autotracing_warning_filters()

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
        "-up",
        "--upload_prefix",
        type=str,
        nargs="?",
        default=None,
        help="Model customize path prefix for upload",
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
    parser.add_argument(
        "-md",
        "--model_description",
        type=str,
        nargs="?",
        default=None,
        const=None,
        help="Model description if you want to overwrite the default description",
    )
    args = parser.parse_args()

    main(
        args.model_id,
        args.model_version,
        args.tracing_format,
        args.embedding_dimension,
        args.pooling_mode,
        args.model_description,
        args.upload_prefix,
    )
