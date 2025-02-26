# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import argparse
import os
import shutil
import sys
from typing import Optional, Tuple

import numpy as np

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(THIS_DIR, "../..")
sys.path.append(ROOT_DIR)

from opensearch_py_ml.ml_commons import MLCommonClient
from opensearch_py_ml.ml_models import SparseEncodingModel
from tests import OPENSEARCH_TEST_CLIENT
from utils.model_uploader.autotracing_utils import (
    ATOL_TEST,
    BOTH_FORMAT,
    ONNX_FOLDER_PATH,
    ONNX_FORMAT,
    RTOL_TEST,
    SPARSE_ALGORITHM,
    TEMP_MODEL_PATH,
    TORCH_SCRIPT_FORMAT,
    TORCHSCRIPT_FOLDER_PATH,
    ModelTraceError,
    autotracing_warning_filters,
    check_model_status,
    delete_model,
    init_sparse_model,
    prepare_files_for_uploading,
    preview_model_config,
    register_and_deploy_model,
    store_description_variable,
    store_license_verified_variable,
    undeploy_model,
    verify_license_by_hfapi,
)

TEST_SENTENCES = ["Nice to meet you.", "I like playing football.", "Thanks."]


def trace_sparse_encoding_model(
    model_id: str,
    model_version: str,
    model_format: str,
    model_description: Optional[str] = None,
    sparse_prune_ratio: float = 0
) -> Tuple[str, str]:
    """
    Trace the pretrained sparse encoding model, create a model config file,
    and return a path to the model file and a path to the model config file required for model registration

    :param model_id: Model ID of the pretrained model
    :type model_id: string
    :param model_version: Version of the pretrained model for registration
    :type model_version: string
    :param model_format: Model format ("TORCH_SCRIPT" or "ONNX")
    :type model_format: string
    :param model_description: Model description input
    :type model_description: string
    :param sparse_prune_ratio: Model-side prune ratio for sparse_encoding
    :type sparse_prune_ratio: float
    :return: Tuple of model_path (path to model zip file) and model_config_path (path to model config json file)
    :rtype: Tuple[str, str]
    """

    folder_path = (
        TORCHSCRIPT_FOLDER_PATH
        if model_format == TORCH_SCRIPT_FORMAT
        else ONNX_FOLDER_PATH
    )

    # 1.) Initiate a sparse encoding model class object
    pre_trained_model = init_sparse_model(
        SparseEncodingModel, model_id, model_format, folder_path
    )

    # 2.) Save the model in the specified format

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
        raise ModelTraceError("saving model", model_format, e)

    # 3.) Create a model config json file
    try:
        model_config_path = pre_trained_model.make_model_config_json(
            version_number=model_version,
            model_format=model_format,
            description=model_description,
        )
    except Exception as e:
        raise ModelTraceError("making model config file", model_format, e)

    # 4.) Preview model config
    preview_model_config(model_format, model_config_path)

    # 5.) Return model_path & model_config_path for model registration
    return model_path, model_config_path


def register_and_deploy_sparse_encoding_model(
    ml_client: "MLCommonClient",
    model_path: str,
    model_config_path: str,
    model_format: str,
    texts: list[str],
) -> list:
    encoding_datas = None
    model_id = register_and_deploy_model(
        ml_client, model_format, model_path, model_config_path
    )
    check_model_status(ml_client, model_id, model_format, SPARSE_ALGORITHM)
    try:
        encoding_input = {"text_docs": texts}
        encoding_output = ml_client.generate_model_inference(model_id, encoding_input)
        encoding_datas = [
            encoding_output["inference_results"][i]["output"][0]["dataAsMap"][
                "response"
            ][0]
            for i in range(len(texts))
        ]
    except Exception as e:
        assert (
            False
        ), f"Raised Exception in generating sparse encoding with {model_format} model: {e}"
    undeploy_model(ml_client, model_id, model_format)
    delete_model(ml_client, model_id, model_format)
    return encoding_datas


def verify_embedding_data_vectors(original_embedding_datas, tracing_embedding_datas):
    if len(original_embedding_datas) != len(tracing_embedding_datas):
        print(
            f"The length of original_embedding_data_vector: {len(original_embedding_datas)} and "
            f"tracing_embedding_data_vector: {len(tracing_embedding_datas)} are different"
        )
        return False

    for index, (original, tracing) in enumerate(
        zip(original_embedding_datas, tracing_embedding_datas)
    ):
        if not verify_sparse_encoding(original, tracing):
            print(
                f"Verification failed for index {index}, whose input is {TEST_SENTENCES[index]}."
            )
            return False

    return True


def verify_sparse_encoding(
    original_embedding_data: dict,
    tracing_embedding_data: dict,
) -> bool:
    if original_embedding_data.keys() != tracing_embedding_data.keys():
        print("Different encoding dimensions")
        return False
    for key in original_embedding_data:
        a = original_embedding_data[key]
        b = tracing_embedding_data[key]
        if not np.allclose(a, b, rtol=RTOL_TEST, atol=ATOL_TEST):
            print(
                f"{key}'s score has gap: {original_embedding_data[key]} != {tracing_embedding_data[key]}"
            )
            return False
    return True


def main(
    model_id: str,
    model_version: str,
    tracing_format: str,
    model_description: Optional[str] = None,
    upload_prefix: Optional[str] = None,
    sparse_prune_ratio: float = 0,
) -> None:
    """
    Perform model auto-tracing and prepare files for uploading to OpenSearch model hub

    :param model_id: Model ID of the pretrained model
    :type model_id: string
    :param model_version: Version of the pretrained model for registration
    :type model_version: string
    :param tracing_format: Tracing format ("TORCH_SCRIPT", "ONNX", or "BOTH")
    :type tracing_format: string
    :param model_description: Model description input
    :type model_description: string
    :param upload_prefix: Model upload prefix input
    :type upload_prefix: string
    :param sparse_prune_ratio: Model-side prune ratio for sparse_encoding
    :type sparse_prune_ratio: float
    :return: No return value expected
    :rtype: None
    """

    print(
        f"""
    === Begin running sparse_model_autotracing.py ===
    Model ID: {model_id}
    Model Version: {model_version}
    Tracing Format: {tracing_format}
    Model Description: {model_description if model_description is not None else 'N/A'}
    Upload Prefix: {upload_prefix if upload_prefix is not None else 'N/A'}
    Sparse Prune Ratio: {sparse_prune_ratio}
    ==========================================
    """
    )

    # Now Sparse model auto tracing only support Torch Script.
    assert (
        tracing_format == TORCH_SCRIPT_FORMAT
    ), f"Now Only {TORCH_SCRIPT_FORMAT} is supported."

    ml_client = MLCommonClient(OPENSEARCH_TEST_CLIENT)
    pre_trained_model = SparseEncodingModel(model_id)
    original_encoding_datas = pre_trained_model.process_sparse_encoding(TEST_SENTENCES)
    pre_trained_model.save(path=TEMP_MODEL_PATH)
    license_verified = verify_license_by_hfapi(model_id)

    try:
        shutil.rmtree(TEMP_MODEL_PATH)
    except Exception as e:
        assert False, f"Raised Exception while deleting {TEMP_MODEL_PATH}: {e}"

    if tracing_format in [TORCH_SCRIPT_FORMAT, BOTH_FORMAT]:
        print("--- Begin tracing a model in TORCH_SCRIPT ---")
        (
            torchscript_model_path,
            torchscript_model_config_path,
        ) = trace_sparse_encoding_model(
            model_id,
            model_version,
            TORCH_SCRIPT_FORMAT,
            model_description=model_description,
        )

        torchscript_encoding_datas = register_and_deploy_sparse_encoding_model(
            ml_client,
            torchscript_model_path,
            torchscript_model_config_path,
            TORCH_SCRIPT_FORMAT,
            TEST_SENTENCES,
        )

        pass_test = verify_embedding_data_vectors(
            original_encoding_datas, torchscript_encoding_datas
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
        ) = trace_sparse_encoding_model(
            model_id, model_version, ONNX_FORMAT, model_description=model_description
        )

        onnx_embedding_datas = register_and_deploy_sparse_encoding_model(
            ml_client,
            onnx_model_path,
            onnx_model_config_path,
            ONNX_FORMAT,
            TEST_SENTENCES,
        )

        pass_test = verify_embedding_data_vectors(
            original_encoding_datas, onnx_embedding_datas
        )
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

    print("\n=== Finished running sparse_model_autotracing.py ===")


if __name__ == "__main__":
    autotracing_warning_filters()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_id",
        type=str,
        help="Model ID for auto-tracing and uploading (e.g. opensearch-project/opensearch-neural-sparse-encoding-v1)",
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
        "-md",
        "--model_description",
        type=str,
        nargs="?",
        default=None,
        const=None,
        help="Model description if you want to overwrite the default description",
    )
    parser.add_argument(
        "-spr",
        "--sparse_prune_ratio",
        type=float,
        nargs="?",
        default=None,
        const=None,
        help="sparse encoding model model-side pruning ratio",
    )
    args = parser.parse_args()

    sparse_prune_ratio = float(args.sparse_prune_ratio) if args.sparse_prune_ratio is not None else 0
    
    main(
        args.model_id,
        args.model_version,
        args.tracing_format,
        args.model_description,
        args.upload_prefix,
        sparse_prune_ratio
    )
