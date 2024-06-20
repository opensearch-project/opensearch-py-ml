# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.
import warnings

from opensearch_py_ml.ml_commons import MLCommonClient


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


def check_model_status(ml_client: "MLCommonClient", model_id: str, model_format: str):
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
        assert ml_model_status.get("algorithm") == "SPARSE_ENCODING"
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
