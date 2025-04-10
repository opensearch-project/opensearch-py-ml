# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from dataclasses import dataclass
from typing import List


@dataclass
class ModelInfo:
    """
    Information about a specific model within a connector
    """

    id: str
    name: str


@dataclass
class ConnectorInfo:
    """
    Information about a connector and its available models
    """

    id: int
    name: str
    file_name: str
    connector_class: str
    init_params: List[str]
    connector_params: List[str]
    available_models: List[ModelInfo] = None


class ConnectorList:
    def __init__(self):
        # TODO: Add more supported connectors for both AOS and open-source
        # List of supported connectors in open-source service
        self._opensource_connectors: List[ConnectorInfo] = [
            ConnectorInfo(
                id=1,
                name="Aleph Alpha",
                file_name="aleph_alpha_model",
                connector_class="AlephAlphaModel",
                init_params=[],
                connector_params=["model_name", "api_key", "connector_body"],
                available_models=[
                    ModelInfo(id="1", name="Luminous-Base embedding model"),
                    ModelInfo(id="2", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=2,
                name="DeepSeek",
                file_name="deepseek_model",
                connector_class="DeepSeekModel",
                init_params=["service_type"],
                connector_params=[
                    "connector_role_prefix",
                    "model_name",
                    "api_key",
                    "connector_body",
                    "secret_name",
                ],
                available_models=[
                    ModelInfo(id="1", name="DeepSeek Chat model"),
                    ModelInfo(id="2", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=3,
                name="OpenAI",
                file_name="openai_model",
                connector_class="OpenAIModel",
                init_params=["service_type"],
                connector_params=[
                    "connector_role_prefix",
                    "model_name",
                    "api_key",
                    "connector_body",
                    "secret_name",
                ],
                available_models=[
                    ModelInfo(id="1", name="Chat model"),
                    ModelInfo(id="2", name="Embedding model"),
                    ModelInfo(id="3", name="Custom model"),
                ],
            ),
        ]
        # List of supported connectors in managed service (AOS)
        self._managed_connectors: List[ConnectorInfo] = [
            ConnectorInfo(
                id=1,
                name="Amazon Bedrock",
                file_name="bedrock_model",
                connector_class="BedrockModel",
                init_params=["opensearch_domain_region"],
                connector_params=[
                    "connector_role_prefix",
                    "region",
                    "model_name",
                    "model_arn",
                    "connector_body",
                ],
                available_models=[
                    ModelInfo(id="1", name="Cohere embedding model"),
                    ModelInfo(id="2", name="Titan embedding model"),
                    ModelInfo(id="3", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=2,
                name="Amazon SageMaker",
                file_name="sagemaker_model",
                connector_class="SageMakerModel",
                init_params=["opensearch_domain_region"],
                connector_params=[
                    "connector_role_prefix",
                    "region",
                    "model_name",
                    "endpoint_arn",
                    "endpoint_url",
                    "connector_body",
                ],
                available_models=[
                    ModelInfo(id="1", name="DeepSeek R1 model"),
                    ModelInfo(id="2", name="Embedding model"),
                    ModelInfo(id="3", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=3,
                name="Cohere",
                file_name="cohere_model",
                connector_class="CohereModel",
                init_params=[],
                connector_params=[
                    "connector_role_prefix",
                    "model_name",
                    "api_key",
                    "connector_body",
                    "secret_name",
                ],
                available_models=[
                    ModelInfo(id="1", name="Embedding model"),
                    ModelInfo(id="2", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=4,
                name="DeepSeek",
                file_name="deepseek_model",
                connector_class="DeepSeekModel",
                init_params=["service_type"],
                connector_params=[
                    "connector_role_prefix",
                    "model_name",
                    "api_key",
                    "connector_body",
                    "secret_name",
                ],
                available_models=[
                    ModelInfo(id="1", name="DeepSeek Chat model"),
                    ModelInfo(id="2", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=5,
                name="OpenAI",
                file_name="openai_model",
                connector_class="OpenAIModel",
                init_params=["service_type"],
                connector_params=[
                    "connector_role_prefix",
                    "model_name",
                    "api_key",
                    "connector_body",
                    "secret_name",
                ],
                available_models=[
                    ModelInfo(id="1", name="Embedding model"),
                    ModelInfo(id="2", name="Custom model"),
                ],
            ),
        ]
