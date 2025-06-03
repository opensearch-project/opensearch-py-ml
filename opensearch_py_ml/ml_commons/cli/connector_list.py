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
                name="Amazon Bedrock",
                file_name="bedrock_model",
                connector_class="BedrockModel",
                init_params=["opensearch_domain_region", "service_type"],
                connector_params=[
                    "connector_role_prefix",
                    "region",
                    "model_name",
                    "connector_body",
                    "aws_access_key",
                    "aws_secret_access_key",
                    "aws_session_token",
                ],
                available_models=[
                    ModelInfo(id="1", name="AI21 Labs Jurassic-2 Mid"),
                    ModelInfo(id="2", name="Anthropic Claude v2"),
                    ModelInfo(id="3", name="Anthropic Claude v3"),
                    ModelInfo(id="4", name="Anthropic Claude v3.7"),
                    ModelInfo(id="5", name="Cohere Embed Model v3 - English"),
                    ModelInfo(id="6", name="Cohere Embed Model v3 - Multilingual"),
                    ModelInfo(id="7", name="Titan Text Embedding"),
                    ModelInfo(id="8", name="Titan Multimodal Embedding"),
                    ModelInfo(id="9", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=3,
                name="Amazon Bedrock Converse",
                file_name="bedrock_converse_model",
                connector_class="BedrockConverseModel",
                init_params=["opensearch_domain_region", "service_type"],
                connector_params=[
                    "connector_role_prefix",
                    "region",
                    "model_name",
                    "connector_body",
                    "aws_access_key",
                    "aws_secret_access_key",
                    "aws_session_token",
                ],
                available_models=[
                    ModelInfo(id="1", name="Anthropic Claude 3 Sonnet"),
                    ModelInfo(id="2", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=4,
                name="Amazon Comprehend",
                file_name="comprehend_model",
                connector_class="ComprehendModel",
                init_params=["opensearch_domain_region", "service_type"],
                connector_params=[
                    "connector_role_prefix",
                    "region",
                    "model_name",
                    "connector_body",
                    "aws_access_key",
                    "aws_secret_access_key",
                    "aws_session_token",
                ],
                available_models=[
                    ModelInfo(id="1", name="Metadata embedding model"),
                    ModelInfo(id="2", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=5,
                name="Amazon Textract",
                file_name="textract_model",
                connector_class="TextractModel",
                init_params=["opensearch_domain_region", "service_type"],
                connector_params=[
                    "connector_role_prefix",
                    "region",
                    "model_name",
                    "connector_body",
                    "aws_access_key",
                    "aws_secret_access_key",
                    "aws_session_token",
                ],
                available_models=[
                    ModelInfo(id="1", name="Amazon Textract model"),
                    ModelInfo(id="2", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=6,
                name="Azure OpenAI",
                file_name="azure_openai_model",
                connector_class="AzureOpenAIModel",
                init_params=[],
                connector_params=[
                    "model_name",
                    "api_key",
                    "resource_name",
                    "deployment_name",
                    "api_version",
                    "connector_body",
                ],
                available_models=[
                    ModelInfo(id="1", name="Chat completion model"),
                    ModelInfo(id="2", name="Embedding model"),
                    ModelInfo(id="3", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=7,
                name="Amazon SageMaker",
                file_name="sagemaker_model",
                connector_class="SageMakerModel",
                init_params=["opensearch_domain_region", "service_type"],
                connector_params=[
                    "connector_role_prefix",
                    "region",
                    "model_name",
                    "endpoint_arn",
                    "endpoint_url",
                    "connector_body",
                    "aws_access_key",
                    "aws_secret_access_key",
                    "aws_session_token",
                ],
                available_models=[
                    ModelInfo(id="1", name="Embedding model"),
                    ModelInfo(id="2", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=8,
                name="Cohere",
                file_name="cohere_model",
                connector_class="CohereModel",
                init_params=["service_type"],
                connector_params=[
                    "connector_role_prefix",
                    "model_name",
                    "api_key",
                    "connector_body",
                    "connector_secret_name",
                ],
                available_models=[
                    ModelInfo(id="1", name="Chat model"),
                    ModelInfo(id="2", name="Embedding model"),
                    ModelInfo(id="3", name="Multi-modal embedding model"),
                    ModelInfo(id="4", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=9,
                name="DeepSeek",
                file_name="deepseek_model",
                connector_class="DeepSeekModel",
                init_params=["service_type"],
                connector_params=[
                    "connector_role_prefix",
                    "model_name",
                    "api_key",
                    "connector_body",
                    "connector_secret_name",
                ],
                available_models=[
                    ModelInfo(id="1", name="DeepSeek Chat model"),
                    ModelInfo(id="2", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=10,
                name="Google Cloud Platform",
                file_name="gcp_model",
                connector_class="GCPModel",
                init_params=[],
                connector_params=[
                    "model_name",
                    "project_id",
                    "model_id",
                    "access_token",
                    "connector_body",
                ],
                available_models=[
                    ModelInfo(id="1", name="VertexAI embedding model"),
                    ModelInfo(id="2", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=11,
                name="OpenAI",
                file_name="openai_model",
                connector_class="OpenAIModel",
                init_params=["service_type"],
                connector_params=[
                    "connector_role_prefix",
                    "model_name",
                    "api_key",
                    "connector_body",
                    "connector_secret_name",
                ],
                available_models=[
                    ModelInfo(id="1", name="Chat model"),
                    ModelInfo(id="2", name="Completion model"),
                    ModelInfo(id="3", name="Embedding model"),
                    ModelInfo(id="4", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=12,
                name="Custom connector",
                file_name="custom_connector",
                connector_class="CustomConnector",
                init_params=["service_type"],
                connector_params=[
                    "connector_role_prefix",
                    "connector_secret_name",
                    "connector_role_inline_policy",
                    "api_key",
                    "required_policy",
                    "required_secret",
                    "connector_body",
                ],
                available_models=[],
            ),
        ]
        # List of supported connectors in managed service (AOS)
        self._managed_connectors: List[ConnectorInfo] = [
            ConnectorInfo(
                id=1,
                name="Amazon Bedrock",
                file_name="bedrock_model",
                connector_class="BedrockModel",
                init_params=["opensearch_domain_region", "service_type"],
                connector_params=[
                    "connector_role_prefix",
                    "region",
                    "model_name",
                    "connector_body",
                    "aws_access_key",
                    "aws_secret_access_key",
                    "aws_session_token",
                ],
                available_models=[
                    ModelInfo(id="1", name="AI21 Labs Jurassic-2 Mid"),
                    ModelInfo(id="2", name="Anthropic Claude v2"),
                    ModelInfo(id="3", name="Anthropic Claude v3"),
                    ModelInfo(id="4", name="Anthropic Claude v3.7"),
                    ModelInfo(id="5", name="Cohere Embed Model v3 - English"),
                    ModelInfo(id="6", name="Cohere Embed Model v3 - Multilingual"),
                    ModelInfo(id="7", name="Titan Text Embedding"),
                    ModelInfo(id="8", name="Titan Multimodal Embedding"),
                    ModelInfo(id="9", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=2,
                name="Amazon Bedrock Converse",
                file_name="bedrock_converse_model",
                connector_class="BedrockConverseModel",
                init_params=["opensearch_domain_region", "service_type"],
                connector_params=[
                    "connector_role_prefix",
                    "region",
                    "model_name",
                    "connector_body",
                    "aws_access_key",
                    "aws_secret_access_key",
                    "aws_session_token",
                ],
                available_models=[
                    ModelInfo(id="1", name="Anthropic Claude 3 Sonnet"),
                    ModelInfo(id="2", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=3,
                name="Amazon Comprehend",
                file_name="comprehend_model",
                connector_class="ComprehendModel",
                init_params=["opensearch_domain_region", "service_type"],
                connector_params=[
                    "connector_role_prefix",
                    "region",
                    "model_name",
                    "connector_body",
                    "aws_access_key",
                    "aws_secret_access_key",
                    "aws_session_token",
                ],
                available_models=[
                    ModelInfo(id="1", name="Metadata embedding model"),
                    ModelInfo(id="2", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=4,
                name="Amazon SageMaker",
                file_name="sagemaker_model",
                connector_class="SageMakerModel",
                init_params=["opensearch_domain_region", "service_type"],
                connector_params=[
                    "connector_role_prefix",
                    "region",
                    "model_name",
                    "endpoint_arn",
                    "endpoint_url",
                    "connector_body",
                    "aws_access_key",
                    "aws_secret_access_key",
                    "aws_session_token",
                ],
                available_models=[
                    ModelInfo(id="1", name="DeepSeek R1 model"),
                    ModelInfo(id="2", name="Embedding model"),
                    ModelInfo(id="3", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=5,
                name="Amazon Textract",
                file_name="textract_model",
                connector_class="TextractModel",
                init_params=["opensearch_domain_region", "service_type"],
                connector_params=[
                    "connector_role_prefix",
                    "region",
                    "model_name",
                    "connector_body",
                    "aws_access_key",
                    "aws_secret_access_key",
                    "aws_session_token",
                ],
                available_models=[
                    ModelInfo(id="1", name="Amazon Textract model"),
                    ModelInfo(id="2", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=6,
                name="Cohere",
                file_name="cohere_model",
                connector_class="CohereModel",
                init_params=["service_type"],
                connector_params=[
                    "connector_role_prefix",
                    "model_name",
                    "api_key",
                    "connector_body",
                    "connector_secret_name",
                ],
                available_models=[
                    ModelInfo(id="1", name="Embedding model"),
                    ModelInfo(id="2", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=7,
                name="DeepSeek",
                file_name="deepseek_model",
                connector_class="DeepSeekModel",
                init_params=["service_type"],
                connector_params=[
                    "connector_role_prefix",
                    "model_name",
                    "api_key",
                    "connector_body",
                    "connector_secret_name",
                ],
                available_models=[
                    ModelInfo(id="1", name="DeepSeek Chat model"),
                    ModelInfo(id="2", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=8,
                name="OpenAI",
                file_name="openai_model",
                connector_class="OpenAIModel",
                init_params=["service_type"],
                connector_params=[
                    "connector_role_prefix",
                    "model_name",
                    "api_key",
                    "connector_body",
                    "connector_secret_name",
                ],
                available_models=[
                    ModelInfo(id="1", name="Embedding model"),
                    ModelInfo(id="2", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=9,
                name="Custom connector",
                file_name="custom_connector",
                connector_class="CustomConnector",
                init_params=["service_type"],
                connector_params=[
                    "connector_role_prefix",
                    "connector_secret_name",
                    "connector_role_inline_policy",
                    "api_key",
                    "required_policy",
                    "required_secret",
                    "connector_body",
                ],
                available_models=[],
            ),
        ]
