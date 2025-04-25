# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.


from typing import Any, Callable, Dict, Optional

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ai_connector_helper import AIConnectorHelper
from opensearch_py_ml.ml_commons.cli.ml_models.model_base import ModelBase


class BedrockModel(ModelBase):
    def __init__(self, opensearch_domain_region, service_type):
        """
        Initializes the Bedrock model with necessary configurations
        """
        self.opensearch_domain_region = opensearch_domain_region
        self.service_type = service_type

    def _get_connector_body(self, model_type: str, region: str) -> Dict[str, Any]:
        """
        Get the connectory body.
        """
        region_prefix = region.split('-')[0].lower()
        connector_configs = {
            "1": {
                "name": "Amazon Bedrock: AI21 Labs Jurassic-2 Mid",
                "description": "The connector to Bedrock for AI21 Labs Jurassic-2 Mid model",
                "model": "ai21.j2-mid-v1",
                "request_body": '{"prompt":"${parameters.inputs}","maxTokens":200,"temperature":0.7,"topP":1,"stopSequences":[],"countPenalty":{"scale":0},"presencePenalty":{"scale":0},"frequencyPenalty":{"scale":0}}',
                "post_process_function": "\n  return params['completions'][0].data.text; \n",
                "parameters": {},
            },
            "2": {
                "name": "Amazon Bedrock: Anthropic Claude v2",
                "description": "The connector to Bedrock for Claude V2 model",
                "model": "anthropic.claude-v2",
                "request_body": '{"prompt":"\\n\\nHuman: ${parameters.inputs}\\n\\nAssistant:","max_tokens_to_sample":300,"temperature":0.5,"top_k":250,"top_p":1,"stop_sequences":["\\\\n\\\\nHuman:"]}',
                "parameters": {},
            },
            "3": {
                "name": "Amazon Bedrock: Anthropic Claude v3",
                "description": "The connector to Bedrock for Claude V3 model",
                "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                "request_body": '{"messages":[{"role":"user","content":[{"type":"text","text":"${parameters.inputs}"}]}],"anthropic_version":"${parameters.anthropic_version}","max_tokens":${parameters.max_tokens_to_sample}}',
                "parameters": {
                    "auth": "Sig_V4",
                    "response_filter": "$.content[0].text",
                    "max_tokens_to_sample": "8000",
                    "anthropic_version": "bedrock-2023-05-31",
                },
            },
            "4": {
                "name": "Amazon Bedrock: Anthropic Claude v3.7",
                "description": "The connector to Bedrock for Claude V3.7 model",
                "model": f"{region_prefix}.anthropic.claude-3-7-sonnet-20250219-v1:0",
                "request_body": '{ "anthropic_version": "${parameters.anthropic_version}", "max_tokens": ${parameters.max_tokens}, "temperature": ${parameters.temperature}, "messages": ${parameters.messages} }',
                "parameters": {
                    "max_tokens": 8000,
                    "temperature": 1,
                    "anthropic_version": "bedrock-2023-05-31",
                },
            },
            "5": {
                "name": "Amazon Bedrock: Cohere embed-english-v3",
                "description": "The connector to Bedrock for Cohere embed-english-v3 model",
                "model": "cohere.embed-english-v3",
                "request_body": '{ "texts": ${parameters.texts}, "truncate": "${parameters.truncate}", "input_type": "${parameters.input_type}" }',
                "pre_process_function": "connector.pre_process.cohere.embedding",
                "post_process_function": "connector.post_process.cohere.embedding",
                "parameters": {"truncate": "END", "input_type": "search_document"},
            },
            "6": {
                "name": "Amazon Bedrock: Cohere embed-multilingual-v3",
                "description": "The connector to Bedrock for Cohere embed-multilingual-v3",
                "model": "cohere.embed-multilingual-v3",
                "request_body": '{ "texts": ${parameters.texts}, "truncate": "${parameters.truncate}", "input_type": "${parameters.input_type}" }',
                "pre_process_function": "connector.pre_process.cohere.embedding",
                "post_process_function": "connector.post_process.cohere.embedding",
                "parameters": {"truncate": "END", "input_type": "search_document"},
            },
            "7": {
                "name": "Amazon Bedrock: Titan embedding model",
                "description": "The connector to Bedrock for Titan embedding model",
                "model": "amazon.titan-embed-text-v1",
                "request_body": '{ "inputText": "${parameters.inputText}" }',
                "pre_process_function": "connector.pre_process.bedrock.embedding",
                "post_process_function": "connector.post_process.bedrock.embedding",
                "parameters": {},
            },
            "8": {
                "name": "Amazon Bedrock: Titan Mulit-modal model",
                "description": "The connector to Bedrock for Titan Multi-modal model",
                "model": "amazon.titan-embed-image-v1",
                "request_body": '{"inputText": "${parameters.inputText:-null}", "inputImage": "${parameters.inputImage:-null}"}',
                "pre_process_function": "connector.pre_process.bedrock.multimodal_embedding",
                "post_process_function": "connector.post_process.bedrock.embedding",
                "parameters": {"input_docs_processed_step_size": 2},
            },
            "9": "Custom model",
        }

        # Handle custom model or invalid choice
        if (
            model_type not in connector_configs
            or connector_configs[model_type] == "Custom model"
        ):
            if model_type not in connector_configs:
                print(
                    f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
                )
            return self.input_custom_model_details()

        config = connector_configs[model_type]

        # Base parameters that all connectors need
        base_parameters = {
            "region": region,
            "service_name": "bedrock",
            "model": config["model"],
        }

        # Merge with model-specific parameters if any
        parameters = {**base_parameters, **config.get("parameters", {})}

        # Return the connector body
        return {
            "name": config["name"],
            "description": config["description"],
            "version": 1,
            "protocol": "aws_sigv4",
            "parameters": parameters,
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "headers": {"content-type": "application/json"},
                    "url": "https://bedrock-runtime.${parameters.region}.amazonaws.com/model/${parameters.model}/invoke",
                    "request_body": config["request_body"],
                    **(
                        {"pre_process_function": config["pre_process_function"]}
                        if "pre_process_function" in config
                        else {}
                    ),
                    **(
                        {"post_process_function": config["post_process_function"]}
                        if "post_process_function" in config
                        else {}
                    ),
                }
            ],
        }

    def _get_connector_role_policy(
        self, model_type: str, connector_body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get the IAM role policy for the connector.
        """
        model_name = connector_body["parameters"]["model"]

        # Handle inference profile
        if model_name.startswith(("us.", "eu.")):
            model_arn = [
                f"arn:aws:bedrock:*::foundation-model/{model_name[3:]}",
                f"arn:aws:bedrock:*:*:inference-profile/{model_name}",
            ]
        else:
            model_arn = [f"arn:aws:bedrock:*::foundation-model/{model_name}"]

        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": ["bedrock:InvokeModel"],
                    "Effect": "Allow",
                    "Resource": model_arn,
                }
            ],
        }

    def create_connector(
        self,
        helper: AIConnectorHelper,
        save_config_method: Callable[[str, Dict[str, Any]], None],
        connector_role_prefix: Optional[str] = None,
        region: Optional[str] = None,
        model_name: Optional[str] = None,
        connector_body: Optional[str] = None,
        aws_access_key: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ) -> bool:
        """
        Create Bedrock connector.

        Args:
            helper: Helper instance for OpenSearch connector operations.
            save_config_method: Method to save connector configuration after creation.
            connector_role_prefix (optional): Prefix for role names.
            region (optional): AWS region.
            model_name (optional): Specific Bedrock model name.
            connector_body (optional): The connector request body.
            aws_access_key (optional): AWS access key ID.
            aws_secret_access_key (optional): AWS secet access key.
            aws_session_token (optional): AWS session token.

        Returns:
            bool: True if connector creation successful, False otherwise.
        """
        # Set trusted connector endpoints for Bedrock
        trusted_endpoint = (
            "^https://bedrock-runtime\\..*[a-z0-9-]\\.amazonaws\\.com/.*$"
        )
        self.set_trusted_endpoint(helper, trusted_endpoint)

        # Prompt to choose model
        model_type = self.get_model_details(
            "Amazon Bedrock", self.service_type, model_name
        )

        # Prompt for region
        region = (
            region
            or input(
                f"Enter your AWS region [{self.opensearch_domain_region}]: "
            ).strip()
            or self.opensearch_domain_region
        )

        # Get connector body
        connector_body = connector_body or self._get_connector_body(model_type, region)

        if self.service_type == self.AMAZON_OPENSEARCH_SERVICE:
            # Create connector role
            connector_role_name, create_connector_role_name = (
                self.create_connector_role(connector_role_prefix, "bedrock")
            )
            # Get the connector role inline policy
            connector_role_inline_policy = self._get_connector_role_policy(
                model_type, connector_body
            )

            # Create connector
            print("\nCreating Bedrock connector...")
            connector_id, connector_role_arn, _ = helper.create_connector_with_role(
                connector_role_inline_policy,
                connector_role_name,
                create_connector_role_name,
                connector_body,
                sleep_time_in_seconds=10,
            )
        else:
            # Prompt for AWS credentials
            self.get_aws_credentials(
                connector_body, aws_access_key, aws_secret_access_key, aws_session_token
            )

            # Create connector
            print("\nCreating Bedrock connector...")
            connector_id = helper.create_connector(
                create_connector_role_name=None,
                body=connector_body,
            )

        if connector_id:
            print(
                f"{Fore.GREEN}\nSuccessfully created Bedrock connector with ID: {connector_id}{Style.RESET_ALL}"
            )
            connector_output = helper.get_connector(connector_id)
            save_config_method(
                connector_id,
                connector_output,
                (
                    connector_role_name
                    if self.service_type == self.AMAZON_OPENSEARCH_SERVICE
                    else None
                ),
                (
                    connector_role_arn
                    if self.service_type == self.AMAZON_OPENSEARCH_SERVICE
                    else None
                ),
            )
            return True
        print(f"{Fore.RED}Failed to create Bedrock connector.{Style.RESET_ALL}")
        return False
