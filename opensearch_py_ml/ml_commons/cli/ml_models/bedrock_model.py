# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.


from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.model_base import ModelBase
from opensearch_py_ml.ml_commons.cli.ml_setup import Setup


class BedrockModel(ModelBase):
    def __init__(self, opensearch_domain_region, service_type):
        """
        Initializes the Bedrock model with necessary configurations
        """
        self.opensearch_domain_region = opensearch_domain_region
        self.service_type = service_type

    def _get_connector_body(self, model_type, region):
        """
        Get the connectory body
        """
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
                "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
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
            return self.input_custom_model_details(region)

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

    def _get_connector_role_policy(self, model_type, connector_body):
        """
        Get the IAM role policy for the connector
        """
        # Get the model arn
        model_arn = (
            input("Enter your custom model ARN: ").strip()
            if model_type == "9"
            else f"arn:aws:bedrock:*::foundation-model/{connector_body['parameters']['model']}"
        )

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
        helper,
        save_config_method,
        connector_role_prefix=None,
        region=None,
        model_name=None,
        model_arn=None,
        connector_body=None,
    ):
        """
        Create Bedrock connector.
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

        if self.service_type == "amazon-opensearch-service":
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
            setup = Setup()
            print("\nPlease enter your AWS credentials:")
            connector_body["credential"] = {
                "access_key": setup.get_password_with_asterisks(
                    "Enter your AWS Access Key ID: "
                ),
                "secret_key": setup.get_password_with_asterisks(
                    "Enter your AWS Secret Access Key: "
                ),
                "session_token": setup.get_password_with_asterisks(
                    "Enter your AWS Session Token: "
                ),
            }

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
                    if self.service_type == "amazon-opensearch-service"
                    else None
                ),
                (
                    connector_role_arn
                    if self.service_type == "amazon-opensearch-service"
                    else None
                ),
            )
            return True
        print(f"{Fore.RED}Failed to create Bedrock connector.{Style.RESET_ALL}")
        return False
