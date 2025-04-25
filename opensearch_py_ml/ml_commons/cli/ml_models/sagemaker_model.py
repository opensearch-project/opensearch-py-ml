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


class SageMakerModel(ModelBase):
    def __init__(
        self,
        opensearch_domain_region,
        service_type,
    ):
        """
        Initializes the SageMaker model with necessary configurations
        """
        self.opensearch_domain_region = opensearch_domain_region
        self.service_type = service_type

    def _get_connector_body(
        self, model_type: str, region: str, endpoint_url: str
    ) -> Dict[str, Any]:
        """
        Get the connectory body.
        """
        connector_configs = {
            self.AMAZON_OPENSEARCH_SERVICE: {
                "1": {
                    "name": "Amazon SageMaker: DeepSeek R1 model",
                    "description": "The connector to SageMaker for DeepSeek R1 model",
                    "request_body": '{ "inputs": "${parameters.inputs}", "parameters": {"do_sample": ${parameters.do_sample}, "top_p": ${parameters.top_p}, "temperature": ${parameters.temperature}, "max_new_tokens": ${parameters.max_new_tokens}} }',
                    "post_process_function": "\n      if (params.result == null || params.result.length == 0) {\n        throw new Exception('No response available');\n      }\n      \n      def completion = params.result[0].generated_text;\n      return '{' +\n               '\"name\": \"response\",'+\n               '\"dataAsMap\": {' +\n                  '\"completion\":\"' + escape(completion) + '\"}' +\n             '}';\n    ",
                    "parameters": {
                        "do_sample": "true",
                        "top_p": 0.9,
                        "temperature": 0.7,
                        "max_new_tokens": 512,
                    },
                },
                "2": {
                    "name": "Amazon SageMaker: Embedding model",
                    "description": "The connector to SageMaker for embedding model",
                    "request_body": "${parameters.input}",
                    "pre_process_function": "connector.pre_process.default.embedding",
                    "post_process_function": "connector.post_process.default.embedding",
                    "parameters": {},
                },
                "3": "Custom model",
            },
            self.OPEN_SOURCE: {
                "1": {
                    "name": "Amazon SageMaker: Embedding model",
                    "description": "The connector to SageMaker for embedding model",
                    "request_body": "${parameters.input}",
                    "pre_process_function": "connector.pre_process.default.embedding",
                    "post_process_function": "connector.post_process.default.embedding",
                    "parameters": {},
                },
                "2": "Custom model",
            },
        }

        service_configs = connector_configs.get(self.service_type)

        # Handle custom model or invalid choice
        if (
            model_type not in service_configs
            or service_configs[model_type] == "Custom model"
        ):
            if model_type not in service_configs:
                print(
                    f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
                )
            return self.input_custom_model_details()

        config = service_configs[model_type]

        # Base parameters that all connectors need
        base_parameters = {
            "region": region,
            "service_name": "sagemaker",
        }

        # Merge with model-specific parameters if any
        parameters = {**base_parameters, **config.get("parameters", {})}

        # Return the connector body
        return {
            "name": config["name"],
            "description": config["description"],
            "version": "1.0",
            "protocol": "aws_sigv4",
            "parameters": parameters,
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "headers": {"content-type": "application/json"},
                    "url": endpoint_url,
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

    def create_connector(
        self,
        helper: AIConnectorHelper,
        save_config_method: Callable[[str, Dict[str, Any]], None],
        connector_role_prefix: Optional[str] = None,
        region: Optional[str] = None,
        model_name: Optional[str] = None,
        endpoint_arn: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        connector_body: Optional[str] = None,
        aws_access_key: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ) -> bool:
        """
        Create SageMaker connector.

        Args:
            helper: Helper instance for OpenSearch connector operations.
            save_config_method: Method to save connector configuration after creation.
            connector_role_prefix (optional): Prefix for role names.
            region (optional): AWS region.
            model_name (optional): Specific SageMaker model name.
            endpoint_arn (optional): SageMaker endpoint ARN.
            endpoint_url (optional): SageMaker endpoint URL.
            connector_body (optional): The connector request body.
            aws_access_key (optional): AWS access key ID.
            aws_secret_access_key (optional): AWS secet access key.
            aws_session_token (optional): AWS session token.

        Returns:
            bool: True if connector creation successful, False otherwise.
        """
        # Set trusted connector endpoints for SageMaker
        trusted_endpoint = (
            "^https://runtime\\.sagemaker\\..*[a-z0-9-]\\.amazonaws\\.com/.*$"
        )
        self.set_trusted_endpoint(helper, trusted_endpoint)

        # Prompt to choose model
        model_type = self.get_model_details(
            "Amazon SageMaker", self.service_type, model_name
        )

        endpoint_arn = (
            endpoint_arn
            or input("Enter your SageMaker inference endpoint ARN: ").strip()
        )
        region = (
            region
            or input(
                f"Enter your SageMaker region [{self.opensearch_domain_region}]: "
            ).strip()
            or self.opensearch_domain_region
        )
        endpoint_url = (
            endpoint_url
            or input("Enter your SageMaker inference endpoint URL: ").strip()
        )

        # Get connector body
        connector_body = connector_body or self._get_connector_body(
            model_type, region, endpoint_url
        )

        if self.service_type == self.AMAZON_OPENSEARCH_SERVICE:
            # Create connector role
            connector_role_name, create_connector_role_name = (
                self.create_connector_role(connector_role_prefix, "sagemaker")
            )

            connector_role_inline_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Action": ["sagemaker:InvokeEndpoint"],
                        "Effect": "Allow",
                        "Resource": endpoint_arn,
                    }
                ],
            }

            # Create connector
            print("\nCreating SageMaker connector...")
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
            print("\nCreating SageMaker connector...")
            connector_id = helper.create_connector(
                create_connector_role_name=None,
                body=connector_body,
            )

        if connector_id:
            print(
                f"{Fore.GREEN}\nSuccessfully created SageMaker connector with ID: {connector_id}{Style.RESET_ALL}"
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
        else:
            print(f"{Fore.RED}Failed to create SageMaker connector.{Style.RESET_ALL}")
            return False
