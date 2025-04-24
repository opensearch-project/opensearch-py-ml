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


class TextractModel(ModelBase):
    def __init__(self, opensearch_domain_region, service_type):
        """
        Initializes the Textract model with necessary configurations
        """
        self.opensearch_domain_region = opensearch_domain_region
        self.service_type = service_type

    def _get_connector_body(self, model_type: str, region: str) -> Dict[str, Any]:
        """
        Get the connectory body.
        """
        connector_configs = {
            "1": {
                "name": "Amazon Textract connector: detect document texts",
                "description": "The connector to Amazon Textract for detect document text",
                "request_body": '{  "Document": { "Bytes": "${parameters.bytes}" }  } ',
                "parameters": {
                    "api_name": "DetectDocumentText",
                },
            },
            "2": "Custom model",
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
            "service_name": "textract",
            "api": "Textract.${parameters.api_name}",
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
                    "headers": {
                        "content-type": "application/x-amz-json-1.1",
                        "X-Amz-Target": "${parameters.api}",
                    },
                    "url": "https://${parameters.service_name}.${parameters.region}.amazonaws.com",
                    "request_body": config["request_body"],
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
        Create Textract connector.

        Args:
            helper: Helper instance for OpenSearch connector operations.
            save_config_method: Method to save connector configuration after creation.
            connector_role_prefix (optional): Prefix for role names.
            region (optional): AWS region.
            model_name (optional): Specific Textract model name.
            connector_body (optional): The connector request body.
            aws_access_key (optional): AWS access key ID.
            aws_secret_access_key (optional): AWS secet access key.
            aws_session_token (optional): AWS session token.

        Returns:
            bool: True if connector creation successful, False otherwise.
        """
        # Set trusted connector endpoints for Textract
        trusted_endpoint = "^https://textract\\..*[a-z0-9-]\\.amazonaws\\.com$"
        self.set_trusted_endpoint(helper, trusted_endpoint)

        # Prompt to choose model
        model_type = self.get_model_details(
            "Amazon Textract", self.service_type, model_name
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
                self.create_connector_role(connector_role_prefix, "textract")
            )

            connector_role_inline_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Action": [
                            f"textract:{connector_body['parameters']['api_name']}"
                        ],
                        "Effect": "Allow",
                        "Resource": "*",
                    }
                ],
            }

            # Create connector
            print("\nCreating Textract connector...")
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
            print("\nCreating Textract connector...")
            connector_id = helper.create_connector(
                create_connector_role_name=None,
                body=connector_body,
            )

        if connector_id:
            print(
                f"{Fore.GREEN}\nSuccessfully created Textract connector with ID: {connector_id}{Style.RESET_ALL}"
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
            print(f"{Fore.RED}Failed to create Textract connector.{Style.RESET_ALL}")
            return False
