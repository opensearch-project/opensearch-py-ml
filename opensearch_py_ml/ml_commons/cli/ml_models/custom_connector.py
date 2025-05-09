# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
from typing import Any, Callable, Dict, Optional

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ai_connector_helper import AIConnectorHelper
from opensearch_py_ml.ml_commons.cli.ml_models.model_base import ModelBase


class CustomConnector(ModelBase):
    def __init__(self, service_type):
        """
        Initializes the custom connector with necessary configuration.
        """
        self.service_type = service_type

    def create_connector(
        self,
        helper: AIConnectorHelper,
        save_config_method: Callable[[str, Dict[str, Any]], None],
        connector_role_prefix: Optional[str] = None,
        connector_secret_name: Optional[str] = None,
        connector_role_inline_policy: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        required_policy: Optional[bool] = None,
        required_secret: Optional[bool] = None,
        connector_body: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Create Custom connector.

        Args:
            helper: Helper instance for OpenSearch connector operations.
            save_config_method: Method to save connector configuration after creation.
            connector_role_prefix (optional): Prefix for role names.
            connector_secret_name (optional): The connector secret name.
            connector_role_inline_policy (optional): The connector inline policy.
            model_name (optional): Model name.
            api_key (optional): Model API key.
            required_policy (optional): Whether to configure IAM inline policy.
            required_secret (optional): Whether to configure AWS Secrets Manager.
            connector_body (optional): The connector request body.
        Returns:
            bool: True if connector creation successful, False otherwise.
        """
        # Prompt for connector body
        connector_body = connector_body or self.input_custom_model_details(
            external=True
        )

        if self.service_type == self.AMAZON_OPENSEARCH_SERVICE:
            # Prompt for model name
            model_name = model_name or input("Enter your model name: ")

            # Create connector role
            connector_role_name, create_connector_role_name = (
                self.create_connector_role(connector_role_prefix, model_name)
            )

            # Handle the connector role inline policy
            required_policy = (
                input("Do you want to set the connector role inline policy? (yes/no): ")
                .strip()
                .lower()
                if required_policy is None
                else required_policy
            )
            if required_policy == "yes" and not connector_role_inline_policy:
                print(
                    "Enter your connector role inline policy as a JSON object (press Enter twice when done): "
                )
                json_input = ""
                lines = []
                while True:
                    line = input()
                    if line.strip() == "":
                        break
                    lines.append(line)

                json_input = "\n".join(lines)
                connector_role_inline_policy = json.loads(json_input)

            # Handle secret configuration
            required_secret = (
                input("Do you want to set the connector secret? (yes/no): ")
                .strip()
                .lower()
                if required_secret is None
                else required_secret
            )

            if required_secret == "yes" or required_secret == True:
                api_key = self.set_api_key(api_key, model_name)

                connector_secret_name, secret_value = self.create_secret_name(
                    connector_secret_name, model_name, api_key
                )

                auth_value = f"Bearer {api_key}"
                connector_body = json.loads(
                    json.dumps(connector_body).replace("${auth}", auth_value)
                )

            # Create connector
            print("\nCreating connector...")
            if connector_role_inline_policy:
                connector_id, connector_role_arn, _ = helper.create_connector_with_role(
                    connector_role_inline_policy,
                    connector_role_name,
                    create_connector_role_name,
                    connector_body,
                    sleep_time_in_seconds=10,
                )
                connector_secret_arn = None
            elif connector_secret_name:
                connector_id, connector_role_arn, connector_secret_arn = (
                    helper.create_connector_with_secret(
                        connector_secret_name,
                        secret_value,
                        connector_role_name,
                        create_connector_role_name,
                        connector_body,
                        sleep_time_in_seconds=10,
                    )
                )
        else:
            # Create connector
            print("\nCreating connector...")
            connector_id = helper.create_connector(
                create_connector_role_name=None,
                body=connector_body,
            )

        if connector_id:
            print(
                f"{Fore.GREEN}\nSuccessfully created connector with ID: {connector_id}{Style.RESET_ALL}"
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
                (
                    connector_secret_name
                    if self.service_type == self.AMAZON_OPENSEARCH_SERVICE
                    else None
                ),
                (
                    connector_secret_arn
                    if self.service_type == self.AMAZON_OPENSEARCH_SERVICE
                    else None
                ),
            )
            return True
        else:
            print(f"{Fore.RED}Failed to create connector.{Style.RESET_ALL}")
            return False
