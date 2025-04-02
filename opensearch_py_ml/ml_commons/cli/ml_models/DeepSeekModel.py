# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import uuid

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.model_base import ModelBase
from opensearch_py_ml.ml_commons.cli.ml_setup import Setup


class DeepSeekModel(ModelBase):

    def __init__(
        self,
        service_type,
    ):
        """
        Initializes the DeepSeek model with necessary configuration.
        """
        self.service_type = service_type

    def create_deepseek_connector(
        self,
        helper,
        save_config_method,
        connector_role_prefix=None,
        model_name=None,
        api_key=None,
        connector_payload=None,
        secret_name=None,
    ):
        """
        Create DeepSeek connector.
        """
        # Set trusted connector endpoints for DeepSeek
        settings_body = {
            "persistent": {
                "plugins.ml_commons.trusted_connector_endpoints_regex": [
                    "^https://api\\.deepseek\\.com/.*$"
                ]
            }
        }
        helper.opensearch_client.cluster.put_settings(body=settings_body)

        if model_name == "DeepSeek Chat model":
            model_type = "1"
        elif model_name == "Custom model":
            model_type = "2"
        else:
            print("\nPlease select a model for the connector creation: ")
            print("1. DeepSeek Chat model")
            print("2. Custom model")
            model_type = input("Enter your choice (1-2): ").strip()

        # Prompt for necessary input
        setup = Setup()
        if not api_key:
            deepseek_api_key = setup.get_password_with_asterisks(
                "Enter your DeepSeek API key: "
            )
        else:
            deepseek_api_key = api_key

        connector_role_name = ""
        connector_role_arn = ""
        if self.service_type == "amazon-opensearch-service":
            # Prompt for necessary inputs
            if not connector_role_prefix:
                connector_role_prefix = (
                    input("Enter your connector role prefix: ") or None
                )
                if not connector_role_prefix:
                    raise ValueError("Connector role prefix cannot be empty.")

            # add unique random id to avoid permission error
            id = str(uuid.uuid1())[:8]
            connector_role_name = f"{connector_role_prefix}_deepseek_connector_{id}"
            create_connector_role_name = (
                f"{connector_role_prefix}_deepseek_connector_create_{id}"
            )

            if not secret_name:
                secret_name = input(
                    "Enter a name for the AWS Secrets Manager secret: "
                ).strip()

            secret_name = f"{secret_name}_{id}"
            secret_key = "deepseek_api_key"
            secret_value = {secret_key: deepseek_api_key}

            if model_type == "1":
                connector_payload = {
                    "name": "DeepSeek Chat",
                    "description": "Test connector for DeepSeek Chat",
                    "version": "1",
                    "protocol": "http",
                    "parameters": {"model": "deepseek-chat"},
                    "actions": [
                        {
                            "action_type": "predict",
                            "method": "POST",
                            "url": "https://api.deepseek.com/v1/chat/completions",
                            "headers": {
                                "Content-Type": "application/json",
                                "Authorization": "${auth}",
                            },
                            "request_body": '{ "model": "${parameters.model}", "messages": ${parameters.messages} }',
                        }
                    ],
                }
            elif model_type == "2":
                if not connector_payload:
                    connector_payload = self.input_custom_model_details(external=True)
            else:
                print(
                    f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
                )
                if not connector_payload:
                    connector_payload = self.input_custom_model_details(external=True)

            auth_value = f"Bearer {deepseek_api_key}"
            connector_payload = json.loads(
                json.dumps(connector_payload).replace("${auth}", auth_value)
            )

            # Create connector
            print("\nCreating DeepSeek connector...")
            connector_id, connector_role_arn = helper.create_connector_with_secret(
                secret_name,
                secret_value,
                connector_role_name,
                create_connector_role_name,
                connector_payload,
                sleep_time_in_seconds=10,
            )
        else:
            if model_type == "1":
                connector_payload = {
                    "name": "DeepSeek Chat",
                    "description": "Test connector for DeepSeek Chat",
                    "version": "1",
                    "protocol": "http",
                    "parameters": {"model": "deepseek-chat"},
                    "credential": {"deepSeek_key": "${credentials}"},
                    "actions": [
                        {
                            "action_type": "predict",
                            "method": "POST",
                            "url": "https://api.deepseek.com/v1/chat/completions",
                            "headers": {
                                "Content-Type": "application/json",
                                "Authorization": "${auth}",
                            },
                            "request_body": '{ "model": "${parameters.model}", "messages": ${parameters.messages} }',
                        }
                    ],
                }
            elif model_type == "2":
                if not connector_payload:
                    connector_payload = self.input_custom_model_details(external=True)
            else:
                print(
                    f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
                )
                if not connector_payload:
                    connector_payload = self.input_custom_model_details(external=True)

            auth_value = f"Bearer {deepseek_api_key}"
            connector_payload = json.loads(
                json.dumps(connector_payload).replace("${auth}", auth_value)
            )
            credential_value = deepseek_api_key
            connector_payload = json.loads(
                json.dumps(connector_payload).replace("${credential}", credential_value)
            )

            # Create connector
            print("\nCreating DeepSeek connector...")
            connector_id = helper.create_connector(
                create_connector_role_name=None,
                payload=connector_payload,
            )

        if connector_id:
            print(
                f"{Fore.GREEN}\nSuccessfully created DeepSeek connector with ID: {connector_id}{Style.RESET_ALL}"
            )
            connector_output = helper.get_connector(connector_id)
            save_config_method(
                connector_id,
                connector_output,
                connector_role_name,
                secret_name,
                connector_role_arn,
            )
            return True
        else:
            print(f"{Fore.RED}Failed to create DeepSeek connector.{Style.RESET_ALL}")
            return False
