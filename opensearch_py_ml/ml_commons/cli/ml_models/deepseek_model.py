# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.model_base import ModelBase


class DeepSeekModel(ModelBase):

    def __init__(
        self,
        service_type,
    ):
        """
        Initializes the DeepSeek model with necessary configuration.
        """
        self.service_type = service_type

    def create_connector(
        self,
        helper,
        save_config_method,
        connector_role_prefix=None,
        model_name=None,
        api_key=None,
        connector_body=None,
        secret_name=None,
    ):
        """
        Create DeepSeek connector.
        """
        # Set trusted connector endpoints for DeepSeek
        trusted_endpoint = "^https://api\\.deepseek\\.com/.*$"
        self.set_trusted_endpoint(helper, trusted_endpoint)

        # Prompt to choose model
        model_type = self.get_model_details("DeepSeek", self.service_type, model_name)

        # Prompt for API key
        deepseek_api_key = self.set_api_key(api_key, "DeepSeek")

        connector_role_arn = ""
        connector_role_name = ""
        if self.service_type == "amazon-opensearch-service":
            # Create connector role and secret name
            connector_role_name, create_connector_role_name = (
                self.create_connector_role(connector_role_prefix, "deepseek")
            )
            secret_name, secret_value = self.create_secret_name(
                secret_name, "deepseek", deepseek_api_key
            )

            if model_type == "1":
                connector_body = {
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
                if not connector_body:
                    connector_body = self.input_custom_model_details(external=True)
            else:
                print(
                    f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
                )
                if not connector_body:
                    connector_body = self.input_custom_model_details(external=True)

            auth_value = f"Bearer {deepseek_api_key}"
            connector_body = json.loads(
                json.dumps(connector_body).replace("${auth}", auth_value)
            )

            # Create connector
            print("\nCreating DeepSeek connector...")
            connector_id, connector_role_arn = helper.create_connector_with_secret(
                secret_name,
                secret_value,
                connector_role_name,
                create_connector_role_name,
                connector_body,
                sleep_time_in_seconds=10,
            )
        else:
            if model_type == "1":
                connector_body = {
                    "name": "DeepSeek Chat",
                    "description": "Test connector for DeepSeek Chat",
                    "version": "1",
                    "protocol": "http",
                    "parameters": {"model": "deepseek-chat"},
                    "credential": {"deepSeek_key": "${credential}"},
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
                if not connector_body:
                    connector_body = self.input_custom_model_details(external=True)
            else:
                print(
                    f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
                )
                if not connector_body:
                    connector_body = self.input_custom_model_details(external=True)

            auth_value = f"Bearer {deepseek_api_key}"
            connector_body = json.loads(
                json.dumps(connector_body).replace("${auth}", auth_value)
            )
            credential_value = deepseek_api_key
            connector_body = json.loads(
                json.dumps(connector_body).replace("${credential}", credential_value)
            )

            # Create connector
            print("\nCreating DeepSeek connector...")
            connector_id = helper.create_connector(
                create_connector_role_name=None,
                body=connector_body,
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
