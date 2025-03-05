# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.connector.ml_models.model_base import ModelBase


class DeepSeekModel(ModelBase):

    def create_deepseek_connector(self, helper, config, save_config_method):
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

        # Prompt for necessary input
        api_key = input("Enter your DeepSeek API key: ").strip()

        create_connector_role_name = "create_deepseek_connector_role"

        # Default connector input
        default_connector_input = {
            "name": "DeepSeek Connector",
            "description": "Test connector for DeepSeek Chat",
            "version": "1",
            "protocol": "http",
            "parameters": {"model": "deepseek-chat"},
            "credential": {"deepSeek_key": api_key},
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": "https://api.deepseek.com/v1/chat/completions",
                    "headers": {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                    "request_body": '{ "model": "${parameters.model}", "messages": ${parameters.messages} }',
                }
            ],
        }

        # Get model details from user
        create_connector_input = self.get_custom_model_details(default_connector_input)
        if not create_connector_input:
            return  # Abort if no valid input

        # Create connector
        print("Creating DeepSeek connector...")
        connector_id = helper.create_connector(
            create_connector_role_name=create_connector_role_name,
            payload=create_connector_input,
        )

        if connector_id:
            print(
                f"{Fore.GREEN}\nSuccessfully created DeepSeek connector with ID: {connector_id}{Style.RESET_ALL}"
            )
            # Update config with connector ID if needed
            config["connector_id"] = connector_id
            save_config_method(config)
            return True
        else:
            print(f"{Fore.RED}Failed to create DeepSeek connector.{Style.RESET_ALL}")
            return False
