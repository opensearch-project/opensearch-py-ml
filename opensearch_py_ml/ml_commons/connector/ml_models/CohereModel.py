# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.connector.ml_models.model_base import ModelBase


class CohereModel(ModelBase):

    def create_cohere_connector(self, helper, config, save_config_method):
        """
        Create Cohere connector.
        """
        # Prompt for necessary inputs
        secret_name = input("Enter a name for the AWS Secrets Manager secret: ").strip()
        secret_key = "cohere_api_key"
        cohere_api_key = input("Enter your Cohere API key: ").strip()
        secret_value = {secret_key: cohere_api_key}

        connector_role_name = "cohere_connector_role"
        create_connector_role_name = "create_cohere_connector_role"

        # Default connector input
        default_connector_input = {
            "name": "Cohere Embedding Model Connector",
            "description": "Connector for Cohere embedding model",
            "version": "1",
            "protocol": "http",
            "parameters": {
                "model": "embed-english-v3.0",
                "input_type": "search_document",
                "truncate": "END",
            },
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": "https://api.cohere.ai/v1/embed",
                    "headers": {
                        "Authorization": f"Bearer {cohere_api_key}",
                        "Request-Source": "unspecified:opensearch",
                    },
                    "request_body": '{ "texts": ${parameters.texts}, "truncate": "${parameters.truncate}", "model": "${parameters.model}", "input_type": "${parameters.input_type}" }',
                    "pre_process_function": "connector.pre_process.cohere.embedding",
                    "post_process_function": "connector.post_process.cohere.embedding",
                }
            ],
        }

        # Get model details from user
        create_connector_input = self.get_custom_model_details(default_connector_input)
        if not create_connector_input:
            return  # Abort if no valid input

        # Create connector
        print("Creating Cohere connector...")
        connector_id = helper.create_connector_with_secret(
            secret_name,
            secret_value,
            connector_role_name,
            create_connector_role_name,
            create_connector_input,
            sleep_time_in_seconds=10,
        )

        if connector_id:
            print(
                f"{Fore.GREEN}\nSuccessfully created Cohere connector with ID: {connector_id}{Style.RESET_ALL}"
            )
            # Update config with connector ID if needed
            config["connector_id"] = connector_id
            save_config_method(config)
            return True
        else:
            print(f"{Fore.RED}Failed to create Cohere connector.{Style.RESET_ALL}")
            return False
