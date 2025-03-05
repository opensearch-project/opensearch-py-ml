# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.connector.ml_models.model_base import ModelBase


class OpenAIModel(ModelBase):
    def __init__(
        self,
        service_type,
    ):
        """
        Initializes the OpenAI model with necessary configuration.
        """
        self.service_type = service_type

    def create_openai_connector(self, helper, config, save_config_method):
        """
        Create OpenAI connector.
        """
        # Set trusted connector endpoints for OpenAI
        settings_body = {
            "persistent": {
                "plugins.ml_commons.trusted_connector_endpoints_regex": [
                    "^https://api\\.openai\\.com/.*$"
                ]
            }
        }
        helper.opensearch_client.cluster.put_settings(body=settings_body)

        openai_api_key = input("Enter your OpenAI API key: ").strip()

        connector_role_name = "openai_connector_role"
        create_connector_role_name = "create_openai_connector_role"

        if self.service_type == "managed":
            # Prompt for necessary inputs
            secret_name = input(
                "Enter a name for the AWS Secrets Manager secret: "
            ).strip()
            secret_key = "openai_api_key"
            secret_value = {secret_key: openai_api_key}

            default_connector_input = {
                "name": "OpenAI embedding model connector",
                "description": "Connector for OpenAI embedding model",
                "version": "1.0",
                "protocol": "http",
                "parameters": {"model": "text-embedding-ada-002"},
                "actions": [
                    {
                        "action_type": "predict",
                        "method": "POST",
                        "url": "https://api.openai.com/v1/embeddings",
                        "headers": {
                            "Authorization": "Bearer ${credential.secretArn.openai_api_key}"
                        },
                        "request_body": '{ "input": ${parameters.input}, "model": "${parameters.model}" }',
                        "pre_process_function": "connector.pre_process.openai.embedding",
                        "post_process_function": "connector.post_process.openai.embedding",
                    }
                ],
            }

            # Get model details from user
            create_connector_input = self.get_custom_model_details(
                default_connector_input
            )
            if not create_connector_input:
                return  # Abort if no valid input

            # Create connector
            print("Creating OpenAI connector...")
            connector_id = helper.create_connector_with_secret(
                secret_name,
                secret_value,
                connector_role_name,
                create_connector_role_name,
                create_connector_input,
                sleep_time_in_seconds=10,
            )

        else:
            # Prompt for necessary inputs
            print("\nPlease select a model for the connector creation: ")
            print("1. Chat model: gpt-3.5-turbo")
            print("2. Embedding model: text-embedding-ada-002")
            model_type = input("Enter your choice (1-2): ").strip()

            if model_type == "1":
                default_connector_input = {
                    "name": "OpenAI Connector",
                    "description": "The connector to the OpenAI chat model",
                    "version": "1",
                    "protocol": "http",
                    "parameters": {"model": "gpt-3.5-turbo"},
                    "credential": {"openAI_key": openai_api_key},
                    "actions": [
                        {
                            "action_type": "predict",
                            "method": "POST",
                            "url": "https://api.openai.com/v1/chat/completions",
                            "headers": {"Authorization": f"Bearer {openai_api_key}"},
                            "request_body": '{ "model": "${parameters.model}", "messages": ${parameters.messages} }',
                        }
                    ],
                }
            elif model_type == "2":
                default_connector_input = {
                    "name": "OpenAI Connector",
                    "description": "The connector to the OpenAI embedding model",
                    "version": "1",
                    "protocol": "http",
                    "parameters": {"model": "text-embedding-ada-002"},
                    "credential": {"openAI_key": openai_api_key},
                    "actions": [
                        {
                            "action_type": "predict",
                            "method": "POST",
                            "url": "https://api.openai.com/v1/embeddings",
                            "headers": {"Authorization": f"Bearer {openai_api_key}"},
                            "request_body": '{ "input": ${parameters.input}, "model": "${parameters.model}" }',
                            "pre_process_function": "connector.pre_process.openai.embedding",
                            "post_process_function": "connector.post_process.openai.embedding",
                        }
                    ],
                }
            else:
                print(
                    f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Chat model'.{Style.RESET_ALL}"
                )
                default_connector_input = {
                    "name": "OpenAI Connector",
                    "description": "The connector to the OpenAI chat model",
                    "version": "1",
                    "protocol": "http",
                    "parameters": {"model": "gpt-3.5-turbo"},
                    "credential": {"openAI_key": openai_api_key},
                    "actions": [
                        {
                            "action_type": "predict",
                            "method": "POST",
                            "url": "https://api.openai.com/v1/chat/completions",
                            "headers": {"Authorization": f"Bearer {openai_api_key}"},
                            "request_body": '{ "model": "${parameters.model}", "messages": ${parameters.messages} }',
                        }
                    ],
                }

            # Get model details from user
            create_connector_input = self.get_custom_model_details(
                default_connector_input
            )
            if not create_connector_input:
                return  # Abort if no valid input

            # Create connector
            print("Creating OpenAI connector...")
            connector_id = helper.create_connector(
                create_connector_role_name=create_connector_role_name,
                payload=create_connector_input,
            )

        if connector_id:
            print(
                f"{Fore.GREEN}\nSuccessfully created OpenAI connector with ID: {connector_id}{Style.RESET_ALL}"
            )
            # Update config with connector ID if needed
            config["connector_id"] = connector_id
            save_config_method(config)
            return True
        else:
            print(f"{Fore.RED}Failed to create OpenAI connector.{Style.RESET_ALL}")
            return False
