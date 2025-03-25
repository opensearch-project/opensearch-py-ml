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


class OpenAIModel(ModelBase):
    def __init__(
        self,
        service_type,
    ):
        """
        Initializes the OpenAI model with necessary configuration.
        """
        self.service_type = service_type

    def create_openai_connector(
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

        setup = Setup()
        if not api_key:
            openai_api_key = setup.get_password_with_asterisks(
                "Enter your OpenAI API key: "
            )
        else:
            openai_api_key = api_key

        connector_role_name = ""
        connector_role_arn = ""
        if self.service_type == "amazon-opensearch-service":
            # Prompt for necessary inputs
            if model_name == "Embedding model":
                model_type = "1"
            elif model_name == "Custom model":
                model_type = "2"
            else:
                print("\nPlease select a model for the connector creation: ")
                print("1. Embedding model")
                print("2. Custom model")
                model_type = input("Enter your choice (1-2): ").strip()

            if not connector_role_prefix:
                connector_role_prefix = (
                    input("Enter your connector role prefix: ") or None
                )
                if not connector_role_prefix:
                    raise ValueError("Connector role prefix cannot be empty.")

            id = str(uuid.uuid1())[:8]
            connector_role_name = f"{connector_role_prefix}_openai_embedding_model_{id}"
            create_connector_role_name = (
                f"{connector_role_prefix}_openai_embedding_model_create_{id}"
            )

            if not secret_name:
                secret_name = input(
                    "Enter a name for the AWS Secrets Manager secret: "
                ).strip()
            secret_name = f"{secret_name}_{id}"
            secret_key = "openai_api_key"
            secret_value = {secret_key: openai_api_key}

            if model_type == "1":
                connector_payload = {
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
                                "Authorization": "${auth}",
                            },
                            "request_body": '{ "input": ${parameters.input}, "model": "${parameters.model}" }',
                            "pre_process_function": "connector.pre_process.openai.embedding",
                            "post_process_function": "connector.post_process.openai.embedding",
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

            auth_value = f"Bearer {openai_api_key}"
            connector_payload = json.loads(
                json.dumps(connector_payload).replace("${auth}", auth_value)
            )

            # Create connector
            print("Creating OpenAI connector...")
            connector_id, connector_role_arn = helper.create_connector_with_secret(
                secret_name,
                secret_value,
                connector_role_name,
                create_connector_role_name,
                connector_payload,
                sleep_time_in_seconds=10,
            )

        else:
            # Prompt for necessary inputs
            if model_name == "Chat model":
                model_type = "1"
            elif model_name == "Embedding model":
                model_type = "2"
            elif model_name == "Custom model":
                model_type = "3"
            else:
                print("\nPlease select a model for the connector creation: ")
                print("1. Chat model: gpt-3.5-turbo")
                print("2. Embedding model: text-embedding-ada-002")
                print("3. Custom model")
                model_type = input("Enter your choice (1-3): ").strip()

            if model_type == "1":
                connector_payload = {
                    "name": "OpenAI Connector",
                    "description": "The connector to the OpenAI chat model",
                    "version": "1",
                    "protocol": "http",
                    "parameters": {"model": "gpt-3.5-turbo"},
                    "credential": {"openAI_key": "${credentials}"},
                    "actions": [
                        {
                            "action_type": "predict",
                            "method": "POST",
                            "url": "https://api.openai.com/v1/chat/completions",
                            "headers": {"Authorization": "${auth}"},
                            "request_body": '{ "model": "${parameters.model}", "messages": ${parameters.messages} }',
                        }
                    ],
                }
            elif model_type == "2":
                connector_payload = {
                    "name": "OpenAI Connector",
                    "description": "The connector to the OpenAI embedding model",
                    "version": "1",
                    "protocol": "http",
                    "parameters": {"model": "text-embedding-ada-002"},
                    "credential": {"openAI_key": "${credentials}"},
                    "actions": [
                        {
                            "action_type": "predict",
                            "method": "POST",
                            "url": "https://api.openai.com/v1/embeddings",
                            "headers": {"Authorization": "${auth}"},
                            "request_body": '{ "input": ${parameters.input}, "model": "${parameters.model}" }',
                            "pre_process_function": "connector.pre_process.openai.embedding",
                            "post_process_function": "connector.post_process.openai.embedding",
                        }
                    ],
                }
            elif model_type == "3":
                if not connector_payload:
                    connector_payload = self.input_custom_model_details(external=True)
            else:
                print(
                    f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
                )
                if not connector_payload:
                    connector_payload = self.input_custom_model_details(external=True)

            auth_value = f"Bearer {openai_api_key}"
            connector_payload = json.loads(
                json.dumps(connector_payload).replace("${auth}", auth_value)
            )
            credential_value = openai_api_key
            connector_payload = json.loads(
                json.dumps(connector_payload).replace("${credential}", credential_value)
            )

            # Create connector
            print("Creating OpenAI connector...")
            connector_id = helper.create_connector(
                create_connector_role_name=None,
                payload=connector_payload,
            )

        if connector_id:
            print(
                f"{Fore.GREEN}\nSuccessfully created OpenAI connector with ID: {connector_id}{Style.RESET_ALL}"
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
            print(f"{Fore.RED}Failed to create OpenAI connector.{Style.RESET_ALL}")
            return False
