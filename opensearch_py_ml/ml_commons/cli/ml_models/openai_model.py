# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.model_base import ModelBase


class OpenAIModel(ModelBase):
    def __init__(
        self,
        service_type,
    ):
        """
        Initializes the OpenAI model with necessary configuration.
        """
        self.service_type = service_type

    def _get_connector_body(self, model_type):
        """
        Get the connectory body
        """
        connector_configs = {
            "amazon-opensearch-service": {
                "1": {
                    "name": "OpenAI embedding model connector",
                    "description": "Connector for OpenAI embedding model",
                    "model": "text-embedding-ada-002",
                    "url": "https://api.openai.com/v1/embeddings",
                    "request_body": '{ "input": ${parameters.input}, "model": "${parameters.model}" }',
                    "pre_process_function": "connector.pre_process.openai.embedding",
                    "post_process_function": "connector.post_process.openai.embedding",
                    "parameters": {},
                },
                "2": "Custom model",
            },
            "open-source": {
                "1": {
                    "name": "OpenAI chat model connector",
                    "description": "The connector to the OpenAI chat model",
                    "model": "gpt-3.5-turbo",
                    "credential": {"openAI_key": "${credential}"},
                    "url": "https://api.openai.com/v1/chat/completions",
                    "request_body": '{ "model": "${parameters.model}", "messages": ${parameters.messages} }',
                    "parameters": {},
                },
                "2": {
                    "name": "OpenAI embedding model connector",
                    "description": "Connector for OpenAI embedding model",
                    "model": "text-embedding-ada-002",
                    "credential": {"openAI_key": "${credential}"},
                    "url": "https://api.openai.com/v1/embeddings",
                    "request_body": '{ "input": ${parameters.input}, "model": "${parameters.model}" }',
                    "pre_process_function": "connector.pre_process.openai.embedding",
                    "post_process_function": "connector.post_process.openai.embedding",
                    "parameters": {},
                },
                "3": "Custom model",
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
            return self.input_custom_model_details(external=True)

        config = service_configs[model_type]

        # Base parameters that all connectors need
        base_parameters = {"model": config["model"]}

        # Merge with model-specific parameters if any
        parameters = {**base_parameters, **config.get("parameters", {})}

        # Return the connector body
        return {
            "name": config["name"],
            "description": config["description"],
            "version": "1.0",
            "protocol": "http",
            "parameters": parameters,
            **({"credential": config["credential"]} if "credential" in config else {}),
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "headers": {
                        "Authorization": "${auth}",
                    },
                    "url": config["url"],
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
        helper,
        save_config_method,
        connector_role_prefix=None,
        model_name=None,
        api_key=None,
        connector_body=None,
        secret_name=None,
    ):
        """
        Create OpenAI connector.
        """
        # Set trusted connector endpoints for OpenAI
        trusted_endpoint = "^https://api\\.openai\\.com/.*$"
        self.set_trusted_endpoint(helper, trusted_endpoint)

        # Prompt to choose model
        model_type = self.get_model_details("OpenAI", self.service_type, model_name)

        # Prompt for API key
        openai_api_key = self.set_api_key(api_key, "OpenAI")

        # Get connector body
        connector_body = connector_body or self._get_connector_body(model_type)

        if self.service_type == "amazon-opensearch-service":
            # Create connector role and secret name
            connector_role_name, create_connector_role_name = (
                self.create_connector_role(connector_role_prefix, "openai")
            )
            secret_name, secret_value = self.create_secret_name(
                secret_name, "openai", openai_api_key
            )

            auth_value = f"Bearer {openai_api_key}"
            connector_body = json.loads(
                json.dumps(connector_body).replace("${auth}", auth_value)
            )

            # Create connector
            print("\nCreating OpenAI connector...")
            connector_id, connector_role_arn, connector_secret_arn = (
                helper.create_connector_with_secret(
                    secret_name,
                    secret_value,
                    connector_role_name,
                    create_connector_role_name,
                    connector_body,
                    sleep_time_in_seconds=10,
                )
            )

        else:
            auth_value = f"Bearer {openai_api_key}"
            connector_body = json.loads(
                json.dumps(connector_body).replace("${auth}", auth_value)
            )
            credential_value = openai_api_key
            connector_body = json.loads(
                json.dumps(connector_body).replace("${credential}", credential_value)
            )

            # Create connector
            print("\nCreating OpenAI connector...")
            connector_id = helper.create_connector(
                create_connector_role_name=None,
                body=connector_body,
            )

        if connector_id:
            print(
                f"{Fore.GREEN}\nSuccessfully created OpenAI connector with ID: {connector_id}{Style.RESET_ALL}"
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
                (
                    secret_name
                    if self.service_type == "amazon-opensearch-service"
                    else None
                ),
                (
                    connector_secret_arn
                    if self.service_type == "amazon-opensearch-service"
                    else None
                ),
            )
            return True
        else:
            print(f"{Fore.RED}Failed to create OpenAI connector.{Style.RESET_ALL}")
            return False
