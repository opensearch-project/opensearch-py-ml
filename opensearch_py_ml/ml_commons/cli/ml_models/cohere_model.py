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


class CohereModel(ModelBase):
    def __init__(
        self,
        service_type,
    ):
        """
        Initializes the Cohere model with necessary configuration.
        """
        self.service_type = service_type

    def _get_connector_body(self, model_type: str) -> Dict[str, Any]:
        """
        Get the connectory body.
        """
        connector_configs = {
            self.AMAZON_OPENSEARCH_SERVICE: {
                "1": {
                    "name": "Cohere Embedding Model Connector",
                    "description": "Connector for Cohere embedding model",
                    "model": "embed-english-v3.0",
                    "url": "https://api.cohere.ai/v1/embed",
                    "request_body": '{ "texts": ${parameters.texts}, "truncate": "${parameters.truncate}", "model": "${parameters.model}", "input_type": "${parameters.input_type}" }',
                    "pre_process_function": "connector.pre_process.cohere.embedding",
                    "post_process_function": "connector.post_process.cohere.embedding",
                    "parameters": {
                        "input_type": "search_document",
                        "truncate": "END",
                    },
                },
                "2": "Custom model",
            },
            self.OPEN_SOURCE: {
                "1": {
                    "name": "Cohere Chat model",
                    "description": "The connector to Cohere's public chat API",
                    "model": "command",
                    "credential": {"cohere_key": "${credential}"},
                    "url": "https://api.cohere.ai/v1/chat",
                    "request_body": '{ "message": "${parameters.message}", "model": "${parameters.model}" }',
                    "parameters": {},
                },
                "2": {
                    "name": "Cohere Embedding model",
                    "description": "The connector to Cohere's public embed API",
                    "model": "embed-english-v3.0",
                    "credential": {"cohere_key": "${credential}"},
                    "url": "https://api.cohere.ai/v1/embed",
                    "request_body": '{ "texts": ${parameters.texts}, "truncate": "${parameters.truncate}", "model": "${parameters.model}", "input_type": "${parameters.input_type}" }',
                    "pre_process_function": "connector.pre_process.cohere.embedding",
                    "post_process_function": "connector.post_process.cohere.embedding",
                    "parameters": {
                        "input_type": "search_document",
                        "truncate": "END",
                    },
                },
                "3": {
                    "name": "Cohere Image Embed model",
                    "description": "The connector to Cohere's public embed API",
                    "model": "embed-english-v3.0",
                    "credential": {"cohere_key": "${credential}"},
                    "url": "https://api.cohere.ai/v1/embed",
                    "request_body": '{ "images": ${parameters.images}, "truncate": "${parameters.truncate}", "model": "${parameters.model}", "input_type": "${parameters.input_type}" }',
                    "pre_process_function": "connector.pre_process.cohere.multimodal_embedding",
                    "post_process_function": "connector.post_process.cohere.embedding",
                    "parameters": {
                        "input_type": "image",
                        "truncate": "END",
                    },
                },
                "4": "Custom model",
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
            "version": 1,
            "protocol": "http",
            "parameters": parameters,
            **({"credential": config["credential"]} if "credential" in config else {}),
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "headers": {
                        "Authorization": "${auth}",
                        "Request-Source": "unspecified:opensearch",
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
        helper: AIConnectorHelper,
        save_config_method: Callable[[str, Dict[str, Any]], None],
        connector_role_prefix: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        connector_body: Optional[Dict[str, Any]] = None,
        connector_secret_name: Optional[str] = None,
    ) -> bool:
        """
        Create Cohere connector.

        Args:
            helper: Helper instance for OpenSearch connector operations.
            save_config_method: Method to save connector configuration after creation.
            connector_role_prefix (optional): Prefix for role names.
            model_name (optional): Specific Cohere model name.
            api_key (optional): Cohere key.
            connector_body (optional): The connector request body.
            connector_secret_name (optional): The connector secret name.

        Returns:
            bool: True if connector creation successful, False otherwise.
        """
        # Set trusted connector endpoints for Cohere
        trusted_endpoint = "^https://api\\.cohere\\.ai/.*$"
        self.set_trusted_endpoint(helper, trusted_endpoint)

        # Prompt to choose model
        model_type = self.get_model_details("Cohere", self.service_type, model_name)

        # Prompt for API key
        cohere_api_key = self.set_api_key(api_key, "Cohere")

        # Get connector body
        connector_body = connector_body or self._get_connector_body(model_type)

        auth_value = f"Bearer {cohere_api_key}"
        connector_body = json.loads(
            json.dumps(connector_body).replace("${auth}", auth_value)
        )

        if self.service_type == self.AMAZON_OPENSEARCH_SERVICE:
            # Create connector role and secret name
            connector_role_name, create_connector_role_name = (
                self.create_connector_role(connector_role_prefix, "cohere")
            )
            connector_secret_name, secret_value = self.create_secret_name(
                connector_secret_name, "cohere", cohere_api_key
            )

            # Create connector
            print("\nCreating Cohere connector...")
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
            credential_value = cohere_api_key
            connector_body = json.loads(
                json.dumps(connector_body).replace("${credential}", credential_value)
            )

            # Create connector
            print("\nCreating Cohere connector...")
            connector_id = helper.create_connector(
                create_connector_role_name=None,
                body=connector_body,
            )

        if connector_id:
            print(
                f"{Fore.GREEN}\nSuccessfully created Cohere connector with ID: {connector_id}{Style.RESET_ALL}"
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
            print(f"{Fore.RED}Failed to create Cohere connector.{Style.RESET_ALL}")
            return False
