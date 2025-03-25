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


class CohereModel(ModelBase):

    def create_cohere_connector(
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
        Create Cohere connector.
        """
        # Set trusted connector endpoints for Cohere
        settings_body = {
            "persistent": {
                "plugins.ml_commons.trusted_connector_endpoints_regex": [
                    "^https://api\\.cohere\\.ai/.*$"
                ]
            }
        }
        helper.opensearch_client.cluster.put_settings(body=settings_body)

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

        setup = Setup()
        if not api_key:
            cohere_api_key = setup.get_password_with_asterisks(
                "Enter your Cohere API key: "
            )
        else:
            cohere_api_key = api_key

        if not connector_role_prefix:
            connector_role_prefix = input("Enter your connector role prefix: ") or None
            if not connector_role_prefix:
                raise ValueError("Connector role prefix cannot be empty.")

        if not secret_name:
            secret_name = input(
                "Enter a name for the AWS Secrets Manager secret: "
            ).strip()

        id = str(uuid.uuid1())[:8]
        secret_name = f"{secret_name}_{id}"
        secret_key = "cohere_api_key"
        secret_value = {secret_key: cohere_api_key}

        connector_role_name = f"{connector_role_prefix}_cohere_connector_{id}"
        create_connector_role_name = (
            f"{connector_role_prefix}_cohere_connector_create_{id}"
        )

        if model_type == "1":
            connector_payload = {
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
                            "Authorization": "${auth}",
                            "Request-Source": "unspecified:opensearch",
                        },
                        "request_body": '{ "texts": ${parameters.texts}, "truncate": "${parameters.truncate}", "model": "${parameters.model}", "input_type": "${parameters.input_type}" }',
                        "pre_process_function": "connector.pre_process.cohere.embedding",
                        "post_process_function": "connector.post_process.cohere.embedding",
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

        auth_value = f"Bearer {cohere_api_key}"
        connector_payload = json.loads(
            json.dumps(connector_payload).replace("${auth}", auth_value)
        )

        # Create connector
        print("Creating Cohere connector...")
        connector_id, connector_role_arn = helper.create_connector_with_secret(
            secret_name,
            secret_value,
            connector_role_name,
            create_connector_role_name,
            connector_payload,
            sleep_time_in_seconds=10,
        )

        if connector_id:
            print(
                f"{Fore.GREEN}\nSuccessfully created Cohere connector with ID: {connector_id}{Style.RESET_ALL}"
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
            print(f"{Fore.RED}Failed to create Cohere connector.{Style.RESET_ALL}")
            return False
