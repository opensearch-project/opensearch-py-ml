# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.model_base import ModelBase


class CohereModel(ModelBase):

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
        Create Cohere connector.
        """
        # Set trusted connector endpoints for Cohere
        trusted_endpoint = "^https://api\\.cohere\\.ai/.*$"
        self.set_trusted_endpoint(helper, trusted_endpoint)

        # Prompt to choose model
        model_type = self.get_model_details(
            "Cohere", "amazon-opensearch-service", model_name
        )

        # Prompt for API key
        cohere_api_key = self.set_api_key(api_key, "Cohere")

        # Create connector role and secret name
        connector_role_name, create_connector_role_name = self.create_connector_role(
            connector_role_prefix, "cohere"
        )
        secret_name, secret_value = self.create_secret_name(
            secret_name, "cohere", cohere_api_key
        )

        if model_type == "1":
            connector_body = {
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
            if not connector_body:
                connector_body = self.input_custom_model_details(external=True)
        else:
            print(
                f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
            )
            if not connector_body:
                connector_body = self.input_custom_model_details(external=True)

        auth_value = f"Bearer {cohere_api_key}"
        connector_body = json.loads(
            json.dumps(connector_body).replace("${auth}", auth_value)
        )

        # Create connector
        print("Creating Cohere connector...")
        connector_id, connector_role_arn = helper.create_connector_with_secret(
            secret_name,
            secret_value,
            connector_role_name,
            create_connector_role_name,
            connector_body,
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
