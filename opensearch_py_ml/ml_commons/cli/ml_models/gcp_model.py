# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.model_base import ModelBase


class GCPModel(ModelBase):

    def _get_connector_body(self, model_type, project_id, model_id, access_token):
        """
        Get the connectory body
        """
        connector_configs = {
            "1": {
                "name": "VertexAI Connector",
                "description": "The connector to public vertexAI model service for text embedding",
                "request_body": '{"instances": [{ "content": "${parameters.prompt}"}]}',
            },
            "2": "Custom model",
        }

        # Handle custom model or invalid choice
        if (
            model_type not in connector_configs
            or connector_configs[model_type] == "Custom model"
        ):
            if model_type not in connector_configs:
                print(
                    f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
                )
            return self.input_custom_model_details()

        config = connector_configs[model_type]

        # Return the connector body
        return {
            "name": config["name"],
            "description": config["description"],
            "version": 1,
            "protocol": "http",
            "parameters": {"project": project_id, "model_id": model_id},
            "credential": {"vertexAI_token": access_token},
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "headers": {"Authorization": "Bearer ${credential.vertexAI_token}"},
                    "url": "https://us-central1-aiplatform.googleapis.com/v1/projects/${parameters.project}/locations/us-central1/publishers/google/models/${parameters.model_id}:predict",
                    "request_body": config["request_body"],
                }
            ],
        }

    def create_connector(
        self,
        helper,
        save_config_method,
        model_name=None,
        project_id=None,
        model_id=None,
        access_token=None,
        connector_body=None,
    ):
        """
        Create Google Cloud Platform connector.
        """
        # Set trusted connector endpoints for GCP
        trusted_endpoint = "^https://.*-aiplatform\\.googleapis\\.com/.*$"
        self.set_trusted_endpoint(helper, trusted_endpoint)

        # Prompt to choose model
        model_type = self.get_model_details(
            "Google Cloud Platform", self.OPEN_SOURCE, model_name
        )

        # Prompt for project ID, model ID, and access token
        project_id = project_id or input("Enter your GCP project ID: ").strip()
        model_id = model_id or input("Enter your GCP model ID: ").strip()
        access_token = access_token or input("Enter your GCP access token: ").strip()

        # Get connector body
        connector_body = connector_body or self._get_connector_body(
            model_type, project_id, model_id, access_token
        )

        # Create connector
        print("\nCreating GCP connector...")
        connector_id = helper.create_connector(
            create_connector_role_name=None,
            body=connector_body,
        )

        if connector_id:
            print(
                f"{Fore.GREEN}\nSuccessfully created GCP connector with ID: {connector_id}{Style.RESET_ALL}"
            )
            connector_output = helper.get_connector(connector_id)
            save_config_method(connector_id, connector_output)
            return True
        else:
            print(f"{Fore.RED}Failed to create GCP connector.{Style.RESET_ALL}")
            return False
