# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from typing import Any, Callable, Dict, Optional

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ai_connector_helper import AIConnectorHelper
from opensearch_py_ml.ml_commons.cli.ml_models.model_base import ModelBase


class GCPModel(ModelBase):

    def _get_connector_body(
        self, model_type: str, project_id: str, model_id: str, access_token: str
    ) -> Dict[str, Any]:
        """
        Get the connectory body.
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
        helper: AIConnectorHelper,
        save_config_method: Callable[[str, Dict[str, Any]], None],
        model_name: Optional[str] = None,
        project_id: Optional[str] = None,
        model_id: Optional[str] = None,
        access_token: Optional[str] = None,
        connector_body: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Create Google Cloud Platform connector.

        Args:
            helper: Helper instance for OpenSearch connector operations.
            save_config_method: Method to save connector configuration after creation.
            model_name (optional): Specific GCP model name.
            project_id (optional): GCP project ID.
            model_id (optional): GCP model ID.
            access_token (optional): GCP access token.
            connector_body (optional): The connector request body.

        Returns:
            bool: True if connector creation successful, False otherwise.
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
