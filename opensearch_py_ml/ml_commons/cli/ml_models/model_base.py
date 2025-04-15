# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import uuid

from rich.console import Console

# Initialize Rich console for enhanced CLI outputs
console = Console()
from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.connector_manager import ConnectorManager
from opensearch_py_ml.ml_commons.cli.ml_setup import Setup


class ModelBase:

    OPEN_SOURCE = Setup.OPEN_SOURCE
    AMAZON_OPENSEARCH_SERVICE = Setup.AMAZON_OPENSEARCH_SERVICE

    def set_trusted_endpoint(self, helper, trusted_endpoint):
        """
        Sets the trusted endpoint for the model
        """
        settings_body = {
            "persistent": {
                "plugins.ml_commons.trusted_connector_endpoints_regex": [
                    trusted_endpoint
                ]
            }
        }
        helper.opensearch_client.cluster.put_settings(body=settings_body)

    def create_connector_role(self, connector_role_prefix, model_name):
        """
        Create connector role name
        """
        if not connector_role_prefix:
            connector_role_prefix = input("Enter your connector role prefix: ") or None
            if not connector_role_prefix:
                raise ValueError("Connector role prefix cannot be empty.")

        # Add a unique ID to prevent permission error
        id = str(uuid.uuid1())[:6]
        connector_role_name = f"{connector_role_prefix}-{model_name}-connector-{id}"
        create_connector_role_name = (
            f"{connector_role_prefix}-{model_name}-connector-create-{id}"
        )
        return connector_role_name, create_connector_role_name

    def create_secret_name(self, secret_name, model_name, api_key):
        """
        Create a secret name for the model
        """
        if not secret_name:
            secret_name = input(
                "Enter a name for the AWS Secrets Manager secret: "
            ).strip()

        secret_key = f"{model_name}_api_key"
        secret_value = {secret_key: api_key}
        return secret_name, secret_value

    def set_api_key(self, api_key, model_name):
        """
        Set the API key for the model
        """
        setup = Setup()
        if not api_key:
            api_key = setup.get_password_with_asterisks(
                f"Enter your {model_name} API key: "
            )
        return api_key

    def get_model_details(self, connector_name, service_type, model_name=None):
        """
        Get model details based on connector name
        """
        # Get available models for the specific connector
        self.connector_manager = ConnectorManager()
        available_models = self.connector_manager.get_available_models(
            connector_name, service_type
        )

        if not available_models:
            raise ValueError(f"No models found for connector: {connector_name}")

        if model_name:
            for model in available_models:
                if model.name == model_name:
                    return model.id

        print("\nPlease select a model for the connector creation: ")
        for model in available_models:
            print(f"{model.id}. {model.name}")

        while True:
            choice = input(f"Enter your choice (1-{len(available_models)}): ").strip()
            if any(model.id == choice for model in available_models):
                return choice
            print("Invalid choice. Please enter a valid number.")

    def input_custom_model_details(self, external=False):
        """
        Prompts the user to input custom model details for the connector creation.
        """
        if external:
            print(
                f"{Fore.YELLOW}\nIMPORTANT: When customizing the connector configuration, ensure you include the following in the 'headers' section:"
            )
            print(f'{Fore.YELLOW}{Style.BRIGHT}"Authorization": "${{auth}}"')
            print(
                f"{Fore.YELLOW}This placeholder will be automatically replaced with the secure reference to your API key.\n"
            )
        print("Please enter your model details as a JSON object.")
        print("\nClick the link below for examples of the connector blueprint: ")
        console.print("[bold]Amazon OpenSearch Service:[/bold]")
        print(
            "https://github.com/opensearch-project/ml-commons/tree/2.x/docs/tutorials/aws"
        )
        console.print("\n[bold]Open-Source Service:[/bold]")
        print(
            "https://github.com/opensearch-project/ml-commons/tree/2.x/docs/remote_inference_blueprints"
        )
        print("\nEnter your JSON object (press Enter twice when done): ")
        json_input = ""
        lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)

        json_input = "\n".join(lines)

        try:
            custom_details = json.loads(json_input)
            return custom_details
        except json.JSONDecodeError as e:
            print(f"Invalid JSON input: {e}")
            return None
