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
from typing import Any, Dict, Optional, Tuple

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ai_connector_helper import AIConnectorHelper
from opensearch_py_ml.ml_commons.cli.connector_manager import ConnectorManager
from opensearch_py_ml.ml_commons.cli.ml_setup import Setup


class ModelBase:

    OPEN_SOURCE = Setup.OPEN_SOURCE
    AMAZON_OPENSEARCH_SERVICE = Setup.AMAZON_OPENSEARCH_SERVICE

    def set_trusted_endpoint(
        self, helper: AIConnectorHelper, trusted_endpoint: str
    ) -> None:
        """
        Sets the trusted endpoint for the model.

        Args:
            helper: Helper instance with OpenSearch client.
            trusted_endpoint: Regex pattern for trusted endpoints.
        """
        settings_body = {
            "persistent": {
                "plugins.ml_commons.trusted_connector_endpoints_regex": [
                    trusted_endpoint
                ]
            }
        }
        helper.opensearch_client.cluster.put_settings(body=settings_body)

    def create_connector_role(
        self, connector_role_prefix: Optional[str], model_name: str
    ) -> Tuple[str, str]:
        """
        Create connector role name.

        Args:
            connector_role_prefix: Prefix for role names. If None, will prompt interactively.
            model_name: Name of the model.

        Returns:
            Tuple[str, str]: Tuple containing:
                - connector_role_name: Role for general connector operations
                - create_connector_role_name: Role for connector creation

        Raises:
            ValueError: If connector_role_prefix is empty after prompt
        """
        if not connector_role_prefix:
            connector_role_prefix = input("Enter your connector role prefix: ") or None
            if not connector_role_prefix:
                raise ValueError("Connector role prefix cannot be empty.")

        # Add a unique ID to prevent permission error
        id = str(uuid.uuid1())[:6]
        if model_name:
            connector_role_name = f"{connector_role_prefix}-{model_name}-connector-{id}"
            create_connector_role_name = (
                f"{connector_role_prefix}-{model_name}-connector-create-{id}"
            )
        else:
            connector_role_name = f"{connector_role_prefix}-connector-{id}"
            create_connector_role_name = (
                f"{connector_role_prefix}-connector-create-{id}"
            )
        return connector_role_name, create_connector_role_name

    def create_secret_name(
        self, secret_name: Optional[str], model_name: str, api_key: str
    ) -> Tuple[str, Dict[str, str]]:
        """
        Create a secret name for the model.

        Args:
            secret_name: Name for the secret. If None, will prompt interactively.
            model_name: Name of the model for secret key.
            api_key: API key to store in the secret.

        Returns:
            Tuple[str, Dict[str, str]]: Tuple containing:
                - secret_name: Name for the AWS Secrets Manager secret
                - secret_value: Dictionary with the API key structure
        """
        if not secret_name:
            secret_name = input(
                "Enter a name for the AWS Secrets Manager secret: "
            ).strip()

        id = str(uuid.uuid1())[:6]
        secret_name = f"{secret_name}-{id}"
        if model_name:
            secret_key = f"{model_name}_api_key"
        else:
            secret_key = "api_key"
        secret_value = {secret_key: api_key}
        return secret_name, secret_value

    def set_api_key(self, api_key: Optional[str], model_name: str) -> str:
        """
        Set the API key for the model.

        Args:
            api_key: Existing API key. If None, will prompt interactively with masked input.
            model_name: Name of the model for prompt formatting.

        Returns:
            str: The API key.
        """
        setup = Setup()
        if not api_key:
            api_key = setup.get_password_with_asterisks(
                f"Enter your {model_name} API key: "
            )
        return api_key

    def get_model_details(
        self, connector_name: str, service_type: str, model_name: Optional[str] = None
    ) -> str:
        """
        Get model details based on connector name.

        Args:
            connector_name: Name of the connector to get models for.
            service_type: Type of service (e.g., "amazon_opensearch_service")
            model_name (optional): Specific model name to look up. If None, enables interactive selection.

        Returns:
            str: Model ID of the selected or specified model

        Raises:
            ValueError: If no models are found for the specified connector
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

    def get_aws_credentials(
        self,
        connector_body: Dict[str, Any],
        aws_access_key: Optional[str],
        aws_secret_access_key: Optional[str],
        aws_session_token: Optional[str],
    ) -> None:
        """
        Get AWS credentials from user input.

        Args:
            connector_body: Connector configuration to update
            aws_access_key: AWS access key ID. If None, prompts for input.
            aws_secret_access_key: AWS secret access key. If None, prompts for input.
            aws_session_token: AWS session token. If None, prompts for input.
        """
        setup = Setup()
        connector_body["credential"] = {
            "access_key": aws_access_key
            or setup.get_password_with_asterisks("Enter your AWS Access Key ID: "),
            "secret_key": aws_secret_access_key
            or setup.get_password_with_asterisks("Enter your AWS Secret Access Key: "),
            "session_token": aws_session_token
            or setup.get_password_with_asterisks("Enter your AWS Session Token: "),
        }

    def input_custom_model_details(
        self, external: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Prompts the user to input custom model details for the connector creation.

        Args:
            external (optional): Flag indicating if this is an external model configuration.
                If True, displays additional authentication guidance for API key handling.
                Defaults to False.

        Returns:
            Optional[Dict[str, Any]]:
                - Dict[str, Any]: Successfully parsed JSON configuration
                - None: If JSON parsing fails or input is invalid
        """
        if external:
            print(
                f"{Fore.YELLOW}\nIMPORTANT: When customizing the connector configuration that requires API key authentication, ensure you include the following in the 'headers' section:"
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
