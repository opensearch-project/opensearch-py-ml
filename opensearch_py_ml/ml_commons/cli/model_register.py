# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from colorama import Fore, Style, init
from rich.console import Console

from opensearch_py_ml.ml_commons.cli.AIConnectorHelper import AIConnectorHelper
from opensearch_py_ml.ml_commons.cli.connector_base import ConnectorBase

# Initialize Rich console for enhanced CLI outputs
console = Console()

# Initialize colorama for colored terminal output
init(autoreset=True)


class Register(ConnectorBase):
    """
    Handles the model registration.
    """

    def __init__(self):
        """
        Initialize the Register class.
        """
        super().__init__()
        self.config = {}
        self.opensearch_domain_name = ""

    def register_command(
        self, config_path, connector_id=None, model_name=None, model_description=None
    ):
        """
        Main register command to orchestrates the entire model registration process.
        """
        try:
            # Load configuration
            config = self.load_config(config_path)
            if not config:
                return False

            opensearch_config = self.config.get("opensearch_config", {})
            aws_credentials = self.config.get("aws_credentials", {})
            opensearch_domain_endpoint = opensearch_config.get(
                "opensearch_domain_endpoint"
            )
            if not opensearch_domain_endpoint:
                print(
                    f"\n{Fore.RED}OpenSearch endpoint not set. Please run setup first.{Style.RESET_ALL}\n"
                )
                return False

            service_type = config.get("service_type")
            if service_type == "open-source":
                # For open-source, check username and password
                if not opensearch_config.get(
                    "opensearch_domain_username"
                ) or not opensearch_config.get("opensearch_domain_password"):
                    print(
                        f"{Fore.RED}OpenSearch username or password not set. Please run setup first.{Style.RESET_ALL}\n"
                    )
                    return False
                else:
                    self.opensearch_domain_name = None
            else:
                # For managed service, check AWS-specific configurations
                self.opensearch_domain_name = self.get_opensearch_domain_name(
                    opensearch_domain_endpoint
                )
                if (
                    not opensearch_config.get("opensearch_domain_region")
                    or not self.opensearch_domain_name
                ):
                    print(
                        f"{Fore.RED}AWS region or domain name not set. Please run setup first.{Style.RESET_ALL}\n"
                    )
                    return False

            # Create AIConnectorHelper instance
            ai_helper = AIConnectorHelper(
                service_type=config.get("service_type"),
                opensearch_domain_region=opensearch_config.get(
                    "opensearch_domain_region"
                ),
                opensearch_domain_name=self.opensearch_domain_name,
                opensearch_domain_username=opensearch_config.get(
                    "opensearch_domain_username"
                ),
                opensearch_domain_password=opensearch_config.get(
                    "opensearch_domain_password"
                ),
                aws_user_name=aws_credentials.get("aws_user_name"),
                aws_role_name=aws_credentials.get("aws_role_name"),
                opensearch_domain_url=opensearch_config.get(
                    "opensearch_domain_endpoint"
                ),
                aws_access_key=aws_credentials.get("aws_access_key"),
                aws_secret_access_key=aws_credentials.get("aws_secret_access_key"),
                aws_session_token=aws_credentials.get("aws_session_token"),
            )

            # Prompt for model name, description, and connector ID if not provided
            if not model_name:
                model_name = input("\nEnter the model name: ").strip()
            if not model_description:
                model_description = input("Enter the model description: ").strip()
            if not connector_id:
                connector_id = input("Enter the connector ID: ").strip()

            model_id = ai_helper.register_model(
                model_name, model_description, connector_id
            )

            if model_id:
                print(
                    f"{Fore.GREEN}\nSuccessfully registered a model with ID: {model_id}{Style.RESET_ALL}"
                )
                self.register_model_output(model_id, model_name)
                return True
            else:
                print(f"{Fore.RED}Failed to register model.{Style.RESET_ALL}")
                return False
        except Exception:
            return False
