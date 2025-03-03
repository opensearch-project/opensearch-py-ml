# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from colorama import Fore, Style, init
from rich.console import Console

from opensearch_py_ml.ml_commons.connector.AIConnectorHelper import AIConnectorHelper
from opensearch_py_ml.ml_commons.connector.connector_base import ConnectorBase

# Initialize Rich console for enhanced CLI outputs
console = Console()

# Initialize colorama for colored terminal output
init(autoreset=True)


class Create(ConnectorBase):
    """
    Handles the creation of connector.
    """

    def __init__(self):
        self.config = {}

    def create_command(self):
        """
        Main create command that orchestrates the entire connector creation process.
        """
        try:
            # Load configuration
            config = self.load_config()
            if not config:
                print(
                    f"{Fore.RED}OpenSearch username or password not set. Please run setup first.{Style.RESET_ALL}\n"
                )
                return False

            # Initialize clients
            if not self.initialize_opensearch_client():
                print(
                    f"{Fore.RED}Failed to initialize OpenSearch client.{Style.RESET_ALL}"
                )
                return False

            # Create AIConnectorHelper instance
            ai_helper = AIConnectorHelper(
                opensearch_domain_region=config.get("opensearch_domain_region"),
                opensearch_domain_name=config.get("opensearch_domain_name"),
                opensearch_domain_username=config.get("opensearch_domain_username"),
                opensearch_domain_password=config.get("opensearch_domain_password"),
                aws_user_name=config.get("aws_user_name"),
                aws_role_name=config.get("aws_role_name"),
                opensearch_domain_url=config.get("opensearch_domain_endpoint"),
            )

            # Get connector blueprint from config
            blueprint = config.get("connector_blueprint")
            if not blueprint:
                print(
                    f"{Fore.RED}No connector blueprint found in configuration.{Style.RESET_ALL}"
                )
                return False

            # Create connector with role or secret based on blueprint
            if self.service_type == "managed":
                # Create connector with role
                connector_id = ai_helper.create_connector_with_role(
                    connector_role_inline_policy=blueprint[
                        "connector_role_inline_policy"
                    ],
                    connector_role_name=blueprint["connector_role_name"],
                    create_connector_role_name=blueprint["create_connector_role_name"],
                    create_connector_input=blueprint["connector_config"],
                )
            else:
                # Create connector with secret
                connector_id = ai_helper.create_connector_with_secret(
                    secret_name=blueprint["secret_name"],
                    secret_value=blueprint["secret_value"],
                    connector_role_name=blueprint["connector_role_name"],
                    create_connector_role_name=blueprint["create_connector_role_name"],
                    create_connector_input=blueprint["connector_config"],
                )

            if connector_id:
                print(
                    f"{Fore.GREEN}Successfully created connector with ID: {connector_id}{Style.RESET_ALL}"
                )
                # Update config with connector ID if needed
                config["connector_id"] = connector_id
                self.save_config(config)
                return True
            else:
                print(f"{Fore.RED}Failed to create connector.{Style.RESET_ALL}")
                return False

        except Exception as e:
            print(f"{Fore.RED}Error creating connector: {str(e)}{Style.RESET_ALL}")
            return False
