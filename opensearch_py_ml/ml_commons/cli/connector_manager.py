# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import logging

from colorama import Fore, Style, init
from rich.console import Console

from opensearch_py_ml.ml_commons.cli.cli_base import CLIBase
from opensearch_py_ml.ml_commons.cli.connector_list import ConnectorList

# Initialize Rich console for enhanced CLI outputs
console = Console()

# Initialize colorama for colored terminal output
init(autoreset=True)

# Configure the logger for this module
logger = logging.getLogger(__name__)


class ConnectorManager(CLIBase):
    """
    Handles the connector operation.
    """

    def __init__(self):
        """
        Initialize the ConnectorManager class.
        """
        super().__init__()
        self.config = {}
        self.connector_list = ConnectorList()

    def get_connectors(self, service_type: str):
        """
        Get connectors for specific service type
        """
        if service_type == self.OPEN_SOURCE:
            return self.connector_list._opensource_connectors
        elif service_type == self.AMAZON_OPENSEARCH_SERVICE:
            return self.connector_list._managed_connectors
        else:
            raise ValueError(f"Unknown service type: {service_type}")

    def print_available_connectors(self, service_type: str):
        """
        Print available connectors for specific service type
        """
        connectors = self.get_connectors(service_type)
        if not connectors:
            logger.warning(f"\nNo connectors available for {service_type}")
            return

        print("\nPlease select a supported connector to create:")
        for connector in connectors:
            print(f"{connector.id}. {connector.name}")
        max_choice = len(connectors)
        print(f"Enter your choice (1-{max_choice}): ", end="")

    def get_connector_by_id(self, connector_id: int, service_type: str):
        """
        Get connector by ID for specific service type
        """
        connectors = self.get_connectors(service_type)
        for connector in connectors:
            if connector.id == connector_id:
                return connector
        raise ValueError

    def get_connector_by_name(self, name: str, service_type: str):
        """
        Get connector by name for specific service type
        """
        connectors = self.get_connectors(service_type)
        for connector in connectors:
            if connector.name.lower() == name.lower():
                return connector
        raise ValueError

    def get_available_models(self, connector_name: str, service_type: str):
        """
        Get available models for a specific connector
        """
        connectors = (
            self.connector_list._managed_connectors
            if service_type == self.AMAZON_OPENSEARCH_SERVICE
            else self.connector_list._opensource_connectors
        )
        for connector in connectors:
            if connector.name == connector_name:
                return connector.available_models
        return []

    def get_connector_class(self, connector_name: str):
        """
        Import and return the connector class
        """
        connector = self.get_connector_info(connector_name)
        if connector:
            module_path = (
                f"opensearch_py_ml.ml_commons.cli.ml_models.{connector.file_name}"
            )
            module = __import__(module_path, fromlist=[connector.connector_class])
            return getattr(module, connector.connector_class)
        return None

    def get_connector_info(self, connector_name: str):
        """
        Get connector information by name
        """
        for connector in (
            self.connector_list._managed_connectors
            + self.connector_list._opensource_connectors
        ):
            if connector.name == connector_name:
                return connector
        return None

    def create_model_instance(
        self, connector_info, connector_class, opensearch_config, config
    ):
        """
        Create model instance
        """
        init_kwargs = {}
        for param in connector_info.init_params:
            # Check opensearch_config
            if param in opensearch_config and opensearch_config.get(param) is not None:
                init_kwargs[param] = opensearch_config.get(param)
            # Check config
            elif param in config and config.get(param) is not None:
                init_kwargs[param] = config.get(param)

        return connector_class(**init_kwargs)

    def create_connector_instance(
        self,
        connector_config_path,
        connector_config_params,
        connector_info,
        opensearch_config,
        config,
        model,
        ai_helper,
    ):
        """
        Create connector instance
        """
        connector_kwargs = {}
        for param in connector_info.connector_params:
            # Check connector_params from connector config
            if (
                connector_config_path
                and param in connector_config_params
                and connector_config_params[param] is not None
            ):
                connector_kwargs[param] = connector_config_params[param]
            # Check opensearch_config
            elif (
                param in opensearch_config and opensearch_config.get(param) is not None
            ):
                connector_kwargs[param] = opensearch_config.get(param)
            # Check config
            elif param in config and config.get(param) is not None:
                connector_kwargs[param] = config.get(param)

        return model.create_connector(
            ai_helper, self.connector_output, **connector_kwargs
        )

    def initialize_create_connector(self, connector_config_path=None):
        """
        Orchestrates the entire connector creation process.
        """
        try:
            # Check if connector config file path is given in the command
            if connector_config_path:
                connector_config = self.load_config(connector_config_path, "connector")
                setup_config_path = connector_config.get("setup_config_path")
                if not connector_config:
                    logger.error(
                        f"{Fore.RED}No connector configuration found.{Style.RESET_ALL}\n"
                    )
                    return False
            else:
                setup_config_path = input(
                    "\nEnter the path to your existing setup configuration file: "
                ).strip()

            # Load and check configuration
            config_result = self.load_and_check_config(setup_config_path)
            if not config_result:
                return False
            ai_helper, config, service_type, opensearch_config = config_result

            # Set the initial value of the connector creation result to False
            result = False

            # Initialize variables with None
            connector_name = model_name = connector_role_prefix = None
            region = model_arn = connector_body = None
            api_key = secret_name = endpoint_arn = endpoint_url = None

            # Retrieve the connector information if the connector config path is given in the command
            if connector_config_path:
                connector_name = connector_config.get("connector_name")
                model_name = connector_config.get("model_name")
                connector_role_prefix = connector_config.get("connector_role_prefix")
                region = connector_config.get("region")
                model_arn = connector_config.get("model_arn")
                connector_body = connector_config.get("connector_body")
                api_key = connector_config.get("api_key")
                secret_name = connector_config.get("connector_secret_name")
                endpoint_arn = connector_config.get("inference_endpoint_arn")
                endpoint_url = connector_config.get("inference_endpoint_url")

                try:
                    connector_info = self.get_connector_by_name(
                        connector_name, service_type
                    )
                except ValueError:
                    logger.error(
                        f"{Fore.YELLOW}Invalid connector choice. Operation cancelled.{Style.RESET_ALL}"
                    )
                    return False
            else:
                self.print_available_connectors(service_type)
                try:
                    choice = int(input().strip())
                    connector_info = self.get_connector_by_id(choice, service_type)
                except ValueError:
                    logger.error(
                        f"{Fore.YELLOW}Invalid connector choice. Operation cancelled.{Style.RESET_ALL}"
                    )
                    return False

            connector_name = connector_info.name
            connector_class = self.get_connector_class(connector_name)

            # Create model instance
            self.model = self.create_model_instance(
                connector_info, connector_class, opensearch_config, config
            )

            # Create a mapping of connector parameters from config
            connector_config_params = {
                "model_name": model_name,
                "connector_role_prefix": connector_role_prefix,
                "region": region,
                "model_arn": model_arn,
                "connector_body": connector_body,
                "api_key": api_key,
                "secret_name": secret_name,
                "endpoint_arn": endpoint_arn,
                "endpoint_url": endpoint_url,
            }

            # Create connector instance
            result = self.create_connector_instance(
                connector_config_path,
                connector_config_params,
                connector_info,
                opensearch_config,
                config,
                self.model,
                ai_helper,
            )

            return result, setup_config_path
        except Exception as e:
            logger.error(
                f"{Fore.RED}Error creating connector: {str(e)}{Style.RESET_ALL}"
            )
            return False
