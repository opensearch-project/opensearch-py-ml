# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from colorama import Fore, Style, init
from rich.console import Console

from opensearch_py_ml.ml_commons.cli.ai_connector_helper import AIConnectorHelper
from opensearch_py_ml.ml_commons.cli.cli_base import CLIBase
from opensearch_py_ml.ml_commons.cli.connector_list import (
    ConnectorInfo,
    ConnectorList,
    ModelInfo,
)

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

    def get_connectors(self, service_type: str) -> List[ConnectorInfo]:
        """
        Get connectors for specific service type.

        Args:
            service_type: The type of service to get connectors for. Either self.OPEN_SOURCE or self.AMAZON_OPENSEARCH_SERVICE.

        Returns:
            List[Connector]: List of connector objects for the specified service type.

        Raises:
            ValueError: If the service_type is not recognized.
        """
        if service_type == self.OPEN_SOURCE:
            return self.connector_list._opensource_connectors
        elif service_type == self.AMAZON_OPENSEARCH_SERVICE:
            return self.connector_list._managed_connectors
        else:
            raise ValueError(f"Unknown service type: {service_type}")

    def print_available_connectors(self, service_type: str) -> None:
        """
        Print available connectors for specific service type.

        Args:
            service_type: The type of service to list connectors for.
        """
        connectors = self.get_connectors(service_type)
        if not connectors:
            logger.warning(f"\nNo connectors available for {service_type}")
            return

        service_name = "Amazon OpenSearch Service"
        if service_type == self.OPEN_SOURCE:
            service_name = "OpenSearch"
        print(f"\nPlease select a supported connector to create in {service_name}:")
        for connector in connectors:
            print(f"{connector.id}. {connector.name}")
        max_choice = len(connectors)
        print(f"Enter your choice (1-{max_choice}): ", end="")

    def get_connector_by_id(
        self, connector_id: int, service_type: str
    ) -> ConnectorInfo:
        """
        Get connector by ID for specific service type.

        Args:
            connector_id: The numeric identifier of the connector.
            service_type: The type of service the connector belongs to.

        Returns:
            ConnectorInfo: The connector object matching the specified ID.

        Raises:
            ValueError: If no connector is found with the specified ID.
        """
        connectors = self.get_connectors(service_type)
        for connector in connectors:
            if connector.id == connector_id:
                return connector
        raise ValueError

    def get_connector_by_name(self, name: str, service_type: str) -> ConnectorInfo:
        """
        Get connector by name for specific service type.

        Args:
            name: The name of the connector.
            service_type: The type of service the connector belongs to.

        Returns:
            ConnectorInfo: he connector object matching the specified name.

        Raises:
            ValueError: If no connector is found with the specified name.
        """
        connectors = self.get_connectors(service_type)
        for connector in connectors:
            if connector.name.lower() == name.lower():
                return connector
        raise ValueError

    def get_available_models(
        self, connector_name: str, service_type: str
    ) -> List[ModelInfo]:
        """
        Get available models for a specific connector.

        Args:
            connector-name: Name of the connector to get models for.
            service_type: The type of service the connector belongs to.

        Returns:
            List[ModelInfo]: List of ModelInfo objects containing available models.
                Each ModelInfo object has:
                - id (str): Unique identifier for the model.
                - name (str): Display name of the model.
                Returns empty list if connector is not found.
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

    def get_connector_class(self, connector_name: str) -> Optional[Type[Any]]:
        """
        Import and return the connector class.

        Args:
            connector_name: Name of the connector to get the class for.

        Returns:
            Optional[Type[Any]]: The connector class if found and successfully imported,
            None if:
                - Connector information not found
                - Module import fails
                - Class not found in module
        """
        connector = self.get_connector_info(connector_name)
        if connector:
            module_path = (
                f"opensearch_py_ml.ml_commons.cli.ml_models.{connector.file_name}"
            )
            module = __import__(module_path, fromlist=[connector.connector_class])
            return getattr(module, connector.connector_class)
        return None

    def get_connector_info(self, connector_name: str) -> Optional[ConnectorInfo]:
        """
        Get connector information by name.

        Args:
            connector_name: Name of the connector to find information for.

        Returns:
            Optional[ConnectorInfo]: ConnectorInfo object if found.
            None if no matching connector is found.
        """
        for connector in (
            self.connector_list._managed_connectors
            + self.connector_list._opensource_connectors
        ):
            if connector.name == connector_name:
                return connector
        return None

    def _get_connector_config_params(
        self, connector_config: Dict[str, Any]
    ) -> Dict[str, Optional[Any]]:
        """
        Get connector configuration parameters.
        """
        param_keys = [
            "access_token",
            "api_key",
            "aws_access_key",
            "aws_secret_access_key",
            "aws_session_token",
            "connector_body",
            "connector_name",
            "connector_role_inline_policy",
            "connector_role_prefix",
            "connector_secret_name",
            "endpoint_arn",
            "endpoint_url",
            "model_id",
            "model_name",
            "project_id",
            "region",
            "required_policy",
            "required_secret",
        ]

        params = {}
        for key in param_keys:
            params[key] = connector_config.get(key)
        return params

    def create_model_instance(
        self,
        connector_info: ConnectorInfo,
        connector_class: Type[Any],
        opensearch_config: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Any:
        """
        Create model instance.

        Args:
            connector_info: Connector information object.
            connector_class: The class to instantiate for the model.
            opensearch_config: OpenSearch domain configuration.
            config: General configuration parameters.

        Returns:
            Any: An instance of the connector_class initialized with the required parameters.
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
        connector_config_path: Optional[str],
        connector_config_params: Optional[Dict[str, Any]],
        connector_info: ConnectorInfo,
        opensearch_config: Dict[str, Any],
        config: Dict[str, Any],
        model: Any,
        ai_helper: AIConnectorHelper,
    ) -> Any:
        """
        Create connector instance.

        Args:
            connector_config_path: Path to connector configuration file.
            connector_config_params: Parameters from connector configuration.
            connector_info: Connector information object.
            opensearch_config: OpenSearch domain configuration.
            config: General configuration parameters.
            model: Model instance to use for connector creation.
            ai_helper: Helper instance for AI connector operations.

        Returns:
            Any: Created connector instance.
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

    def initialize_create_connector(
        self, connector_config_path: Optional[str] = None
    ) -> Union[Tuple[bool, str], bool]:
        """
        Orchestrates the entire connector creation process.

        Args:
            connector_config_path: Path to connector configuration file. If None, will prompt for setup configuration path.

        Returns:
            Union[Tuple[bool, str], bool]:
                - If successful: Tuple(True, setup_config_path)
                - If failed: False
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

            # Retrieve the connector information if the connector config path is given in the command
            if connector_config_path:
                params = self._get_connector_config_params(connector_config)
                connector_name = params.pop("connector_name")

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

            # Create connector instance
            result = self.create_connector_instance(
                connector_config_path,
                params if connector_config_path else None,
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
