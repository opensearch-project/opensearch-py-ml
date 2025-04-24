# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import urlparse

import yaml
from colorama import Fore, Style
from rich.console import Console

from opensearch_py_ml.ml_commons.cli.ai_connector_helper import AIConnectorHelper
from opensearch_py_ml.ml_commons.cli.aws_config import AWSConfig
from opensearch_py_ml.ml_commons.cli.opensearch_domain_config import (
    OpenSearchDomainConfig,
)

# Import readline to enable arrow key navigation, import it this way to avoid lint error
__import__("readline")

# Initialize Rich console for enhanced CLI outputs
console = Console()

# Configure the logger for this module
logger = logging.getLogger(__name__)


class CLIBase:

    # Default setup configuration and output file name
    CONFIG_FILE = os.path.join(os.getcwd(), "setup_config.yml")
    OUTPUT_FILE = os.path.join(os.getcwd(), "output.yml")

    OPEN_SOURCE = AIConnectorHelper.OPEN_SOURCE
    AMAZON_OPENSEARCH_SERVICE = AIConnectorHelper.AMAZON_OPENSEARCH_SERVICE

    def __init__(self):
        """
        Initialize the CLIBase class.
        """
        self.config = {}
        self.output_config = {
            "connector_create": [],
            "register_model": [],
            "predict_model": [],
        }

    def load_config(
        self, config_path: str = None, config_type: str = "setup"
    ) -> Dict[str, Any]:
        """
        Load configuration from the specified file path.

        Args:
            config_path (optional): Path to the configuration file.
            config_type (optional): Type of configuration being loaded. Defaults to "setup".

        Returns:
            Dict[str, Any]: The loaded configuration as a dictionary. Returns an empty dictionary if:
                - The file doesn't exist
                - There are permission issues
                - The YAML is invalid
                - The file is empty
                - Any other error occurs
        """
        try:
            # Handle config path for setup configuration
            if config_type == "setup":
                config_path = config_path or self.CONFIG_FILE

            # Normalize the path
            config_path = os.path.abspath(os.path.expanduser(config_path))

            # Check if file exists
            if not os.path.exists(config_path):
                logger.warning(
                    f"{Fore.YELLOW}Configuration file not found at {config_path}{Style.RESET_ALL}"
                )
                return {}

            with open(config_path, "r") as file:
                config = yaml.safe_load(file) or {}

                # Update stored config for setup configuration
                if config_type == "setup":
                    self.CONFIG_FILE = config_path
                    self.config = config

                print(
                    f"{Fore.GREEN}\n{config_type.capitalize()} configuration loaded successfully from {config_path}{Style.RESET_ALL}"
                )
                return config
        except yaml.YAMLError as ye:
            logger.error(
                f"{Fore.RED}Error parsing YAML configuration: {str(ye)}{Style.RESET_ALL}"
            )
        except PermissionError:
            logger.error(
                f"{Fore.RED}Permission denied: Unable to read {config_path}{Style.RESET_ALL}"
            )
        except Exception as e:
            logger.error(
                f"{Fore.RED}Error loading {config_type} configuration: {str(e)}{Style.RESET_ALL}"
            )
        return {}

    def save_yaml_file(
        self,
        config: Dict[str, Any],
        file_type: str = "configuration",
        merge_existing: bool = False,
    ) -> Optional[str]:
        """
        Save data to a YAML file with optional merging of existing content.

        Args:
            config: The configuration dictionary to save.
            file_type (optional): Type of file being saved. Defaults to "configuration". Common values:
                - "configuration": Uses CONFIG_FILE as default path
                - "output": Uses OUTPUT_FILE as default path
            merge_existing (optional): Whether to merge with existing file content
                if the file exists. Defaults to False. If False and file exists, will
                prompt for overwrite confirmation.

        Returns:
            Optional[str]: The absolute path where the file was saved, or None if:
                - User cancelled the operation
                - Permission error occurred
                - Any other error during saving
                - User declined to overwrite existing file
        """
        try:
            # Determine default path and prompt message based on file type
            default_path = (
                self.CONFIG_FILE if file_type == "configuration" else self.OUTPUT_FILE
            )

            # Check write permissions before attempting any operations
            directory = os.path.dirname(default_path) or os.getcwd()
            if not os.access(directory, os.W_OK):
                raise PermissionError(f"No write access to directory: {directory}")

            # Get save path from user
            path = (
                input(
                    f"\nEnter the path to save the {file_type} information, "
                    f"or press Enter to save it in the current directory [{default_path}]: "
                ).strip()
                or default_path
            )

            # Validate and normalize the path
            path = os.path.abspath(os.path.expanduser(path))
            if not path.endswith((".yaml", ".yml")):
                path = f"{path}.yaml"

            if merge_existing and os.path.exists(path):
                # Read and merge with existing content
                config = self._merge_configs(path, config)
            elif os.path.exists(path):
                # Ask for confirmation to overwrite
                if not self._confirm_overwrite(path):
                    return None

            # Create directory if it doesn't exist
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            # Save file
            with open(path, "w") as file:
                yaml.dump(config, file, default_flow_style=False, sort_keys=False)

            # Update relevant path attribute
            if file_type == "configuration":
                self.CONFIG_FILE = path
            else:
                self.OUTPUT_FILE = path

            print(
                f"{Fore.GREEN}\n{file_type.capitalize()} information saved successfully to {path}{Style.RESET_ALL}"
            )
            return path

        except PermissionError as pe:
            logger.error(f"{Fore.RED}Permission denied: {str(pe)}{Style.RESET_ALL}")
        except KeyboardInterrupt:
            logger.error(
                f"\n{Fore.YELLOW}Operation cancelled by user.{Style.RESET_ALL}"
            )
        except Exception as e:
            logger.error(
                f"{Fore.RED}Error saving {file_type}: {str(e)}{Style.RESET_ALL}"
            )
        return None

    def _merge_configs(self, path: str, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Read and merge the config content.
        """
        # Read existing content
        with open(path, "r") as file:
            existing_config = yaml.safe_load(file) or {}

        # Merge new config with existing config
        for key, value in new_config.items():
            if key not in existing_config:
                existing_config[key] = value
            else:
                existing_config[key].extend(value)
        return existing_config

    def _confirm_overwrite(self, path: str) -> bool:
        """
        Ask for confirmation to overwrite existing file.
        """
        while True:
            response = (
                input(
                    f"{Fore.YELLOW}File already exists at {path}. "
                    f"Do you want to overwrite it? (yes/no): {Style.RESET_ALL}"
                )
                .strip()
                .lower()
            )
            if response in ["yes", "no"]:
                break
            logger.warning(f"{Fore.YELLOW}Please enter 'yes' or 'no'.{Style.RESET_ALL}")

        if response == "no":
            logger.warning(
                f"{Fore.YELLOW}Operation cancelled. Please choose a different path.{Style.RESET_ALL}"
            )
            return False
        return True

    def update_config(self, config: Dict[str, Any], config_path: str) -> bool:
        """
        Update config file with new configurations.

        Args:
            config:  The configuration dictionary to save.
            config_path: The file path where the configuration should be saved.

        Returns:
            bool: True if the configuration was successfully saved, False if any error
                occurred during the save operation.
        """
        try:
            with open(config_path, "w") as file:
                yaml.dump(config, file, default_flow_style=False, sort_keys=False)
            print(
                f"{Fore.GREEN}Configuration saved successfully to {config_path}.{Style.RESET_ALL}"
            )
            return True
        except Exception as e:
            logger.error(
                f"{Fore.RED}Error saving configuration: {str(e)}{Style.RESET_ALL}"
            )
            return False

    def connector_output(
        self,
        output_id: str,
        output_config: str,
        role_name: Optional[str] = None,
        role_arn: Optional[str] = None,
        secret_name: Optional[str] = None,
        secret_arn: Optional[str] = None,
    ) -> None:
        """
        Save connector output to a YAML file.

        Args:
            output_id: The connector ID created.
            output_config: The connector configuration output.
            role_name (optional): Name of the IAM role associated with the connector. Defaults to None.
            role_arn (optional): IAM role ARN associated with the connector. Defaults to None.
            secret_name (optional): Name of the secret associated with the connector. Defaults to None.
            secret_arn (optional): Secret ARN associated with the connector. Defaults to None.
        """
        connector_data = json.loads(output_config)
        connector_name = connector_data.get("name")
        # Update the connector_create section
        self.output_config["connector_create"].append(
            {
                "connector_id": output_id,
                "connector_name": connector_name,
                "connector_role_arn": role_arn or "",
                "connector_role_name": role_name or "",
                "connector_secret_arn": secret_arn or "",
                "connector_secret_name": secret_name or "",
            }
        )
        self.save_yaml_file(self.output_config, "output", merge_existing=True)

    def register_model_output(
        self, model_id: str, model_name: str, connector_id: str
    ) -> None:
        """
        Save register model output to a YAML file.

        Args:
            model_id: The registered model ID.
            model_name: The registered model name.
            connector_id: The connector ID associated with the registered model.
        """
        # Update the register_model section
        self.output_config["register_model"].append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "connector_id": connector_id,
            }
        )
        self.save_yaml_file(self.output_config, "output", merge_existing=True)

    def predict_model_output(self, model_id: str, response: str) -> None:
        """
        Save predict model output to a YAML file.

        Args:
            model_id: The model ID used in the prediction.
            response: The response from the prediction.
        """
        # Update the predict_model section
        self.output_config["predict_model"].append(
            {
                "model_id": model_id,
                "response": response,
            }
        )
        self.save_yaml_file(self.output_config, "output", merge_existing=True)

    def get_opensearch_domain_name(
        self, opensearch_domain_endpoint: str
    ) -> Optional[str]:
        """
        Extract the domain name from the OpenSearch endpoint URL.

        Args:
            opensearch_domain_endpoint: URL like 'https://search-domain-abc123.region.es.amazonaws.com'

        Returns:
            Optional[str]:
                - str: Domain name without prefixes and unique suffixes (e.g., 'domain')
                - None: If URL is invalid or empty
        """
        if not opensearch_domain_endpoint:
            return None

        try:
            parsed_url = urlparse(opensearch_domain_endpoint)
            if not parsed_url.hostname:
                return None

            hostname = parsed_url.hostname.split(".")[0]

            # Remove prefix if present
            if hostname.startswith("search-"):
                hostname = hostname[len("search-") :]
            elif hostname.startswith("vpc-"):
                hostname = hostname[len("vpc-") :]

            # Remove unique ID suffix if present
            if "-" in hostname:
                parts = hostname.split("-")
                return "-".join(parts[:-1])
            return hostname

        except Exception as e:
            logger.error(
                f"Error parsing OpenSearch domain endpoint: {opensearch_domain_endpoint}. Error: {str(e)}"
            )
            return None

    def _check_config(
        self,
        config: Dict[str, Any],
        service_type: str,
        opensearch_config: Dict[str, Any],
    ) -> Union[AIConnectorHelper, bool]:
        """
        Check if the configuration is valid for the given service type and OpenSearch configuration.
        """
        aws_credentials = self.config.get("aws_credentials", {})
        opensearch_domain_endpoint = opensearch_config.get("opensearch_domain_endpoint")
        self.opensearch_domain_name = None

        if not opensearch_domain_endpoint:
            logger.warning(
                f"\n{Fore.RED}OpenSearch endpoint not set. Please run setup first.{Style.RESET_ALL}\n"
            )
            return False

        if service_type == self.AMAZON_OPENSEARCH_SERVICE:
            # For managed service, check AWS-specific configurations
            self.opensearch_domain_name = self.get_opensearch_domain_name(
                opensearch_domain_endpoint
            )
            if (
                not opensearch_config.get("opensearch_domain_region")
                or not self.opensearch_domain_name
            ):
                logger.warning(
                    f"{Fore.RED}AWS region or domain name not set. Please run setup first.{Style.RESET_ALL}\n"
                )
                return False

        # Create OpenSearch config
        opensearch_config = OpenSearchDomainConfig(
            opensearch_domain_region=opensearch_config.get("opensearch_domain_region"),
            opensearch_domain_name=self.opensearch_domain_name,
            opensearch_domain_username=opensearch_config.get(
                "opensearch_domain_username"
            ),
            opensearch_domain_password=opensearch_config.get(
                "opensearch_domain_password"
            ),
            opensearch_domain_endpoint=opensearch_config.get(
                "opensearch_domain_endpoint"
            ),
        )

        # Create AWS config
        aws_config = AWSConfig(
            aws_user_name=aws_credentials.get("aws_user_name"),
            aws_role_name=aws_credentials.get("aws_role_name"),
            aws_access_key=aws_credentials.get("aws_access_key"),
            aws_secret_access_key=aws_credentials.get("aws_secret_access_key"),
            aws_session_token=aws_credentials.get("aws_session_token"),
        )

        # Create AIConnectorHelper
        ai_helper = AIConnectorHelper(
            service_type=config.get("service_type"),
            opensearch_config=opensearch_config,
            aws_config=aws_config,
            ssl_check_enabled=config.get("ssl_check_enabled"),
        )
        return ai_helper

    def load_and_check_config(
        self, config_path: str
    ) -> Union[Tuple[AIConnectorHelper, Dict[str, Any], str, Dict[str, Any]], bool]:
        """
        Load and check configuration.

        Args:
            config_path: The file path where the configuration should be loaded from and checked.

        Returns:
            Union[Tuple[AIConnectorHelper, Dict[str, Any], str, Dict[str, Any]], bool]:
                If successful, returns a tuple containing:
                - AIConnectorHelper: Initialized helper instance for AI connectors.
                - Dict[str, Any]: Complete configuration dictionary.
                - str: Service type extracted from configuration.
                - Dict[str, Any]: OpenSearch domain configuration.

                If any validation fails, returns False.

        """
        # Load configuration
        config = self.load_config(config_path)
        if not config:
            return False

        service_type = config.get("service_type")
        opensearch_config = self.config.get("opensearch_config", {})

        # Check configuration validity
        ai_helper = self._check_config(config, service_type, opensearch_config)
        if not ai_helper:
            return False
        return ai_helper, config, service_type, opensearch_config
