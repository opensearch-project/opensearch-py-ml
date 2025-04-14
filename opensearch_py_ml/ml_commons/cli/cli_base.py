# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import logging
import os
from urllib.parse import urlparse

import yaml
from colorama import Fore, Style
from rich.console import Console

from opensearch_py_ml.ml_commons.cli.ai_connector_helper import AIConnectorHelper
from opensearch_py_ml.ml_commons.cli.aws_config import AWSConfig
from opensearch_py_ml.ml_commons.cli.opensearch_domain_config import (
    OpenSearchDomainConfig,
)

# Initialize Rich console for enhanced CLI outputs
console = Console()

# Configure the logger for this module
logger = logging.getLogger(__name__)


class CLIBase:

    # Default setup configuration and output file name
    CONFIG_FILE = os.path.join(os.getcwd(), "setup_config.yml")
    OUTPUT_FILE = os.path.join(os.getcwd(), "output.yml")

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

    def load_config(self, config_path=None, config_type="setup"):
        """
        Load configuration from the specified file path.
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

    def save_yaml_file(self, config, file_type="configuration", merge_existing=False):
        """
        Save data to a YAML file with optional merging of existing content.
        """
        try:
            # Determine default path and prompt message based on file type
            default_path = (
                self.CONFIG_FILE if file_type == "configuration" else self.OUTPUT_FILE
            )

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

        except PermissionError:
            logger.error(
                f"{Fore.RED}Error: Permission denied. Unable to write to {path}{Style.RESET_ALL}"
            )
        except KeyboardInterrupt:
            logger.error(
                f"\n{Fore.YELLOW}Operation cancelled by user.{Style.RESET_ALL}"
            )
        except Exception as e:
            logger.error(
                f"{Fore.RED}Error saving {file_type}: {str(e)}{Style.RESET_ALL}"
            )
        return None

    def _merge_configs(self, path, new_config):
        """Read and merge the config content"""
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
        """Ask for confirmation to overwrite existing file."""
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

    def update_config(self, config: dict, config_path: str):
        """
        Update config file with new configurations.
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
        output_config,
        role_name=None,
        secret_name=None,
        role_arn=None,
    ):
        """
        Save connector output to a YAML file.
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
                "connector_secret_name": secret_name or "",
            }
        )
        self.save_yaml_file(self.output_config, "output", merge_existing=True)

    def register_model_output(self, model_id, model_name):
        """
        Save register model output to a YAML file.
        """
        # Update the register_model section
        self.output_config["register_model"].append(
            {
                "model_id": model_id,
                "model_name": model_name,
            }
        )
        self.save_yaml_file(self.output_config, "output", merge_existing=True)

    def predict_model_output(self, response):
        """
        Save predict model output to a YAML file.
        """
        # Update the predict_model section
        self.output_config["predict_model"].append(
            {
                "response": response,
            }
        )
        self.save_yaml_file(self.output_config, "output", merge_existing=True)

    def get_opensearch_domain_name(self, opensearch_domain_endpoint) -> str:
        """
        Extract the domain name from the OpenSearch endpoint URL.
        """
        if opensearch_domain_endpoint:
            parsed_url = urlparse(opensearch_domain_endpoint)
            hostname = parsed_url.hostname
            if hostname:
                # Split the hostname into parts
                parts = hostname.split(".")
                domain_part = parts[0]
                # Handle both search- and vpc- prefixes
                if domain_part.startswith("search-"):
                    domain_part = domain_part[len("search-") :]
                elif domain_part.startswith("vpc-"):
                    domain_part = domain_part[len("vpc-") :]
                # Remove the unique ID suffix after the domain name
                if "-" in domain_part:
                    # Split by all dashes and reconstruct the domain name without the last part
                    parts = domain_part.split("-")
                    domain_name = "-".join(parts[:-1])
                    return domain_name
                return domain_part
        return None

    def _check_config(self, config, service_type, opensearch_config):
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

        if service_type == "amazon-opensearch-service":
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
            ssl_check_enabled=config.get("ssl_check_enabled"),
            opensearch_config=opensearch_config,
            aws_config=aws_config,
        )
        return ai_helper

    def load_and_check_config(self, config_path):
        """
        Load and check configuration.
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
