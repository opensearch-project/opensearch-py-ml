# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import os
from urllib.parse import urlparse

import yaml
from colorama import Fore, Style
from rich.console import Console

# Initialize Rich console for enhanced CLI outputs
console = Console()


class ConnectorBase:

    CONFIG_FILE = os.path.join(os.getcwd(), "setup_config.yml")
    OUTPUT_FILE = os.path.join(os.getcwd(), "output.yml")

    def __init__(self):
        self.config = {}
        self.output_config = {
            "connector_create": {},
            "register_model": {},
            "predict_model": {},
        }

    def load_config(self, config_path=None) -> dict:
        """
        Load configuration from the config file path.
        """
        try:
            config_path = config_path or self.CONFIG_FILE

            # Normalize the path
            config_path = os.path.abspath(os.path.expanduser(config_path))

            if not os.path.exists(config_path):
                print(
                    f"{Fore.YELLOW}Configuration file not found at {config_path}{Style.RESET_ALL}"
                )
                return {}

            # Check if file is readable
            if not os.access(config_path, os.R_OK):
                print(
                    f"{Fore.RED}Error: No permission to read configuration file at {config_path}{Style.RESET_ALL}"
                )
                return {}

            with open(config_path, "r") as file:
                config = yaml.safe_load(file) or {}

                # Update the stored config path and config
                self.CONFIG_FILE = config_path
                self.config = config

                print(
                    f"{Fore.GREEN}\nSetup configuration loaded successfully from {config_path}{Style.RESET_ALL}"
                )
                return config

        except yaml.YAMLError as ye:
            print(
                f"{Fore.RED}Error parsing YAML configuration: {str(ye)}{Style.RESET_ALL}"
            )
            return {}
        except PermissionError:
            print(
                f"{Fore.RED}Permission denied: Unable to read {config_path}{Style.RESET_ALL}"
            )
            return {}
        except Exception as e:
            print(
                f"{Fore.RED}Error loading setup configuration: {str(e)}{Style.RESET_ALL}"
            )
            return {}

    def load_connector_config(self, connector_config_path=None) -> dict:
        """
        Load configuration from the config file path.
        """
        try:
            if not os.path.exists(connector_config_path):
                print(
                    f"{Fore.YELLOW}Configuration file not found at {connector_config_path}{Style.RESET_ALL}"
                )
                return {}
            # Check if file is readable
            if not os.access(connector_config_path, os.R_OK):
                print(
                    f"{Fore.RED}Error: No permission to read configuration file at {connector_config_path}{Style.RESET_ALL}"
                )
                return {}

            with open(connector_config_path, "r") as file:
                config = yaml.safe_load(file) or {}

                print(
                    f"{Fore.GREEN}\nConnector configuration loaded successfully from {connector_config_path}{Style.RESET_ALL}"
                )
                return config

        except yaml.YAMLError as ye:
            print(
                f"{Fore.RED}Error parsing YAML configuration: {str(ye)}{Style.RESET_ALL}"
            )
            return {}
        except PermissionError:
            print(
                f"{Fore.RED}Permission denied: Unable to read {connector_config_path}{Style.RESET_ALL}"
            )
            return {}
        except Exception as e:
            print(
                f"{Fore.RED}Error loading connector configuration: {str(e)}{Style.RESET_ALL}"
            )
            return {}

    def save_config(self, config: dict):
        """
        Save configuration to the config file.
        """
        try:
            default_path = self.CONFIG_FILE
            path = (
                input(
                    f"\nEnter the path to save your configuration file, "
                    f"or press Enter to save it in the current directory [{default_path}]: "
                ).strip()
                or default_path
            )

            # Validate and normalize the path
            path = os.path.abspath(os.path.expanduser(path))

            # Validate file extension
            if not path.endswith((".yaml", ".yml")):
                path = f"{path}.yaml"

            # Check if file exists and ask for confirmation to overwrite
            if os.path.exists(path):
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
                    print(f"{Fore.YELLOW}Please enter 'yes or 'no'.{Style.RESET_ALL}")

                if response == "no":
                    print(
                        f"{Fore.YELLOW}Operation cancelled. Please choose a different path.{Style.RESET_ALL}"
                    )
                    return None

            # Create directory if it doesn't exist
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            # Save the configuration
            with open(path, "w") as file:
                yaml.dump(config, file, default_flow_style=False, sort_keys=False)

            # Update the config_file path
            # self.config_file = path
            self.CONFIG_FILE = path

            print(
                f"{Fore.GREEN}Configuration saved successfully to {path}{Style.RESET_ALL}"
            )
            return path
        except PermissionError:
            print(
                f"{Fore.RED}Error: Permission denied. Unable to write to {path}{Style.RESET_ALL}"
            )
            return None
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Operation cancelled by user.{Style.RESET_ALL}")
            return None
        except Exception as e:
            print(f"{Fore.RED}Error saving configuration: {str(e)}{Style.RESET_ALL}")
            return None

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
            print(f"{Fore.RED}Error saving configuration: {str(e)}{Style.RESET_ALL}")
            return False

    def connector_output(
        self,
        output_id: str,
        output_config,
        role_name=None,
        secret_name=None,
        role_arn=None,
    ):
        connector_data = json.loads(output_config)
        connector_name = connector_data.get("name")
        # Update the connector_create section
        self.output_config["connector_create"].update(
            {
                "connector_id": output_id,
                "connector_name": connector_name,
                "connector_role_arn": role_arn or "",
                "connector_role_name": role_name or "",
                "connector_secret_name": secret_name or "",
            }
        )
        self.save_output(self.output_config)

    def register_model_output(self, model_id, model_name):
        # Update the register_model section
        self.output_config["register_model"].update(
            {
                "model_id": model_id,
                "model_name": model_name,
            }
        )
        self.save_output(self.output_config)

    def predict_model_output(self, response):
        self.output_config["predict_model"].update(
            {
                "response": response,
            }
        )
        self.save_output(self.output_config)

    def save_output(self, config: dict):
        """
        Save output to a yaml file.
        """
        try:
            default_path = self.OUTPUT_FILE
            path = (
                input(
                    f"\nEnter the path to save the output information, "
                    f"or press Enter to save it in the current directory [{default_path}]: "
                ).strip()
                or default_path
            )

            # Validate and normalize the path
            path = os.path.abspath(os.path.expanduser(path))

            # Validate file extension
            if not path.endswith((".yaml", ".yml")):
                path = f"{path}.yaml"

            # Read existing content if file exists
            existing_config = {}
            if os.path.exists(path):
                with open(path, "r") as file:
                    existing_config = yaml.safe_load(file) or {}

            # Merge new config with existing config
            for key, value in config.items():
                if key in existing_config and isinstance(existing_config[key], dict):
                    existing_config[key].update(value)
                else:
                    existing_config[key] = value

            # Create directory if it doesn't exist
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            # Save the configuration
            with open(path, "w") as file:
                yaml.dump(
                    existing_config, file, default_flow_style=False, sort_keys=False
                )

            # Update the config_file path
            self.OUTPUT_FILE = path

            print(
                f"{Fore.GREEN}\nOutput information saved successfully to {path}{Style.RESET_ALL}"
            )
            return path
        except PermissionError:
            print(
                f"{Fore.RED}Error: Permission denied. Unable to write to {path}{Style.RESET_ALL}"
            )
            return None
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Operation cancelled by user.{Style.RESET_ALL}")
            return None
        except Exception as e:
            print(
                f"{Fore.RED}Error saving output information: {str(e)}{Style.RESET_ALL}"
            )
            return None

    def get_opensearch_domain_name(self, opensearch_domain_endpoint) -> str:
        """
        Extract the domain name from the OpenSearch endpoint URL.
        """
        if opensearch_domain_endpoint:
            parsed_url = urlparse(opensearch_domain_endpoint)
            hostname = (
                parsed_url.hostname
            )  # e.g., 'search-your-domain-name-uniqueid.region.es.amazonaws.com'
            if hostname:
                # Split the hostname into parts
                parts = hostname.split(".")
                domain_part = parts[0]  # e.g., 'search-your-domain-name-uniqueid'
                # Remove the 'search-' prefix if present
                if domain_part.startswith("search-"):
                    domain_part = domain_part[len("search-") :]
                # Remove the unique ID suffix after the domain name
                domain_name = domain_part.rsplit("-", 1)[0]
                return domain_name
        return None
