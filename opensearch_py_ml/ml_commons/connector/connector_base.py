# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import os

import yaml
from colorama import Fore, Style
from rich.console import Console

# Initialize Rich console for enhanced CLI outputs
console = Console()


class ConnectorBase:

    CONFIG_FILE = os.path.join(os.getcwd(), "connector_config.yml")

    def __init__(self):
        self.config = {}

    def load_config(self, config_path=None) -> dict:
        """
        Load configuration from the config file.
        """
        try:
            config_path = config_path or self.CONFIG_FILE

            if not os.path.exists(config_path):
                print(
                    f"{Fore.YELLOW}Configuration file not found at {config_path}{Style.RESET_ALL}"
                )
                return {}

            with open(config_path, "r") as file:
                config = yaml.safe_load(file) or {}
                self.config = config
                return config

        except Exception as e:
            print(f"{Fore.RED}Error loading configuration: {str(e)}{Style.RESET_ALL}")
            return {}

    def save_config(self, config: dict):
        """
        Save configuration to the config file.
        """
        try:
            with open(self.CONFIG_FILE, "w") as file:
                yaml.dump(config, file, default_flow_style=False, sort_keys=False)
            print(
                f"{Fore.GREEN}Configuration saved successfully to {self.CONFIG_FILE}.{Style.RESET_ALL}"
            )
            return True
        except Exception as e:
            print(f"{Fore.RED}Error saving configuration: {str(e)}{Style.RESET_ALL}")
            return False
