# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import configparser
import os
import sys

from colorama import Fore, Style
from rich.console import Console

# Initialize Rich console for enhanced CLI outputs
console = Console()


class ConnectorBase:

    CONFIG_FILE = "connector_config.yml"

    def load_config(self) -> dict:
        """
        Load configuration from the config file.
        """
        config = configparser.ConfigParser()
        if os.path.exists(self.CONFIG_FILE):
            config.read(self.CONFIG_FILE)
            if "DEFAULT" not in config:
                console.print(
                    f"[{Fore.RED}ERROR{Style.RESET_ALL}] 'DEFAULT' section missing in {self.CONFIG_FILE}. Please run the setup command first."
                )
                sys.exit(1)
            return dict(config["DEFAULT"])
        return {}

    def save_config(self, config: dict):
        """
        Save configuration to the config file.
        """
        parser = configparser.ConfigParser()
        parser["DEFAULT"] = config
        try:
            with open(self.CONFIG_FILE, "w") as f:
                parser.write(f)
            console.print(
                f"[{Fore.GREEN}SUCCESS{Style.RESET_ALL}] Configuration saved to {self.CONFIG_FILE}."
            )
        except Exception as e:
            console.print(
                f"[{Fore.RED}ERROR{Style.RESET_ALL}] Failed to save configuration: {e}"
            )
