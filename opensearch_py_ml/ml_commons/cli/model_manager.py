# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import logging

from colorama import Fore, Style, init
from rich.console import Console

from opensearch_py_ml.ml_commons.cli.cli_base import CLIBase

# Initialize Rich console for enhanced CLI outputs
console = Console()

# Initialize colorama for colored terminal output
init(autoreset=True)

# Configure the logger for this module
logger = logging.getLogger(__name__)


class ModelManager(CLIBase):
    """
    Handles the model operations.
    """

    def __init__(self):
        """
        Initialize the ModelManager class.
        """
        super().__init__()
        self.config = {}

    def initialize_predict_model(self, config_path, model_id=None, body=None):
        """
        Orchestrates the entire model prediction process.
        """
        try:
            # Load and check configuration
            config_result = self.load_and_check_config(config_path)
            if not config_result:
                return False
            ai_helper, _, _, _ = config_result

            # Prompt for model id and predict request body if not provided
            if not model_id:
                model_id = input("\nEnter the model ID: ").strip()
            if not body:
                print(
                    "\nEnter your predict request body as a JSON object (press Enter twice when done): "
                )
                json_input = ""
                lines = []
                while True:
                    line = input()
                    if line.strip() == "":
                        break
                    lines.append(line)

                json_input = "\n".join(lines)
                body = json.loads(json_input)
            else:
                body = json.loads(body)

            response, status = ai_helper.predict(model_id, body)
            if status == 200:
                print(f"{Fore.GREEN}\nSuccessfully predict the model.{Style.RESET_ALL}")

                predict_output = input(
                    "Do you want to save the full prediction output? (yes/no): "
                ).lower()
                if predict_output == "yes":
                    self.predict_model_output(response)
                return True
            else:
                logger.warning(f"{Fore.RED}Failed to predict model.{Style.RESET_ALL}")
                return False
        except Exception as e:
            logger.error(f"{Fore.RED}Error predicting model: {str(e)}{Style.RESET_ALL}")
            return False

    def initialize_register_model(
        self, config_path, connector_id=None, model_name=None, model_description=None
    ):
        """
        Orchestrates the entire model registration process.
        """
        try:
            # Load and check configuration
            config_result = self.load_and_check_config(config_path)
            if not config_result:
                return False
            ai_helper, _, _, _ = config_result

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
                logger.warning(f"{Fore.RED}Failed to register model.{Style.RESET_ALL}")
                return False
        except Exception:
            return False
