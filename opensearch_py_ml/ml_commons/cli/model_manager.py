# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import logging
from typing import Any, Dict, Optional

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

    def initialize_predict_model(
        self,
        config_path: str,
        model_id: Optional[str] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Orchestrates the entire model prediction process.

        Args:
            config_path: Path to the setup configuration file.
            model_id (optional): The model ID to use for prediction.
            body (optional): Prediction request body.

        Returns:
            bool: True if prediction successful, False otherwise
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
                    self.predict_model_output(model_id, response)
                return True
            else:
                logger.warning(f"{Fore.RED}Failed to predict model.{Style.RESET_ALL}")
                return False
        except Exception as e:
            logger.error(f"{Fore.RED}Error predicting model: {str(e)}{Style.RESET_ALL}")
            return False

    def initialize_register_model(
        self,
        config_path: str,
        connector_id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_description: Optional[str] = None,
        output_path: Optional[str] = None,
        interactive: bool = False,
    ) -> bool:
        """
        Orchestrates the entire model registration process.

        Args:
            config_path: Path to the setup configuration file.
            connector_id (optional): The connector ID to register the model with.
            model_name (optional): Name of the model.
            model_description (optional): Description of the model.
            output_path (optional): Path to save the output information.
            interactive (optional): Whether to run in interactive mode.

        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            # Load and check configuration
            config_result = self.load_and_check_config(config_path)
            if not config_result:
                return False
            ai_helper, _, _, _ = config_result

            # In non-interactive mode, ensure all required parameters are provided
            if not interactive:
                # Specify which parameters are missing
                missing_params = []
                if not model_name:
                    missing_params.append("--name")
                if not model_description:
                    missing_params.append("--description")
                if not connector_id:
                    missing_params.append("--connectorId")

                if missing_params:
                    missing_str = "\n\t".join(missing_params)
                    logger.error(
                        "%sMissing required parameters: \n\t%s%s",
                        Fore.RED,
                        missing_str,
                        Style.RESET_ALL,
                    )
                    return False  # Early return due to missing parameters

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
                    f"{Fore.GREEN}\nSuccessfully registered a model with IDs: {model_id}{Style.RESET_ALL}"
                )
                self.register_model_output(
                    model_id, model_name, connector_id, output_path, interactive
                )
                return True

            logger.warning(f"{Fore.RED}Failed to register model.{Style.RESET_ALL}")
            return False
        except Exception:
            return False
