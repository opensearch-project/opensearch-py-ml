# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json

from colorama import Fore, Style, init
from rich.console import Console

from opensearch_py_ml.ml_commons.cli.AIConnectorHelper import AIConnectorHelper
from opensearch_py_ml.ml_commons.cli.connector_base import ConnectorBase

# Initialize Rich console for enhanced CLI outputs
console = Console()

# Initialize colorama for colored terminal output
init(autoreset=True)


class Predict(ConnectorBase):
    """
    Handles the model prediction.
    """

    def __init__(self):
        super().__init__()
        self.config = {}
        self.opensearch_domain_name = ""

    def predict_command(self, config_path, model_id=None, payload=None):
        """
        Main predict command to orchestrates the entire model prediction process.
        """
        try:
            # Load configuration
            config = self.load_config(config_path)
            if not config:
                print(
                    f"{Fore.RED}No configuration found. Please run setup first.{Style.RESET_ALL}\n"
                )
                return False

            opensearch_config = self.config.get("opensearch_config", {})
            aws_credentials = self.config.get("aws_credentials", {})
            opensearch_domain_endpoint = opensearch_config.get(
                "opensearch_domain_endpoint"
            )
            if not opensearch_domain_endpoint:
                print(
                    f"\n{Fore.RED}OpenSearch endpoint not set. Please run setup first.{Style.RESET_ALL}\n"
                )
                return False

            service_type = config.get("service_type")
            if service_type == "open-source":
                # For open-source, check username and password
                if not opensearch_config.get(
                    "opensearch_domain_username"
                ) or not opensearch_config.get("opensearch_domain_password"):
                    print(
                        f"{Fore.RED}OpenSearch username or password not set. Please run setup first.{Style.RESET_ALL}\n"
                    )
                    return False
                else:
                    self.opensearch_domain_name = None
            else:
                # For managed service, check AWS-specific configurations
                self.opensearch_domain_name = self.get_opensearch_domain_name(
                    opensearch_domain_endpoint
                )
                if (
                    not opensearch_config.get("opensearch_domain_region")
                    or not self.opensearch_domain_name
                ):
                    print(
                        f"{Fore.RED}AWS region or domain name not set. Please run setup first.{Style.RESET_ALL}\n"
                    )
                    return False

            # Create AIConnectorHelper instance
            ai_helper = AIConnectorHelper(
                service_type=config.get("service_type"),
                opensearch_domain_region=opensearch_config.get(
                    "opensearch_domain_region"
                ),
                opensearch_domain_name=self.opensearch_domain_name,
                opensearch_domain_username=opensearch_config.get(
                    "opensearch_domain_username"
                ),
                opensearch_domain_password=opensearch_config.get(
                    "opensearch_domain_password"
                ),
                aws_user_name=aws_credentials.get("aws_user_name"),
                aws_role_name=aws_credentials.get("aws_role_name"),
                opensearch_domain_url=opensearch_config.get(
                    "opensearch_domain_endpoint"
                ),
                aws_access_key=aws_credentials.get("aws_access_key"),
                aws_secret_access_key=aws_credentials.get("aws_secret_access_key"),
                aws_session_token=aws_credentials.get("aws_session_token"),
            )

            if not model_id:
                model_id = input("\nEnter the model ID: ").strip()
            if not payload:
                print(
                    "\nEnter your predict request payload as a JSON object (press Enter twice when done): "
                )
                json_input = ""
                lines = []
                while True:
                    line = input()
                    if line.strip() == "":
                        break
                    lines.append(line)

                json_input = "\n".join(lines)
            else:
                json_input = payload
                payload = json.loads(json_input)

            response, status = ai_helper.predict(model_id, payload)
            if status == 200:
                print(f"{Fore.GREEN}\nSuccessfully predict the model.{Style.RESET_ALL}")

                predict_output = input(
                    "Do you want to save the full prediction output? (yes/no): "
                ).lower()
                if predict_output == "yes":
                    self.predict_model_output(response)
                return True
            else:
                print(f"{Fore.RED}Failed to predict model.{Style.RESET_ALL}")
                return False
        except Exception as e:
            print(f"{Fore.RED}Error predicting model: {str(e)}{Style.RESET_ALL}")
            return False
