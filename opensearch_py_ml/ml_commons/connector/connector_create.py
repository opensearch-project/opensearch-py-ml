# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from colorama import Fore, Style, init
from rich.console import Console

from opensearch_py_ml.ml_commons.connector.AIConnectorHelper import AIConnectorHelper
from opensearch_py_ml.ml_commons.connector.connector_base import ConnectorBase
from opensearch_py_ml.ml_commons.connector.ml_models.AlephAlphaModel import (
    AlephAlphaModel,
)
from opensearch_py_ml.ml_commons.connector.ml_models.BedrockModel import BedrockModel
from opensearch_py_ml.ml_commons.connector.ml_models.CohereModel import CohereModel
from opensearch_py_ml.ml_commons.connector.ml_models.DeepSeekModel import DeepSeekModel
from opensearch_py_ml.ml_commons.connector.ml_models.OpenAIModel import OpenAIModel
from opensearch_py_ml.ml_commons.connector.ml_models.SageMakerModel import (
    SageMakerModel,
)

# Initialize Rich console for enhanced CLI outputs
console = Console()

# Initialize colorama for colored terminal output
init(autoreset=True)


class Create(ConnectorBase):
    """
    Handles the creation of connector.
    """

    def __init__(self):
        super().__init__()
        self.config = {}

    def create_command(self):
        """
        Main create command that orchestrates the entire connector creation process.
        """
        try:
            # Load configuration
            config = self.load_config()
            if not config:
                print(
                    f"{Fore.RED}No configuration found. Please run setup first.{Style.RESET_ALL}\n"
                )
                return False

            service_type = config.get("service_type")
            if service_type == "open-source":
                # For open-source, check username and password
                if not config.get("opensearch_domain_username") or not config.get(
                    "opensearch_domain_password"
                ):
                    print(
                        f"{Fore.RED}OpenSearch username or password not set. Please run setup first.{Style.RESET_ALL}\n"
                    )
                    return False
            else:
                # For managed service, check AWS-specific configurations
                if not config.get("opensearch_domain_region") or not config.get(
                    "opensearch_domain_name"
                ):
                    print(
                        f"{Fore.RED}AWS region or domain name not set. Please run setup first.{Style.RESET_ALL}\n"
                    )
                    return False

            # Create AIConnectorHelper instance
            ai_helper = AIConnectorHelper(
                service_type=config.get("service_type"),
                opensearch_domain_region=config.get("opensearch_domain_region"),
                opensearch_domain_name=config.get("opensearch_domain_name"),
                opensearch_domain_username=config.get("opensearch_domain_username"),
                opensearch_domain_password=config.get("opensearch_domain_password"),
                aws_user_name=config.get("aws_user_name"),
                aws_role_name=config.get("aws_role_name"),
                opensearch_domain_url=config.get("opensearch_domain_endpoint"),
            )

            # TODO: Add more supported connectors for managed and open-source
            if service_type == "managed":
                # Prompt for supported connectors
                print("\nPlease select a supported connector to create:")
                print("1. Amazon Bedrock")
                print("2. Amazon SageMaker")
                print("3. Cohere")
                print("4. OpenAI")
                connector_choice = input("Enter your choice (1-4): ").strip()

                if connector_choice == "1":
                    self.bedrock_model = BedrockModel(
                        opensearch_domain_region=config.get("opensearch_domain_region"),
                    )
                    result = self.bedrock_model.create_bedrock_connector(
                        ai_helper, self.config, self.save_config
                    )
                elif connector_choice == "2":
                    self.sagemaker_model = SageMakerModel(
                        opensearch_domain_region=config.get("opensearch_domain_region"),
                    )
                    result = self.sagemaker_model.create_sagemaker_connector(
                        ai_helper, self.config, self.save_config
                    )
                elif connector_choice == "3":
                    self.cohere_model = CohereModel()
                    result = self.cohere_model.create_cohere_connector(
                        ai_helper, self.config, self.save_config
                    )
                elif connector_choice == "4":
                    self.openai_model = OpenAIModel(
                        service_type=config.get("service_type"),
                    )
                    result = self.openai_model.create_openai_connector(
                        ai_helper, self.config, self.save_config
                    )
                else:
                    print(
                        f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Amazon Bedrock'.{Style.RESET_ALL}"
                    )
            else:
                # Prompt for supported connectors
                print("\nPlease select a supported connector to create:")
                print("1. Aleph Alpha")
                print("2. DeepSeek")
                print("3. OpenAI")
                connector_choice = input("Enter your choice (1-3): ").strip()

                if connector_choice == "1":
                    self.aleph_alpha_model = AlephAlphaModel()
                    result = self.aleph_alpha_model.create_aleph_alpha_connector(
                        ai_helper, self.config, self.save_config
                    )
                elif connector_choice == "2":
                    self.deepseek_model = DeepSeekModel()
                    result = self.deepseek_model.create_deepseek_connector(
                        ai_helper, self.config, self.save_config
                    )
                elif connector_choice == "3":
                    self.openai_model = OpenAIModel(
                        service_type=config.get("service_type"),
                    )
                    result = self.openai_model.create_openai_connector(
                        ai_helper, self.config, self.save_config
                    )
                else:
                    print(
                        f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Aleph Alpha'.{Style.RESET_ALL}"
                    )
                    self.aleph_alpha_model = AlephAlphaModel()
                    result = self.aleph_alpha_model.create_aleph_alpha_connector(
                        ai_helper, self.config, self.save_config
                    )
            return result
        except Exception as e:
            print(f"{Fore.RED}Error creating connector: {str(e)}{Style.RESET_ALL}")
            return False
