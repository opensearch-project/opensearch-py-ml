# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from colorama import Fore, Style, init
from rich.console import Console

from opensearch_py_ml.ml_commons.cli.AIConnectorHelper import AIConnectorHelper
from opensearch_py_ml.ml_commons.cli.connector_base import ConnectorBase
from opensearch_py_ml.ml_commons.cli.ml_models.AlephAlphaModel import AlephAlphaModel
from opensearch_py_ml.ml_commons.cli.ml_models.BedrockModel import BedrockModel
from opensearch_py_ml.ml_commons.cli.ml_models.CohereModel import CohereModel
from opensearch_py_ml.ml_commons.cli.ml_models.DeepSeekModel import DeepSeekModel
from opensearch_py_ml.ml_commons.cli.ml_models.OpenAIModel import OpenAIModel
from opensearch_py_ml.ml_commons.cli.ml_models.SageMakerModel import SageMakerModel

# Initialize Rich console for enhanced CLI outputs
console = Console()

# Initialize colorama for colored terminal output
init(autoreset=True)


class Create(ConnectorBase):
    """
    Handles the creation of connector.
    """

    def __init__(self):
        """
        Initialize the Create class.
        """
        super().__init__()
        self.config = {}
        self.opensearch_domain_name = ""

    def create_command(self, connector_config_path=None):
        """
        Main create command that orchestrates the entire connector creation process.
        """
        try:
            # Check if connector config file path is given in the command
            if connector_config_path:
                connector_config = self.load_connector_config(connector_config_path)
                if not connector_config:
                    print(
                        f"{Fore.RED}No connector configuration found.{Style.RESET_ALL}\n"
                    )
                    return False
                setup_config_path = connector_config.get("setup_config_path")
                if not setup_config_path:
                    print(
                        f"{Fore.RED}No setup configuration found. Please run setup first.{Style.RESET_ALL}\n"
                    )
                    return False
            else:
                setup_config_path = input(
                    "\nEnter the path to your existing setup configuration file: "
                ).strip()

            config = self.load_config(setup_config_path)
            if not config:
                print(
                    f"{Fore.RED}No setup configuration found. Please run setup first.{Style.RESET_ALL}\n"
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

            # Set the initial value of the connector creation result to False
            result = False

            # Retrieve the connector information if the connector config path is given in the command
            if connector_config_path:
                connector_name = connector_config.get("connector_name")
                model_name = connector_config.get("model_name")
                connector_role_prefix = connector_config.get("connector_role_prefix")
                region = connector_config.get("region")
                model_arn = connector_config.get("model_arn")
                connector_payload = connector_config.get("connector_payload")
                api_key = connector_config.get("api_key")
                secret_name = connector_config.get("connector_secret_name")
                endpoint_arn = connector_config.get("inference_endpoint_arn")
                endpoint_url = connector_config.get("inference_endpoint_url")

                if service_type == "amazon-opensearch-service":
                    if connector_name == "Amazon Bedrock":
                        self.bedrock_model = BedrockModel(
                            opensearch_domain_region=opensearch_config.get(
                                "opensearch_domain_region"
                            ),
                        )
                        result = self.bedrock_model.create_bedrock_connector(
                            ai_helper,
                            self.connector_output,
                            connector_role_prefix=connector_role_prefix,
                            region=region,
                            model_name=model_name,
                            model_arn=model_arn,
                            connector_payload=connector_payload,
                        )
                    elif connector_name == "Amazon SageMaker":
                        self.sagemaker_model = SageMakerModel(
                            opensearch_domain_region=opensearch_config.get(
                                "opensearch_domain_region"
                            ),
                        )
                        result = self.sagemaker_model.create_sagemaker_connector(
                            ai_helper,
                            self.connector_output,
                            connector_role_prefix=connector_role_prefix,
                            region=region,
                            model_name=model_name,
                            endpoint_arn=endpoint_arn,
                            endpoint_url=endpoint_url,
                            connector_payload=connector_payload,
                        )
                    elif connector_name == "Cohere":
                        self.cohere_model = CohereModel()
                        result = self.cohere_model.create_cohere_connector(
                            ai_helper,
                            self.connector_output,
                            connector_role_prefix=connector_role_prefix,
                            model_name=model_name,
                            api_key=api_key,
                            connector_payload=connector_payload,
                            secret_name=secret_name,
                        )
                    elif connector_name == "DeepSeek":
                        self.deepseek_model = DeepSeekModel(
                            service_type=config.get("service_type"),
                        )
                        result = self.deepseek_model.create_deepseek_connector(
                            ai_helper,
                            self.connector_output,
                            connector_role_prefix=connector_role_prefix,
                            model_name=model_name,
                            api_key=api_key,
                            connector_payload=connector_payload,
                            secret_name=secret_name,
                        )
                    elif connector_name == "OpenAI":
                        self.openai_model = OpenAIModel(
                            service_type=config.get("service_type"),
                        )
                        result = self.openai_model.create_openai_connector(
                            ai_helper,
                            self.connector_output,
                            connector_role_prefix=connector_role_prefix,
                            model_name=model_name,
                            api_key=api_key,
                            connector_payload=connector_payload,
                            secret_name=secret_name,
                        )
                    else:
                        print(
                            f"{Fore.RED}Invalid connector. Please make sure you provide the correct connector name.{Style.RESET_ALL}"
                        )
                else:
                    if connector_name == "Aleph Alpha":
                        self.aleph_alpha_model = AlephAlphaModel()
                        result = self.aleph_alpha_model.create_aleph_alpha_connector(
                            ai_helper,
                            self.connector_output,
                            model_name=model_name,
                            api_key=api_key,
                            connector_payload=connector_payload,
                        )
                    elif connector_name == "DeepSeek":
                        self.deepseek_model = DeepSeekModel(
                            service_type=config.get("service_type"),
                        )
                        result = self.deepseek_model.create_deepseek_connector(
                            ai_helper,
                            self.connector_output,
                            connector_role_prefix=connector_role_prefix,
                            model_name=model_name,
                            api_key=api_key,
                            connector_payload=connector_payload,
                            secret_name=secret_name,
                        )
                    elif connector_name == "OpenAI":
                        self.openai_model = OpenAIModel(
                            service_type=config.get("service_type"),
                        )
                        result = self.openai_model.create_openai_connector(
                            ai_helper,
                            self.connector_output,
                            connector_role_prefix=connector_role_prefix,
                            model_name=model_name,
                            api_key=api_key,
                            connector_payload=connector_payload,
                            secret_name=secret_name,
                        )
                    else:
                        print(
                            f"{Fore.RED}Invalid connector. Please make sure you provide the correct connector name.{Style.RESET_ALL}"
                        )
                return result, setup_config_path

            # TODO: Add more supported connectors for amazon-opensearch-service and open-source
            if service_type == "amazon-opensearch-service":
                # Prompt for supported connectors
                print("\nPlease select a supported connector to create:")
                print("1. Amazon Bedrock")
                print("2. Amazon SageMaker")
                print("3. Cohere")
                print("4. DeepSeek")
                print("5. OpenAI")
                connector_choice = input("Enter your choice (1-5): ").strip()

                if connector_choice == "1":
                    self.bedrock_model = BedrockModel(
                        opensearch_domain_region=opensearch_config.get(
                            "opensearch_domain_region"
                        ),
                    )
                    result = self.bedrock_model.create_bedrock_connector(
                        ai_helper, self.connector_output
                    )
                elif connector_choice == "2":
                    self.sagemaker_model = SageMakerModel(
                        opensearch_domain_region=opensearch_config.get(
                            "opensearch_domain_region"
                        ),
                    )
                    result = self.sagemaker_model.create_sagemaker_connector(
                        ai_helper, self.connector_output
                    )
                elif connector_choice == "3":
                    self.cohere_model = CohereModel()
                    result = self.cohere_model.create_cohere_connector(
                        ai_helper, self.connector_output
                    )
                elif connector_choice == "4":
                    self.deepseek_model = DeepSeekModel(
                        service_type=config.get("service_type"),
                    )
                    result = self.deepseek_model.create_deepseek_connector(
                        ai_helper, self.connector_output
                    )
                elif connector_choice == "5":
                    self.openai_model = OpenAIModel(
                        service_type=config.get("service_type"),
                    )
                    result = self.openai_model.create_openai_connector(
                        ai_helper, self.connector_output
                    )
                else:
                    print(
                        f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Amazon Bedrock'.{Style.RESET_ALL}"
                    )
                    self.bedrock_model = BedrockModel(
                        opensearch_domain_region=opensearch_config.get(
                            "opensearch_domain_region"
                        ),
                    )
                    result = self.bedrock_model.create_bedrock_connector(
                        ai_helper, self.connector_output
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
                        ai_helper, self.connector_output
                    )
                elif connector_choice == "2":
                    self.deepseek_model = DeepSeekModel(
                        service_type=config.get("service_type"),
                    )
                    result = self.deepseek_model.create_deepseek_connector(
                        ai_helper, self.connector_output
                    )
                elif connector_choice == "3":
                    self.openai_model = OpenAIModel(
                        service_type=config.get("service_type"),
                    )
                    result = self.openai_model.create_openai_connector(
                        ai_helper, self.connector_output
                    )
                else:
                    print(
                        f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Aleph Alpha'.{Style.RESET_ALL}"
                    )
                    self.aleph_alpha_model = AlephAlphaModel()
                    result = self.aleph_alpha_model.create_aleph_alpha_connector(
                        ai_helper,
                        self.connector_output,
                        model_name=model_name,
                        api_key=api_key,
                        connector_payload=connector_payload,
                    )
            return result, setup_config_path
        except Exception as e:
            print(f"{Fore.RED}Error creating connector: {str(e)}{Style.RESET_ALL}")
            return False
