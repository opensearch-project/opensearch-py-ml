# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.


import sys
import time

import boto3
from colorama import Fore, Style, init

from opensearch_py_ml.ml_commons.IAMRoleHelper import IAMRoleHelper
from opensearch_py_ml.ml_commons.rag_pipeline.rag.AIConnectorHelper import (
    AIConnectorHelper,
)
from opensearch_py_ml.ml_commons.rag_pipeline.rag.ml_models.BedrockModel import (
    BedrockModel,
)
from opensearch_py_ml.ml_commons.rag_pipeline.rag.ml_models.CohereModel import (
    CohereModel,
)
from opensearch_py_ml.ml_commons.rag_pipeline.rag.ml_models.HuggingFaceModel import (
    HuggingFaceModel,
)
from opensearch_py_ml.ml_commons.rag_pipeline.rag.ml_models.OpenAIModel import (
    OpenAIModel,
)
from opensearch_py_ml.ml_commons.rag_pipeline.rag.ml_models.PyTorchModel import (
    CustomPyTorchModel,
)

# Initialize colorama for colored terminal output
init(autoreset=True)


class ModelRegister:
    """
    Handles the registration of various embedding models with OpenSearch.
    Supports multiple model providers and manages their integration.
    """

    def __init__(self, config, opensearch_client, opensearch_domain_name):
        """
        Initialize ModelRegister with necessary configurations.

        :param config: Configuration dictionary containing necessary parameters.
        :param opensearch_client: Instance of the OpenSearch client.
        :param opensearch_domain_name: Name of the OpenSearch domain.
        """
        self.config = config
        self.aws_region = config.get("region")
        self.opensearch_client = opensearch_client
        self.opensearch_domain_name = opensearch_domain_name
        self.opensearch_endpoint = config.get("opensearch_endpoint")
        self.opensearch_username = config.get("opensearch_username")
        self.opensearch_password = config.get("opensearch_password")
        self.iam_principal = config.get("iam_principal")
        self.embedding_dimension = int(config.get("embedding_dimension", 768))
        self.service_type = config.get("service_type", "managed")

        # Initialize IAMRoleHelper with necessary parameters
        self.iam_role_helper = IAMRoleHelper(
            self.aws_region,
            self.opensearch_domain_name,
            self.opensearch_username,
            self.opensearch_password,
            self.iam_principal,
        )

        # Initialize AWS clients if the service type is not open-source
        if self.service_type != "open-source":
            self.initialize_clients()

        # Initialize instances of different model providers
        self.bedrock_model = BedrockModel(
            aws_region=self.aws_region,
            opensearch_domain_name=self.opensearch_domain_name,
            opensearch_username=self.opensearch_username,
            opensearch_password=self.opensearch_password,
            iam_role_helper=self.iam_role_helper,
        )
        self.openai_model = OpenAIModel(
            aws_region=self.aws_region,
            opensearch_domain_name=self.opensearch_domain_name,
            opensearch_username=self.opensearch_username,
            opensearch_password=self.opensearch_password,
            iam_role_helper=self.iam_role_helper,
        )
        self.cohere_model = CohereModel(
            aws_region=self.aws_region,
            opensearch_domain_name=self.opensearch_domain_name,
            opensearch_username=self.opensearch_username,
            opensearch_password=self.opensearch_password,
            iam_role_helper=self.iam_role_helper,
        )
        self.huggingface_model = HuggingFaceModel(
            aws_region=self.aws_region,
            opensearch_domain_name=self.opensearch_domain_name,
            opensearch_username=self.opensearch_username,
            opensearch_password=self.opensearch_password,
            iam_role_helper=self.iam_role_helper,
        )
        self.custom_pytorch_model = CustomPyTorchModel(
            aws_region=self.aws_region,
            opensearch_domain_name=self.opensearch_domain_name,
            opensearch_username=self.opensearch_username,
            opensearch_password=self.opensearch_password,
            iam_role_helper=self.iam_role_helper,
        )

    def initialize_clients(self) -> bool:
        """
        Initialize AWS clients based on the service type.

        :return: True if clients are initialized successfully, False otherwise.
        """
        if self.service_type in ["managed", "serverless"]:
            try:
                # Initialize Bedrock client for managed services
                self.bedrock_client = boto3.client(
                    "bedrock-runtime", region_name=self.aws_region
                )
                # Add any other clients initialization if needed
                time.sleep(7)  # Wait for client initialization
                print("AWS clients initialized successfully.")
                return True
            except Exception as e:
                print(f"Failed to initialize AWS clients: {e}")
                return False
        else:
            # No AWS clients needed for open-source
            return True

    def prompt_model_registration(self):
        """
        Prompt the user to either register a new embedding model or use an existing model ID.
        """
        print("\nTo proceed, you need to configure an embedding model.")
        print("1. Register a new embedding model")
        print("2. Use an existing embedding model ID")
        choice = input("Enter your choice (1-2): ").strip()

        if choice == "1":
            self.register_model_interactive()
        elif choice == "2":
            model_id = input("Please enter your existing embedding model ID: ").strip()
            if model_id:
                self.config["embedding_model_id"] = model_id
                self.save_config(self.config)
                print(
                    f"{Fore.GREEN}Model ID '{model_id}' saved successfully in configuration.{Style.RESET_ALL}"
                )
            else:
                print(
                    f"{Fore.RED}No model ID provided. Cannot proceed without an embedding model.{Style.RESET_ALL}"
                )
                sys.exit(1)  # Exit the setup as we cannot proceed without a model ID
        else:
            print(
                f"{Fore.RED}Invalid choice. Please run setup again and select a valid option.{Style.RESET_ALL}"
            )
            sys.exit(1)  # Exit the setup as we cannot proceed without a valid choice

    def save_config(self, config):
        """
        Save the updated configuration to the config file.

        :param config: Configuration dictionary to save.
        """
        import configparser

        parser = configparser.ConfigParser()
        parser["DEFAULT"] = config
        with open("config.ini", "w") as f:
            parser.write(f)

    def register_model_interactive(self):
        """
        Interactive method to register a new embedding model during setup.
        """
        # Initialize clients
        if not self.initialize_clients():
            print(
                f"{Fore.RED}Failed to initialize AWS clients. Cannot proceed.{Style.RESET_ALL}"
            )
            return

        # Ensure opensearch_endpoint is set
        if not self.config.get("opensearch_endpoint"):
            print(
                f"{Fore.RED}OpenSearch endpoint not set. Please run 'setup' command first.{Style.RESET_ALL}"
            )
            return

        # Extract the IAM user name from the IAM principal ARN
        aws_user_name = self.iam_role_helper.get_iam_user_name_from_arn(
            self.iam_principal
        )

        if not aws_user_name:
            print("Could not extract IAM user name from IAM principal ARN.")
            aws_user_name = input("Enter your AWS IAM user name: ")

        # Instantiate AIConnectorHelper

        helper = AIConnectorHelper(
            region=self.aws_region,
            opensearch_domain_name=self.opensearch_domain_name,
            opensearch_domain_username=self.opensearch_username,
            opensearch_domain_password=self.opensearch_password,
            aws_user_name=aws_user_name,
            aws_role_name=None,  # Set to None or provide if applicable
            opensearch_domain_url=self.opensearch_endpoint,  # Pass the endpoint from config
        )

        # Prompt user to select a model
        print("Please select an embedding model to register:")
        print("1. Bedrock Embedding Model")
        print("2. OpenAI Embedding Model")
        print("3. Cohere Embedding Model")
        print("4. Hugging Face Transformers Model")
        print("5. Custom PyTorch Model")
        model_choice = input("Enter your choice (1-5): ")

        # Call the appropriate method based on the user's choice
        if model_choice == "1":
            self.bedrock_model.register_bedrock_model(
                helper, self.config, self.save_config
            )
        elif model_choice == "2":
            if self.service_type != "open-source":
                self.openai_model.register_openai_model(
                    helper, self.config, self.save_config
                )
            else:
                self.openai_model.register_openai_model_opensource(
                    self.opensearch_client, self.config, self.save_config
                )
        elif model_choice == "3":
            if self.service_type != "open-source":
                self.cohere_model.register_cohere_model(
                    helper, self.config, self.save_config
                )
            else:
                self.cohere_model.register_cohere_model_opensource(
                    self.opensearch_client, self.config, self.save_config
                )
        elif model_choice == "4":
            if self.service_type != "open-source":
                print(
                    f"{Fore.RED}Hugging Face Transformers models are only supported in open-source OpenSearch.{Style.RESET_ALL}"
                )
            else:
                self.huggingface_model.register_huggingface_model(
                    self.opensearch_client, self.config, self.save_config
                )
        elif model_choice == "5":
            if self.service_type != "open-source":
                print(
                    f"{Fore.RED}Custom PyTorch models are only supported in open-source OpenSearch.{Style.RESET_ALL}"
                )
            else:
                self.custom_pytorch_model.register_custom_pytorch_model(
                    self.opensearch_client, self.config, self.save_config
                )
        else:
            print(
                f"{Fore.RED}Invalid choice. Exiting model registration.{Style.RESET_ALL}"
            )
            return

    def prompt_opensource_model_registration(self):
        """
        Handle model registration specifically for open-source OpenSearch.
        """
        print("\nWould you like to register an embedding model now?")
        print("1. Yes, register a new model")
        print("2. No, I will register the model later")
        choice = input("Enter your choice (1-2): ").strip()

        if choice == "1":
            self.register_model_opensource_interactive()
        elif choice == "2":
            print(
                "Skipping model registration. You can register models later using the appropriate commands."
            )
        else:
            print(
                f"{Fore.RED}Invalid choice. Skipping model registration.{Style.RESET_ALL}"
            )

    def register_model_opensource_interactive(self):
        """
        Interactive method to register a new embedding model for open-source OpenSearch.
        """
        # Ensure OpenSearch client is initialized
        if not self.opensearch_client:
            print(
                f"{Fore.RED}OpenSearch client is not initialized. Please run setup again.{Style.RESET_ALL}"
            )
            return

        # Prompt user to select a model
        print("\nPlease select an embedding model to register:")
        print("1. OpenAI Embedding Model")
        print("2. Cohere Embedding Model")
        print("3. Hugging Face Transformers Model")
        print("4. Custom PyTorch Model")
        model_choice = input("Enter your choice (1-4): ")

        if model_choice == "1":
            self.openai_model.register_openai_model_opensource(
                self.opensearch_client, self.config, self.save_config
            )
        elif model_choice == "2":
            self.cohere_model.register_cohere_model_opensource(
                self.opensearch_client, self.config, self.save_config
            )
        elif model_choice == "3":
            self.huggingface_model.register_huggingface_model(
                self.opensearch_client, self.config, self.save_config
            )
        elif model_choice == "4":
            self.custom_pytorch_model.register_custom_pytorch_model(
                self.opensearch_client, self.config, self.save_config
            )
        else:
            print(
                f"{Fore.RED}Invalid choice. Exiting model registration.{Style.RESET_ALL}"
            )
            return
