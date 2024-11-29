# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.


#  Licensed to Elasticsearch B.V. under one or more contributor
#  license agreements. See the NOTICE file distributed with
#  this work for additional information regarding copyright
#  ownership. Elasticsearch B.V. licenses this file to you under
#  the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.



import os
import json
import time
import boto3
from urllib.parse import urlparse
from colorama import Fore, Style, init
from AIConnectorHelper import AIConnectorHelper  # Ensure this module is accessible
import sys

init(autoreset=True)

class ModelRegister:
    def __init__(self, config, opensearch_client, opensearch_domain_name):
        # Initialize ModelRegister with necessary configurations
        self.config = config
        self.aws_region = config.get('region')
        self.opensearch_client = opensearch_client
        self.opensearch_domain_name = opensearch_domain_name
        self.opensearch_username = config.get('opensearch_username')
        self.opensearch_password = config.get('opensearch_password')
        self.iam_principal = config.get('iam_principal')
        self.embedding_dimension = int(config.get('embedding_dimension', 768))
        self.service_type = config.get('service_type', 'managed')
        self.bedrock_client = None
        if self.service_type != 'open-source':
            self.initialize_clients()

    def initialize_clients(self):
        # Initialize AWS clients only if necessary
        if self.service_type in ['managed', 'serverless']:
            try:
                self.bedrock_client = boto3.client('bedrock-runtime', region_name=self.aws_region)
                # Add any other clients initialization if needed
                time.sleep(7)
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
        Prompt the user to register a model or input an existing model ID.
        """
        print("\nTo proceed, you need to configure an embedding model.")
        print("1. Register a new embedding model")
        print("2. Use an existing embedding model ID")
        choice = input("Enter your choice (1-2): ").strip()

        if choice == '1':
            self.register_model_interactive()
        elif choice == '2':
            model_id = input("Please enter your existing embedding model ID: ").strip()
            if model_id:
                self.config['embedding_model_id'] = model_id
                self.save_config(self.config)
                print(f"{Fore.GREEN}Model ID '{model_id}' saved successfully in configuration.{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}No model ID provided. Cannot proceed without an embedding model.{Style.RESET_ALL}")
                sys.exit(1)  # Exit the setup as we cannot proceed without a model ID
        else:
            print(f"{Fore.RED}Invalid choice. Please run setup again and select a valid option.{Style.RESET_ALL}")
            sys.exit(1)  # Exit the setup as we cannot proceed without a valid choice
    def get_custom_model_details(self, default_input):
        """
        Prompt the user to enter custom model details or use default.
        Returns a dictionary with the model details.
        """
        print("\nDo you want to use the default configuration or provide custom model settings?")
        print("1. Use default configuration")
        print("2. Provide custom model settings")
        choice = input("Enter your choice (1-2): ").strip()

        if choice == '1':
            return default_input
        elif choice == '2':
            print("Please enter your model details as a JSON object.")
            print("Example:")
            print(json.dumps(default_input, indent=2))
            json_input = input("Enter your JSON object: ").strip()
            try:
                custom_details = json.loads(json_input)
                return custom_details
            except json.JSONDecodeError as e:
                print(f"{Fore.RED}Invalid JSON input: {e}{Style.RESET_ALL}")
                return None
        else:
            print(f"{Fore.RED}Invalid choice. Aborting model registration.{Style.RESET_ALL}")
            return None
        
    def save_config(self, config):
        # Save configuration to the config file
        import configparser
        parser = configparser.ConfigParser()
        parser['DEFAULT'] = config
        with open('config.ini', 'w') as f:
            parser.write(f)

    def register_model_interactive(self):
        """
        Interactive method to register a new embedding model during setup.
        """
        # Initialize clients
        if not self.initialize_clients():
            print(f"{Fore.RED}Failed to initialize AWS clients. Cannot proceed.{Style.RESET_ALL}")
            return

        # Ensure opensearch_endpoint is set
        if not self.config.get('opensearch_endpoint'):
            print(f"{Fore.RED}OpenSearch endpoint not set. Please run 'setup' command first.{Style.RESET_ALL}")
            return

        # Extract the IAM user name from the IAM principal ARN
        aws_user_name = self.get_iam_user_name_from_arn(self.iam_principal)

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
            aws_role_name=None  # Set to None or provide if applicable
        )

        # Prompt user to select a model
        print("Please select an embedding model to register:")
        print("1. Bedrock Titan Embedding Model")
        print("2. SageMaker Embedding Model")
        print("3. Cohere Embedding Model")
        print("4. OpenAI Embedding Model")
        model_choice = input("Enter your choice (1-4): ")

        # Call the appropriate method based on the user's choice
        if model_choice == '1':
            self.register_bedrock_model(helper)
        elif model_choice == '2':
            self.register_sagemaker_model(helper)
        elif model_choice == '3':
            self.register_cohere_model(helper)
        elif model_choice == '4':
            self.register_openai_model(helper)
        else:
            print(f"{Fore.RED}Invalid choice. Exiting model registration.{Style.RESET_ALL}")
            return

    def get_iam_user_name_from_arn(self, iam_principal_arn):
        """
        Extract the IAM user name from the IAM principal ARN.
        """
        # IAM user ARN format: arn:aws:iam::123456789012:user/user-name
        if iam_principal_arn and ':user/' in iam_principal_arn:
            return iam_principal_arn.split(':user/')[-1]
        else:
            return None

    def register_bedrock_model(self, helper):
        """
        Register a Bedrock embedding model by creating the necessary connector and model in OpenSearch.
        """
        # Prompt for necessary inputs
        bedrock_region = input(f"Enter your Bedrock region [{self.aws_region}]: ") or self.aws_region
        connector_role_name = "my_test_bedrock_connector_role"
        create_connector_role_name = "my_test_create_bedrock_connector_role"

        # Set up connector role inline policy
        connector_role_inline_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": ["bedrock:InvokeModel"],
                    "Effect": "Allow",
                    "Resource": "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v1"
                }
            ]
        }

        # Default connector input
        default_connector_input = {
            "name": "Amazon Bedrock Connector: titan embedding v1",
            "description": "The connector to Bedrock Titan embedding model",
            "version": 1,
            "protocol": "aws_sigv4",
            "parameters": {
                "region": bedrock_region,
                "service_name": "bedrock"
            },
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": f"https://bedrock-runtime.{bedrock_region}.amazonaws.com/model/amazon.titan-embed-text-v1/invoke",
                    "headers": {
                        "content-type": "application/json",
                        "x-amz-content-sha256": "required"
                    },
                    "request_body": "{ \"inputText\": \"${parameters.inputText}\" }",
                    "pre_process_function": "\n    StringBuilder builder = new StringBuilder();\n    builder.append(\"\\\"\");\n    String first = params.text_docs[0];\n    builder.append(first);\n    builder.append(\"\\\"\");\n    def parameters = \"{\" +\"\\\"inputText\\\":\" + builder + \"}\";\n    return  \"{\" +\"\\\"parameters\\\":\" + parameters + \"}\";",
                    "post_process_function": "\n      def name = \"sentence_embedding\";\n      def dataType = \"FLOAT32\";\n      if (params.embedding == null || params.embedding.length == 0) {\n        return params.message;\n      }\n      def shape = [params.embedding.length];\n      def json = \"{\" +\n                 \"\\\"name\\\":\\\"\" + name + \"\\\",\" +\n                 \"\\\"data_type\\\":\\\"\" + dataType + \"\\\",\" +\n                 \"\\\"shape\\\":\" + shape + \",\" +\n                 \"\\\"data\\\":\" + params.embedding +\n                 \"}\";\n      return json;\n    "
                }
            ]
        }

        # Get model details from user
        create_connector_input = self.get_custom_model_details(default_connector_input)
        if not create_connector_input:
            return  # Abort if no valid input

        # Create connector
        print("Creating connector...")
        connector_id = helper.create_connector_with_role(
            connector_role_inline_policy,
            connector_role_name,
            create_connector_role_name,
            create_connector_input,
            sleep_time_in_seconds=10
        )

        if not connector_id:
            print(f"{Fore.RED}Failed to create connector. Aborting.{Style.RESET_ALL}")
            return

        # Register model
        print("Registering model...")
        model_name = create_connector_input.get('name', 'Bedrock embedding model')
        description = create_connector_input.get('description', 'Bedrock embedding model for semantic search')
        model_id = helper.create_model(model_name, description, connector_id, create_connector_role_name)

        if not model_id:
            print(f"{Fore.RED}Failed to create model. Aborting.{Style.RESET_ALL}")
            return

        # Save model_id to config
        self.config['embedding_model_id'] = model_id
        self.save_config(self.config)
        print(f"{Fore.GREEN}Model registered successfully. Model ID '{model_id}' saved in configuration.{Style.RESET_ALL}")
        
    def register_sagemaker_model(self, helper):
        """
        Register a SageMaker embedding model by creating the necessary connector and model in OpenSearch.
        """
        # Prompt for necessary inputs
        sagemaker_endpoint_arn = input("Enter your SageMaker inference endpoint ARN: ")
        sagemaker_endpoint_url = input("Enter your SageMaker inference endpoint URL: ")
        sagemaker_region = input(f"Enter your SageMaker region [{self.aws_region}]: ") or self.aws_region
        connector_role_name = "my_test_sagemaker_connector_role"
        create_connector_role_name = "my_test_create_sagemaker_connector_role"

        # Set up connector role inline policy
        connector_role_inline_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": ["sagemaker:InvokeEndpoint"],
                    "Effect": "Allow",
                    "Resource": sagemaker_endpoint_arn
                }
            ]
        }

        # Create connector input
        create_connector_input = {
            "name": "SageMaker embedding model connector",
            "description": "Connector for my SageMaker embedding model",
            "version": "1.0",
            "protocol": "aws_sigv4",
            "parameters": {
                "region": sagemaker_region,
                "service_name": "sagemaker"
            },
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "headers": {
                        "content-type": "application/json"
                    },
                    "url": sagemaker_endpoint_url,
                    "request_body": "${parameters.input}",
                    "pre_process_function": "connector.pre_process.default.embedding",
                    "post_process_function": "connector.post_process.default.embedding"
                }
            ]
        }

        # Create connector
        connector_id = helper.create_connector_with_role(
            connector_role_inline_policy,
            connector_role_name,
            create_connector_role_name,
            create_connector_input,
            sleep_time_in_seconds=10
        )

        if not connector_id:
            print(f"{Fore.RED}Failed to create connector. Aborting.{Style.RESET_ALL}")
            return

        # Register model
        model_name = 'SageMaker embedding model'
        description = 'SageMaker embedding model for semantic search'
        model_id = helper.create_model(model_name, description, connector_id, create_connector_role_name)

        if not model_id:
            print(f"{Fore.RED}Failed to create model. Aborting.{Style.RESET_ALL}")
            return

        # Save model_id to config
        self.config['embedding_model_id'] = model_id
        self.save_config(self.config)
        print(f"{Fore.GREEN}Model registered successfully. Model ID: {model_id}{Style.RESET_ALL}")

    def register_cohere_model(self, helper):
        """
        Register a Cohere embedding model by creating the necessary connector and model in OpenSearch.
        """
        # Prompt for necessary inputs
        secret_name = input("Enter a name for the AWS Secrets Manager secret: ")
        secret_key = 'cohere_api_key'
        cohere_api_key = input("Enter your Cohere API key: ")
        secret_value = {secret_key: cohere_api_key}

        connector_role_name = "my_test_cohere_connector_role"
        create_connector_role_name = "my_test_create_cohere_connector_role"

        # Default connector input
        default_connector_input = {
            "name": "Cohere Embedding Model Connector",
            "description": "Connector for Cohere embedding model",
            "version": "1.0",
            "protocol": "http",
            "parameters": {
                "model": "embed-english-v3.0",
                "input_type": "search_document",
                "truncate": "END"
            },
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": "https://api.cohere.ai/v1/embed",
                    "headers": {
                        "Authorization": f"Bearer ${{credential.secretArn.{secret_key}}}",
                        "Request-Source": "unspecified:opensearch"
                    },
                    "request_body": "{ \"texts\": ${parameters.texts}, \"truncate\": \"${parameters.truncate}\", \"model\": \"${parameters.model}\", \"input_type\": \"${parameters.input_type}\" }",
                    "pre_process_function": "connector.pre_process.cohere.embedding",
                    "post_process_function": "connector.post_process.cohere.embedding"
                }
            ]
        }

        # Get model details from user
        create_connector_input = self.get_custom_model_details(default_connector_input)
        if not create_connector_input:
            return  # Abort if no valid input

        # Create connector
        connector_id = helper.create_connector_with_secret(
            secret_name,
            secret_value,
            connector_role_name,
            create_connector_role_name,
            create_connector_input,
            sleep_time_in_seconds=10
        )

        if not connector_id:
            print(f"{Fore.RED}Failed to create connector. Aborting.{Style.RESET_ALL}")
            return

        # Register model
        model_name = create_connector_input.get('name', 'Cohere embedding model')
        description = create_connector_input.get('description', 'Cohere embedding model for semantic search')
        model_id = helper.create_model(model_name, description, connector_id, create_connector_role_name)

        if not model_id:
            print(f"{Fore.RED}Failed to create model. Aborting.{Style.RESET_ALL}")
            return

        # Save model_id to config
        self.config['embedding_model_id'] = model_id
        self.save_config(self.config)
        print(f"{Fore.GREEN}Model registered successfully. Model ID: {model_id}{Style.RESET_ALL}")

    def register_openai_model(self, helper):
        """
        Register an OpenAI embedding model by creating the necessary connector and model in OpenSearch.
        """
        # Prompt for necessary inputs
        secret_name = input("Enter a name for the AWS Secrets Manager secret: ")
        secret_key = 'openai_api_key'
        openai_api_key = input("Enter your OpenAI API key: ")
        secret_value = {secret_key: openai_api_key}

        connector_role_name = "my_test_openai_connector_role"
        create_connector_role_name = "my_test_create_openai_connector_role"

        # Default connector input
        default_connector_input = {
            "name": "OpenAI Embedding Model Connector",
            "description": "Connector for OpenAI embedding model",
            "version": "1.0",
            "protocol": "http",
            "parameters": {
                "model": "text-embedding-ada-002"
            },
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": "https://api.openai.com/v1/embeddings",
                    "headers": {
                        "Authorization": f"Bearer ${{credential.secretArn.{secret_key}}}",
                        "Content-Type": "application/json"
                    },
                    "request_body": "{ \"input\": ${parameters.input}, \"model\": \"${parameters.model}\" }",
                    "pre_process_function": "connector.pre_process.openai.embedding",
                    "post_process_function": "connector.post_process.openai.embedding"
                }
            ]
        }

        # Get model details from user
        create_connector_input = self.get_custom_model_details(default_connector_input)
        if not create_connector_input:
            return  # Abort if no valid input

        # Create connector
        connector_id = helper.create_connector_with_secret(
            secret_name,
            secret_value,
            connector_role_name,
            create_connector_role_name,
            create_connector_input,
            sleep_time_in_seconds=10
        )

        if not connector_id:
            print(f"{Fore.RED}Failed to create connector. Aborting.{Style.RESET_ALL}")
            return

        # Register model
        model_name = create_connector_input.get('name', 'OpenAI embedding model')
        description = create_connector_input.get('description', 'OpenAI embedding model for semantic search')
        model_id = helper.create_model(model_name, description, connector_id, create_connector_role_name)

        if not model_id:
            print(f"{Fore.RED}Failed to create model. Aborting.{Style.RESET_ALL}")
            return

        # Save model_id to config
        self.config['embedding_model_id'] = model_id
        self.save_config(self.config)
        print(f"{Fore.GREEN}Model registered successfully. Model ID: {model_id}{Style.RESET_ALL}")

    def prompt_opensource_model_registration(self):
        """
        Handle model registration for open-source OpenSearch.
        """
        print("\nWould you like to register an embedding model now?")
        print("1. Yes, register a new model")
        print("2. No, I will register the model later")
        choice = input("Enter your choice (1-2): ").strip()

        if choice == '1':
            self.register_model_opensource_interactive()
        elif choice == '2':
            print("Skipping model registration. You can register models later using the appropriate commands.")
        else:
            print(f"{Fore.RED}Invalid choice. Skipping model registration.{Style.RESET_ALL}")

    def register_model_opensource_interactive(self):
        """
        Interactive method to register a new embedding model for open-source OpenSearch.
        """
        # Ensure OpenSearch client is initialized
        if not self.opensearch_client:
            print(f"{Fore.RED}OpenSearch client is not initialized. Please run setup again.{Style.RESET_ALL}")
            return

        # Prompt user to select a model
        print("\nPlease select an embedding model to register:")
        print("1. Cohere Embedding Model")
        print("2. OpenAI Embedding Model")
        print("3. Hugging Face Transformers Model")
        print("4. Custom PyTorch Model")
        model_choice = input("Enter your choice (1-4): ")

        if model_choice == '1':
            self.register_cohere_model_opensource()
        elif model_choice == '2':
            self.register_openai_model_opensource()
        elif model_choice == '3':
            self.register_huggingface_model()
        elif model_choice == '4':
            self.register_custom_pytorch_model()
        else:
            print(f"{Fore.RED}Invalid choice. Exiting model registration.{Style.RESET_ALL}")
            return

    def register_cohere_model_opensource(self):
        """
        Register a Cohere embedding model in open-source OpenSearch.
        """
        cohere_api_key = input("Enter your Cohere API key: ").strip()
        if not cohere_api_key:
            print(f"{Fore.RED}API key is required. Aborting.{Style.RESET_ALL}")
            return

        print("\nDo you want to use the default configuration or provide custom settings?")
        print("1. Use default configuration")
        print("2. Provide custom settings")
        config_choice = input("Enter your choice (1-2): ").strip()

        if config_choice == '1':
            # Use default configurations
            connector_payload = {
                "name": "Cohere Embedding Connector",
                "description": "Connector for Cohere embedding model",
                "version": "1.0",
                "protocol": "http",
                "parameters": {
                    "model": "embed-english-v3.0",
                    "input_type": "search_document",
                    "truncate": "END"
                },
                "credential": {
                    "cohere_key": cohere_api_key
                },
                "actions": [
                    {
                        "action_type": "predict",
                        "method": "POST",
                        "url": "https://api.cohere.ai/v1/embed",
                        "headers": {
                            "Authorization": "Bearer ${credential.cohere_key}",
                            "Request-Source": "unspecified:opensearch"
                        },
                        "request_body": "{ \"texts\": ${parameters.texts}, \"truncate\": \"${parameters.truncate}\", \"model\": \"${parameters.model}\", \"input_type\": \"${parameters.input_type}\" }",
                        "pre_process_function": "connector.pre_process.cohere.embedding",
                        "post_process_function": "connector.post_process.cohere.embedding"
                    }
                ]
            }
            model_group_payload = {
                "name": f"cohere_model_group_{int(time.time())}",
                "description": "Model group for Cohere models"
            }
        elif config_choice == '2':
            # Get custom configurations
            print("\nPlease enter your connector details as a JSON object.")
            connector_payload = self.get_custom_json_input()
            if not connector_payload:
                return

            print("\nPlease enter your model group details as a JSON object.")
            model_group_payload = self.get_custom_json_input()
            if not model_group_payload:
                return
        else:
            print(f"{Fore.RED}Invalid choice. Aborting model registration.{Style.RESET_ALL}")
            return

        # Register the connector
        try:
            connector_response = self.opensearch_client.transport.perform_request(
                method="POST",
                url="/_plugins/_ml/connectors/_create",
                body=connector_payload
            )
            connector_id = connector_response.get('connector_id')
            if not connector_id:
                print(f"{Fore.RED}Failed to register connector. Response: {connector_response}{Style.RESET_ALL}")
                return
            print(f"{Fore.GREEN}Connector registered successfully. Connector ID: {connector_id}{Style.RESET_ALL}")
        except Exception as ex:
            print(f"{Fore.RED}Error registering connector: {ex}{Style.RESET_ALL}")
            return

        # Create model group
        try:
            model_group_response = self.opensearch_client.transport.perform_request(
                method="POST",
                url="/_plugins/_ml/model_groups/_register",
                body=model_group_payload
            )
            model_group_id = model_group_response.get('model_group_id')
            if not model_group_id:
                print(f"{Fore.RED}Failed to create model group. Response: {model_group_response}{Style.RESET_ALL}")
                return
            print(f"{Fore.GREEN}Model group created successfully. Model Group ID: {model_group_id}{Style.RESET_ALL}")
        except Exception as ex:
            print(f"{Fore.RED}Error creating model group: {ex}{Style.RESET_ALL}")
            if 'illegal_argument_exception' in str(ex) and 'already being used' in str(ex):
                print(f"{Fore.YELLOW}A model group with this name already exists. Using the existing group.{Style.RESET_ALL}")
                model_group_id = str(ex).split('ID: ')[-1].strip("'.")
            else:
                return

        # Create model payload
        model_payload = {
            "name": connector_payload.get('name', 'Cohere embedding model'),
            "function_name": "REMOTE",
            "model_group_id": model_group_id,
            "description": connector_payload.get('description', 'Cohere embedding model for semantic search'),
            "connector_id": connector_id
        }

        # Register the model
        try:
            response = self.opensearch_client.transport.perform_request(
                method="POST",
                url="/_plugins/_ml/models/_register",
                body=model_payload
            )
            task_id = response.get('task_id')
            if task_id:
                print(f"{Fore.GREEN}Model registration initiated. Task ID: {task_id}{Style.RESET_ALL}")
                # Wait for the task to complete and retrieve the model_id
                model_id = self.wait_for_model_registration(task_id)
                if model_id:
                    # Deploy the model
                    deploy_response = self.opensearch_client.transport.perform_request(
                        method="POST",
                        url=f"/_plugins/_ml/models/{model_id}/_deploy"
                    )
                    print(f"{Fore.GREEN}Model deployed successfully. Model ID: {model_id}{Style.RESET_ALL}")
                    self.config['embedding_model_id'] = model_id
                    self.save_config(self.config)
                else:
                    print(f"{Fore.RED}Model registration failed or timed out.{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Failed to initiate model registration. Response: {response}{Style.RESET_ALL}")
        except Exception as ex:
            print(f"{Fore.RED}Error registering model: {ex}{Style.RESET_ALL}")

    def register_openai_model_opensource(self):
        """
        Register an OpenAI embedding model in open-source OpenSearch.
        """
        openai_api_key = input("Enter your OpenAI API key: ").strip()
        if not openai_api_key:
            print(f"{Fore.RED}API key is required. Aborting.{Style.RESET_ALL}")
            return

        print("\nDo you want to use the default configuration or provide custom settings?")
        print("1. Use default configuration")
        print("2. Provide custom settings")
        config_choice = input("Enter your choice (1-2): ").strip()

        if config_choice == '1':
            # Use default configurations
            connector_payload = {
                "name": "OpenAI Embedding Connector",
                "description": "Connector for OpenAI embedding model",
                "version": "1",
                "protocol": "http",
                "parameters": {
                    "model": "text-embedding-ada-002"
                },
                "credential": {
                    "openAI_key": openai_api_key
                },
                "actions": [
                    {
                        "action_type": "predict",
                        "method": "POST",
                        "url": "https://api.openai.com/v1/embeddings",
                        "headers": {
                            "Authorization": "Bearer ${credential.openAI_key}",
                            "Content-Type": "application/json"
                        },
                        "request_body": "{ \"input\": ${parameters.input}, \"model\": \"${parameters.model}\" }",
                        "pre_process_function": "connector.pre_process.openai.embedding",
                        "post_process_function": "connector.post_process.openai.embedding"
                    }
                ]
            }
            model_group_payload = {
                "name": f"openai_model_group_{int(time.time())}",
                "description": "Model group for OpenAI models"
            }
        elif config_choice == '2':
            # Get custom configurations
            print("\nPlease enter your connector details as a JSON object.")
            connector_payload = self.get_custom_json_input()
            if not connector_payload:
                return

            print("\nPlease enter your model group details as a JSON object.")
            model_group_payload = self.get_custom_json_input()
            if not model_group_payload:
                return
        else:
            print(f"{Fore.RED}Invalid choice. Aborting model registration.{Style.RESET_ALL}")
            return

        # Register the connector
        try:
            connector_response = self.opensearch_client.transport.perform_request(
                method="POST",
                url="/_plugins/_ml/connectors/_create",
                body=connector_payload
            )
            connector_id = connector_response.get('connector_id')
            if not connector_id:
                print(f"{Fore.RED}Failed to register connector. Response: {connector_response}{Style.RESET_ALL}")
                return
            print(f"{Fore.GREEN}Connector registered successfully. Connector ID: {connector_id}{Style.RESET_ALL}")
        except Exception as ex:
            print(f"{Fore.RED}Error registering connector: {ex}{Style.RESET_ALL}")
            return

        # Create model group
        try:
            model_group_response = self.opensearch_client.transport.perform_request(
                method="POST",
                url="/_plugins/_ml/model_groups/_register",
                body=model_group_payload
            )
            model_group_id = model_group_response.get('model_group_id')
            if not model_group_id:
                print(f"{Fore.RED}Failed to create model group. Response: {model_group_response}{Style.RESET_ALL}")
                return
            print(f"{Fore.GREEN}Model group created successfully. Model Group ID: {model_group_id}{Style.RESET_ALL}")
        except Exception as ex:
            print(f"{Fore.RED}Error creating model group: {ex}{Style.RESET_ALL}")
            if 'illegal_argument_exception' in str(ex) and 'already being used' in str(ex):
                print(f"{Fore.YELLOW}A model group with this name already exists. Using the existing group.{Style.RESET_ALL}")
                model_group_id = str(ex).split('ID: ')[-1].strip("'.")
            else:
                return

        # Create model payload
        model_payload = {
            "name": connector_payload.get('name', 'OpenAI embedding model'),
            "function_name": "REMOTE",
            "model_group_id": model_group_id,
            "description": connector_payload.get('description', 'OpenAI embedding model for semantic search'),
            "connector_id": connector_id
        }

        # Register the model
        try:
            response = self.opensearch_client.transport.perform_request(
                method="POST",
                url="/_plugins/_ml/models/_register",
                body=model_payload
            )
            task_id = response.get('task_id')
            if task_id:
                print(f"{Fore.GREEN}Model registration initiated. Task ID: {task_id}{Style.RESET_ALL}")
                # Wait for the task to complete and retrieve the model_id
                model_id = self.wait_for_model_registration(task_id)
                if model_id:
                    # Deploy the model
                    deploy_response = self.opensearch_client.transport.perform_request(
                        method="POST",
                        url=f"/_plugins/_ml/models/{model_id}/_deploy"
                    )
                    print(f"{Fore.GREEN}Model deployed successfully. Model ID: {model_id}{Style.RESET_ALL}")
                    self.config['embedding_model_id'] = model_id
                    self.save_config(self.config)
                else:
                    print(f"{Fore.RED}Model registration failed or timed out.{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Failed to initiate model registration. Response: {response}{Style.RESET_ALL}")
        except Exception as ex:
            print(f"{Fore.RED}Error registering model: {ex}{Style.RESET_ALL}")

    def get_custom_json_input(self):
        """Helper method to get custom JSON input from the user."""
        json_input = input("Enter your JSON object: ").strip()
        try:
            return json.loads(json_input)
        except json.JSONDecodeError as e:
            print(f"{Fore.RED}Invalid JSON input: {e}{Style.RESET_ALL}")
            return None


    def get_model_id_from_task(self, task_id, timeout=600, interval=10):
        """
        Wait for the model registration task to complete and return the model_id.
        """
        import time
        end_time = time.time() + timeout
        while time.time() < end_time:
            try:
                response = self.opensearch_client.transport.perform_request(
                    method="GET",
                    url=f"/_plugins/_ml/tasks/{task_id}"
                )
                state = response.get('state')
                if state == 'COMPLETED':
                    model_id = response.get('model_id')
                    return model_id
                elif state in ['FAILED', 'STOPPED']:
                    print(f"{Fore.RED}Model registration task {task_id} failed with state: {state}{Style.RESET_ALL}")
                    return None
                else:
                    print(f"Model registration task {task_id} is in state: {state}. Waiting...")
                    time.sleep(interval)
            except Exception as ex:
                print(f"{Fore.RED}Error checking task status: {ex}{Style.RESET_ALL}")
                time.sleep(interval)
        print(f"{Fore.RED}Timed out waiting for model registration to complete.{Style.RESET_ALL}")
        return None

    def register_huggingface_model(self):
        """
        Register a Hugging Face Transformers model in open-source OpenSearch.
        """
        print("\nDo you want to use the default configuration or provide custom settings?")
        print("1. Use default configuration")
        print("2. Provide custom settings")
        config_choice = input("Enter your choice (1-2): ").strip()

        if config_choice == '1':
            # Use default configurations
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            model_payload = {
                "name": f"huggingface_{model_name.split('/')[-1]}",
                "model_format": "TORCH_SCRIPT",
                "model_config": {
                    "embedding_dimension": self.embedding_dimension,
                    "framework_type": "SENTENCE_TRANSFORMERS",
                    "model_type": "bert",
                    "embedding_model": model_name
                },
                "description": f"Hugging Face Transformers model: {model_name}"
            }
        elif config_choice == '2':
            # Get custom configurations
            model_name = input("Enter the Hugging Face model ID (e.g., 'sentence-transformers/all-MiniLM-L6-v2'): ").strip()
            if not model_name:
                print(f"{Fore.RED}Model ID is required. Aborting.{Style.RESET_ALL}")
                return

            print("\nPlease enter your model details as a JSON object.")
            print("Example:")
            example_payload = {
                "name": f"huggingface_{model_name.split('/')[-1]}",
                "model_format": "TORCH_SCRIPT",
                "model_config": {
                    "embedding_dimension": self.embedding_dimension,
                    "framework_type": "SENTENCE_TRANSFORMERS",
                    "model_type": "bert",
                    "embedding_model": model_name
                },
                "description": f"Hugging Face Transformers model: {model_name}"
            }
            print(json.dumps(example_payload, indent=2))
            
            model_payload = self.get_custom_json_input()
            if not model_payload:
                return
        else:
            print(f"{Fore.RED}Invalid choice. Aborting model registration.{Style.RESET_ALL}")
            return

        # Register the model
        try:
            response = self.opensearch_client.transport.perform_request(
                method="POST",
                url="/_plugins/_ml/models/_register",
                body=model_payload
            )
            task_id = response.get('task_id')
            if task_id:
                print(f"{Fore.GREEN}Model registration initiated. Task ID: {task_id}{Style.RESET_ALL}")
                # Wait for the task to complete and retrieve the model_id
                model_id = self.wait_for_model_registration(task_id)
                if model_id:
                    # Deploy the model
                    deploy_response = self.opensearch_client.transport.perform_request(
                        method="POST",
                        url=f"/_plugins/_ml/models/{model_id}/_deploy"
                    )
                    print(f"{Fore.GREEN}Model deployed successfully. Model ID: {model_id}{Style.RESET_ALL}")
                    self.config['embedding_model_id'] = model_id
                    self.save_config(self.config)
                else:
                    print(f"{Fore.RED}Model registration failed or timed out.{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Failed to initiate model registration. Response: {response}{Style.RESET_ALL}")
        except Exception as ex:
            print(f"{Fore.RED}Error registering model: {ex}{Style.RESET_ALL}")

    def wait_for_model_registration(self, task_id, timeout=600, interval=10):
        """
        Wait for the model registration task to complete and return the model_id.
        """
        import time
        end_time = time.time() + timeout
        while time.time() < end_time:
            try:
                response = self.opensearch_client.transport.perform_request(
                    method="GET",
                    url=f"/_plugins/_ml/tasks/{task_id}"
                )
                state = response.get('state')
                if state == 'COMPLETED':
                    model_id = response.get('model_id')
                    return model_id
                elif state in ['FAILED', 'STOPPED']:
                    print(f"{Fore.RED}Model registration task {task_id} failed with state: {state}{Style.RESET_ALL}")
                    return None
                else:
                    print(f"Model registration task {task_id} is in state: {state}. Waiting...")
                    time.sleep(interval)
            except Exception as ex:
                print(f"{Fore.RED}Error checking task status: {ex}{Style.RESET_ALL}")
                time.sleep(interval)
        print(f"{Fore.RED}Timed out waiting for model registration to complete.{Style.RESET_ALL}")
        return None

def register_custom_pytorch_model(self):
    """
    Register a custom PyTorch model in open-source OpenSearch.
    """
    print("\nDo you want to use the default configuration or provide custom settings?")
    print("1. Use default configuration")
    print("2. Provide custom settings")
    config_choice = input("Enter your choice (1-2): ").strip()

    if config_choice == '1':
        # Use default configurations
        model_path = input("Enter the path to your PyTorch model file (.pt or .pth): ").strip()
        if not os.path.isfile(model_path):
            print(f"{Fore.RED}Model file not found at '{model_path}'. Aborting.{Style.RESET_ALL}")
            return

        model_name = os.path.basename(model_path).split('.')[0]
        model_payload = {
            "name": f"custom_pytorch_{model_name}",
            "model_format": "TORCH_SCRIPT",
            "model_config": {
                "embedding_dimension": self.embedding_dimension,
                "framework_type": "CUSTOM",
                "model_type": "bert"
            },
            "description": f"Custom PyTorch model: {model_name}"
        }
    elif config_choice == '2':
        # Get custom configurations
        model_path = input("Enter the path to your PyTorch model file (.pt or .pth): ").strip()
        if not os.path.isfile(model_path):
            print(f"{Fore.RED}Model file not found at '{model_path}'. Aborting.{Style.RESET_ALL}")
            return

        print("\nPlease enter your model details as a JSON object.")
        print("Example:")
        example_payload = {
            "name": "custom_pytorch_model",
            "model_format": "TORCH_SCRIPT",
            "model_config": {
                "embedding_dimension": self.embedding_dimension,
                "framework_type": "CUSTOM",
                "model_type": "bert"
            },
            "description": "Custom PyTorch model for semantic search"
        }
        print(json.dumps(example_payload, indent=2))
        
        model_payload = self.get_custom_json_input()
        if not model_payload:
            return
    else:
        print(f"{Fore.RED}Invalid choice. Aborting model registration.{Style.RESET_ALL}")
        return

    # Upload the model file to OpenSearch
    try:
        with open(model_path, 'rb') as f:
            model_content = f.read()

        # Use the ML plugin's model upload API
        upload_response = self.opensearch_client.transport.perform_request(
            method="POST",
            url="/_plugins/_ml/models/_upload",
            params={"model_name": model_payload['name']},
            body=model_content,
            headers={'Content-Type': 'application/octet-stream'}
        )
        if 'model_id' not in upload_response:
            print(f"{Fore.RED}Failed to upload model. Response: {upload_response}{Style.RESET_ALL}")
            return
        model_id = upload_response['model_id']
        print(f"{Fore.GREEN}Model uploaded successfully. Model ID: {model_id}{Style.RESET_ALL}")
    except Exception as ex:
        print(f"{Fore.RED}Error uploading model: {ex}{Style.RESET_ALL}")
        return

    # Add the model_id to the payload
    model_payload['model_id'] = model_id

    # Register the model
    try:
        response = self.opensearch_client.transport.perform_request(
            method="POST",
            url="/_plugins/_ml/models/_register",
            body=model_payload
        )
        task_id = response.get('task_id')
        if task_id:
            print(f"{Fore.GREEN}Model registration initiated. Task ID: {task_id}{Style.RESET_ALL}")
            # Wait for the task to complete and retrieve the model_id
            registered_model_id = self.wait_for_model_registration(task_id)
            if registered_model_id:
                # Deploy the model
                deploy_response = self.opensearch_client.transport.perform_request(
                    method="POST",
                    url=f"/_plugins/_ml/models/{registered_model_id}/_deploy"
                )
                print(f"{Fore.GREEN}Model deployed successfully. Model ID: {registered_model_id}{Style.RESET_ALL}")
                self.config['embedding_model_id'] = registered_model_id
                self.save_config(self.config)
            else:
                print(f"{Fore.RED}Model registration failed or timed out.{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Failed to initiate model registration. Response: {response}{Style.RESET_ALL}")
    except Exception as ex:
        print(f"{Fore.RED}Error registering model: {ex}{Style.RESET_ALL}")
