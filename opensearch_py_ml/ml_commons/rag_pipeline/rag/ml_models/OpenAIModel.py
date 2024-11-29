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
import json
from colorama import Fore, Style
import time

class OpenAIModel:
    def __init__(self, aws_region, opensearch_domain_name, opensearch_username, opensearch_password, iam_role_helper):
        """
        Initializes the OpenAIModel with necessary configurations.
        
        Args:
            aws_region (str): AWS region.
            opensearch_domain_name (str): OpenSearch domain name.
            opensearch_username (str): OpenSearch username.
            opensearch_password (str): OpenSearch password.
            iam_role_helper (IAMRoleHelper): Instance of IAMRoleHelper.
        """
        self.aws_region = aws_region
        self.opensearch_domain_name = opensearch_domain_name
        self.opensearch_username = opensearch_username
        self.opensearch_password = opensearch_password
        self.iam_role_helper = iam_role_helper

    def register_openai_model(self, helper, config, save_config_method):
        """
        Register a Managed OpenAI embedding model by creating the necessary connector and model in OpenSearch.

        Args:
            helper (AIConnectorHelper): Instance of AIConnectorHelper.
            config (dict): Configuration dictionary.
            save_config_method (function): Method to save the configuration.
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
        config['embedding_model_id'] = model_id
        save_config_method(config)
        print(f"{Fore.GREEN}OpenAI model registered successfully. Model ID '{model_id}' saved in configuration.{Style.RESET_ALL}")

    def register_openai_model_opensource(self, opensearch_client, config, save_config_method):
        """
        Register an OpenAI embedding model in open-source OpenSearch.

        Args:
            opensearch_client: OpenSearch client instance.
            config (dict): Configuration dictionary.
            save_config_method (function): Method to save the configuration.
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
            connector_response = opensearch_client.transport.perform_request(
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
            model_group_response = opensearch_client.transport.perform_request(
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
            response = opensearch_client.transport.perform_request(
                method="POST",
                url="/_plugins/_ml/models/_register",
                body=model_payload
            )
            task_id = response.get('task_id')
            if task_id:
                print(f"{Fore.GREEN}Model registration initiated. Task ID: {task_id}{Style.RESET_ALL}")
                # Wait for the task to complete and retrieve the model_id
                model_id = self.wait_for_model_registration(opensearch_client, task_id)
                if model_id:
                    # Deploy the model
                    deploy_response = opensearch_client.transport.perform_request(
                        method="POST",
                        url=f"/_plugins/_ml/models/{model_id}/_deploy"
                    )
                    print(f"{Fore.GREEN}Model deployed successfully. Model ID: {model_id}{Style.RESET_ALL}")
                    config['embedding_model_id'] = model_id
                    save_config_method(config)
                else:
                    print(f"{Fore.RED}Model registration failed or timed out.{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Failed to initiate model registration. Response: {response}{Style.RESET_ALL}")
        except Exception as ex:
            print(f"{Fore.RED}Error registering model: {ex}{Style.RESET_ALL}")

    def get_custom_model_details(self, default_input):
        """
        Prompt the user to enter custom model details or use default.
        Returns a dictionary with the model details.

        Args:
            default_input (dict): Default model configuration.

        Returns:
            dict or None: Custom or default model configuration, or None if invalid input.
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

    def get_custom_json_input(self):
        """Helper method to get custom JSON input from the user."""
        json_input = input("Enter your JSON object: ").strip()
        try:
            return json.loads(json_input)
        except json.JSONDecodeError as e:
            print(f"{Fore.RED}Invalid JSON input: {e}{Style.RESET_ALL}")
            return None

    def wait_for_model_registration(self, opensearch_client, task_id, timeout=600, interval=10):
        """
        Wait for the model registration task to complete and return the model_id.

        Args:
            opensearch_client: OpenSearch client instance.
            task_id (str): Task ID to monitor.
            timeout (int): Maximum time to wait in seconds.
            interval (int): Time between status checks in seconds.

        Returns:
            str or None: The model ID if successful, else None.
        """
        end_time = time.time() + timeout
        while time.time() < end_time:
            try:
                response = opensearch_client.transport.perform_request(
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