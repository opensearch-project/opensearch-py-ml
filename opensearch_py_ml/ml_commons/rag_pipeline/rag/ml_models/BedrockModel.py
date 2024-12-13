# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json

from colorama import Fore, Style


class BedrockModel:
    def __init__(
        self,
        aws_region,
        opensearch_domain_name,
        opensearch_username,
        opensearch_password,
        iam_role_helper,
    ):
        """
        Initializes the BedrockModel with necessary configurations.

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

    def register_bedrock_model(self, helper, config, save_config_method):
        """
        Register a Managed Bedrock embedding model by creating the necessary connector and model in OpenSearch.

        Args:
            helper (AIConnectorHelper): Instance of AIConnectorHelper.
            config (dict): Configuration dictionary.
            save_config_method (function): Method to save the configuration.
        """
        # Prompt for necessary inputs
        bedrock_region = (
            input(f"Enter your Bedrock region [{self.aws_region}]: ") or self.aws_region
        )
        connector_role_name = "my_test_bedrock_connector_role"
        create_connector_role_name = "my_test_create_bedrock_connector_role"

        # Set up connector role inline policy
        connector_role_inline_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": ["bedrock:InvokeModel"],
                    "Effect": "Allow",
                    "Resource": "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v1",
                }
            ],
        }

        # Default connector input
        default_connector_input = {
            "name": "Amazon Bedrock Connector: titan embedding v1",
            "description": "The connector to Bedrock Titan embedding model",
            "version": 1,
            "protocol": "aws_sigv4",
            "parameters": {"region": bedrock_region, "service_name": "bedrock"},
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": f"https://bedrock-runtime.{bedrock_region}.amazonaws.com/model/amazon.titan-embed-text-v1/invoke",
                    "headers": {
                        "content-type": "application/json",
                        "x-amz-content-sha256": "required",
                    },
                    "request_body": '{ "inputText": "${parameters.inputText}" }',
                    "pre_process_function": '\n    StringBuilder builder = new StringBuilder();\n    builder.append("\\"");\n    String first = params.text_docs[0];\n    builder.append(first);\n    builder.append("\\"");\n    def parameters = "{" +"\\"inputText\\":" + builder + "}";\n    return  "{" +"\\"parameters\\":" + parameters + "}";',
                    "post_process_function": '\n      def name = "sentence_embedding";\n      def dataType = "FLOAT32";\n      if (params.embedding == null || params.embedding.length == 0) {\n        return params.message;\n      }\n      def shape = [params.embedding.length];\n      def json = "{" +\n                 "\\"name\\":\\"" + name + "\\"," +\n                 "\\"data_type\\":\\"" + dataType + "\\"," +\n                 "\\"shape\\":" + shape + "," +\n                 "\\"data\\":" + params.embedding +\n                 "}";\n      return json;\n    ',
                }
            ],
        }

        # Get model details from user
        create_connector_input = self.get_custom_model_details(default_connector_input)
        if not create_connector_input:
            return  # Abort if no valid input

        # Create connector
        print("Creating Bedrock connector...")
        connector_id = helper.create_connector_with_role(
            connector_role_inline_policy,
            connector_role_name,
            create_connector_role_name,
            create_connector_input,
            sleep_time_in_seconds=10,
        )

        if not connector_id:
            print(
                f"{Fore.RED}Failed to create Bedrock connector. Aborting.{Style.RESET_ALL}"
            )
            return

        # Register model
        print("Registering Bedrock model...")
        model_name = create_connector_input.get("name", "Bedrock embedding model")
        description = create_connector_input.get(
            "description", "Bedrock embedding model for semantic search"
        )
        model_id = helper.create_model(
            model_name, description, connector_id, create_connector_role_name
        )

        if not model_id:
            print(
                f"{Fore.RED}Failed to create Bedrock model. Aborting.{Style.RESET_ALL}"
            )
            return

        # Save model_id to config
        self.save_model_id(config, save_config_method, model_id)
        print(
            f"{Fore.GREEN}Bedrock model registered successfully. Model ID '{model_id}' saved in configuration.{Style.RESET_ALL}"
        )

    def save_model_id(self, config, save_config_method, model_id):
        """
        Save the model ID to the configuration.

        Args:
            config (dict): Configuration dictionary.
            save_config_method (function): Method to save the configuration.
            model_id (str): The model ID to save.
        """
        config["embedding_model_id"] = model_id
        save_config_method(config)

    def get_custom_model_details(self, default_input):
        print(
            "\nDo you want to use the default configuration or provide custom model settings?"
        )
        print("1. Use default configuration")
        print("2. Provide custom model settings")
        choice = input("Enter your choice (1-2): ").strip()

        if choice == "1":
            return default_input
        elif choice == "2":
            print("Please enter your model details as a JSON object.")
            print("Example:")
            print(json.dumps(default_input, indent=2))
            json_input = input("Enter your JSON object: ").strip()
            try:
                custom_details = json.loads(json_input)
                return custom_details
            except json.JSONDecodeError as e:
                print(f"Invalid JSON input: {e}")
                return None
        else:
            print("Invalid choice. Aborting model registration.")
            return None
