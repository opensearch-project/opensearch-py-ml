# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.connector.ml_models.model_base import ModelBase


class BedrockModel(ModelBase):
    def __init__(self, opensearch_domain_region):
        """
        Initializes the Bedrock model with necessary configurations.
        """
        self.opensearch_domain_region = opensearch_domain_region

    def create_bedrock_connector(self, helper, config, save_config_method):
        """
        Create Bedrock connector.
        """
        # Prompt for necessary inputs
        bedrock_region = (
            input(
                f"Enter your Bedrock region [{self.opensearch_domain_region}]: "
            ).strip()
            or self.opensearch_domain_region
        )

        print("\nPlease select an embedding model for the connector creation: ")
        print("1. Cohere embedding model")
        print("2. Titan embedding model")
        model_type = input("Enter your choice (1-2): ").strip()

        connector_role_name = "bedrock_connector_role"
        create_connector_role_name = "create_bedrock_connector_role"

        if model_type == "1":
            connector_role_inline_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Action": ["bedrock:InvokeModel"],
                        "Effect": "Allow",
                        "Resource": "arn:aws:bedrock:*::foundation-model/cohere.embed-english-v3",
                    }
                ],
            }
            default_connector_input = {
                "name": "Amazon Bedrock Cohere Connector: embedding v3",
                "description": "The connector to Bedrock Cohere embedding model",
                "version": 1,
                "protocol": "aws_sigv4",
                "parameters": {
                    "region": bedrock_region,
                    "service_name": "bedrock",
                    "input_type": "search_document",
                    "truncate": "END",
                },
                "actions": [
                    {
                        "action_type": "predict",
                        "method": "POST",
                        "url": f"https://bedrock-runtime.{bedrock_region}.amazonaws.com/model/cohere.embed-english-v3/invoke",
                        "headers": {
                            "content-type": "application/json",
                            "x-amz-content-sha256": "required",
                        },
                        "request_body": '{ "texts": ${parameters.texts}, "truncate": "${parameters.truncate}", "input_type": "${parameters.input_type}" }',
                        "pre_process_function": "connector.pre_process.cohere.embedding",
                        "post_process_function": "connector.post_process.cohere.embedding",
                    }
                ],
            }
        elif model_type == "2":
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
            default_connector_input = {
                "name": "Amazon Bedrock Connector: titan embedding v1",
                "description": "The connector to bedrock Titan embedding model",
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
        else:
            print(
                f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Cohere embedding model'.{Style.RESET_ALL}"
            )
            connector_role_inline_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Action": ["bedrock:InvokeModel"],
                        "Effect": "Allow",
                        "Resource": "arn:aws:bedrock:*::foundation-model/cohere.embed-english-v3",
                    }
                ],
            }
            default_connector_input = {
                "name": "Amazon Bedrock Cohere Connector: embedding v3",
                "description": "The connector to Bedrock Cohere embedding model",
                "version": 1,
                "protocol": "aws_sigv4",
                "parameters": {
                    "region": bedrock_region,
                    "service_name": "bedrock",
                    "input_type": "search_document",
                    "truncate": "END",
                },
                "actions": [
                    {
                        "action_type": "predict",
                        "method": "POST",
                        "url": f"https://bedrock-runtime.{bedrock_region}.amazonaws.com/model/cohere.embed-english-v3/invoke",
                        "headers": {
                            "content-type": "application/json",
                            "x-amz-content-sha256": "required",
                        },
                        "request_body": '{ "texts": ${parameters.texts}, "truncate": "${parameters.truncate}", "input_type": "${parameters.input_type}" }',
                        "pre_process_function": "connector.pre_process.cohere.embedding",
                        "post_process_function": "connector.post_process.cohere.embedding",
                    }
                ],
            }

        # Get model details from user
        create_connector_input = self.get_custom_model_details(default_connector_input)
        if not create_connector_input:
            return  # Abort if no valid input

        # Create connector
        print("\nCreating Bedrock connector...")
        connector_id = helper.create_connector_with_role(
            connector_role_inline_policy,
            connector_role_name,
            create_connector_role_name,
            create_connector_input,
            sleep_time_in_seconds=10,
        )

        if connector_id:
            print(
                f"{Fore.GREEN}\nSuccessfully created Bedrock connector with ID: {connector_id}{Style.RESET_ALL}"
            )
            # Update config with connector ID if needed
            config["connector_id"] = connector_id
            save_config_method(config)
            return True
        else:
            print(f"{Fore.RED}Failed to create Bedrock connector.{Style.RESET_ALL}")
            return False
