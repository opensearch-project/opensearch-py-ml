# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.


from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.model_base import ModelBase


class BedrockModel(ModelBase):
    def __init__(self, opensearch_domain_region):
        """
        Initializes the Bedrock model with necessary configuration.
        """
        self.opensearch_domain_region = opensearch_domain_region

    def create_connector(
        self,
        helper,
        save_config_method,
        connector_role_prefix=None,
        region=None,
        model_name=None,
        model_arn=None,
        connector_body=None,
    ):
        """
        Create Bedrock connector.
        """
        # Set trusted connector endpoints for Bedrock
        trusted_endpoint = (
            "^https://bedrock-runtime\\..*[a-z0-9-]\\.amazonaws\\.com/.*$"
        )
        self.set_trusted_endpoint(helper, trusted_endpoint)

        # Prompt to choose model
        model_type = self.get_model_details(
            "Amazon Bedrock", "amazon-opensearch-service", model_name
        )

        # Create connector role
        connector_role_name, create_connector_role_name = self.create_connector_role(
            connector_role_prefix, "bedrock"
        )

        if model_type == "1":
            if not region:
                region = (
                    input(
                        f"Enter your Bedrock region [{self.opensearch_domain_region}]: "
                    ).strip()
                    or self.opensearch_domain_region
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
            connector_body = {
                "name": "Amazon Bedrock Cohere Connector: embedding v3",
                "description": "The connector to Bedrock Cohere embedding model",
                "version": 1,
                "protocol": "aws_sigv4",
                "parameters": {
                    "region": region,
                    "service_name": "bedrock",
                    "input_type": "search_document",
                    "truncate": "END",
                },
                "actions": [
                    {
                        "action_type": "predict",
                        "method": "POST",
                        "url": f"https://bedrock-runtime.{region}.amazonaws.com/model/cohere.embed-english-v3/invoke",
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
            if not region:
                region = (
                    input(
                        f"Enter your Bedrock region [{self.opensearch_domain_region}]: "
                    ).strip()
                    or self.opensearch_domain_region
                )

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
            connector_body = {
                "name": "Amazon Bedrock Connector: titan embedding v1",
                "description": "The connector to bedrock Titan embedding model",
                "version": 1,
                "protocol": "aws_sigv4",
                "parameters": {"region": region, "service_name": "bedrock"},
                "actions": [
                    {
                        "action_type": "predict",
                        "method": "POST",
                        "url": f"https://bedrock-runtime.{region}.amazonaws.com/model/amazon.titan-embed-text-v1/invoke",
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
        elif model_type == "3":
            if not model_arn:
                model_arn = input("Enter your custom model ARN: ").strip()

            connector_role_inline_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Action": ["bedrock:InvokeModel"],
                        "Effect": "Allow",
                        "Resource": model_arn,
                    }
                ],
            }
            if not connector_body:
                connector_body = self.input_custom_model_details()
        else:
            print(
                f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
            )
            if not model_arn:
                model_arn = input("Enter your custom model ARN: ").strip()

            connector_role_inline_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Action": ["bedrock:InvokeModel"],
                        "Effect": "Allow",
                        "Resource": {model_arn},
                    }
                ],
            }
            if not connector_body:
                connector_body = self.input_custom_model_details()

        # Create connector
        print("\nCreating Bedrock connector...")
        connector_id, connector_role_arn = helper.create_connector_with_role(
            connector_role_inline_policy,
            connector_role_name,
            create_connector_role_name,
            connector_body,
            sleep_time_in_seconds=10,
        )

        if connector_id:
            print(
                f"{Fore.GREEN}\nSuccessfully created Bedrock connector with ID: {connector_id}{Style.RESET_ALL}"
            )
            connector_output = helper.get_connector(connector_id)
            save_config_method(
                connector_id,
                connector_output,
                connector_role_name,
                None,
                connector_role_arn,
            )
            return True
        else:
            print(f"{Fore.RED}Failed to create Bedrock connector.{Style.RESET_ALL}")
            return False
