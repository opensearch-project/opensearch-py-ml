# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import uuid

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.model_base import ModelBase


class SageMakerModel(ModelBase):
    def __init__(
        self,
        opensearch_domain_region,
    ):
        """
        Initializes the SageMaker model with necessary configurations.
        """
        self.opensearch_domain_region = opensearch_domain_region

    def create_sagemaker_connector(
        self,
        helper,
        save_config_method,
        connector_role_prefix=None,
        region=None,
        model_name=None,
        endpoint_arn=None,
        endpoint_url=None,
        connector_payload=None,
    ):
        """
        Create SageMaker connector.
        """
        # Set trusted connector endpoints for Bedrock
        settings_body = {
            "persistent": {
                "plugins.ml_commons.trusted_connector_endpoints_regex": [
                    "^https://runtime\\.sagemaker\\..*[a-z0-9-]\\.amazonaws\\.com/.*$"
                ]
            }
        }
        helper.opensearch_client.cluster.put_settings(body=settings_body)

        # Prompt for necessary inputs
        if model_name == "DeepSeek R1 model":
            model_type = "1"
        elif model_name == "Embedding model":
            model_type = "2"
        elif model_name == "Custom model":
            model_type = "3"
        else:
            print("\nPlease select a model for the connector creation: ")
            print("1. DeepSeek R1 model")
            print("2. Embedding model")
            print("3. Custom model")
            model_type = input("Enter your choice (1-3): ").strip()

        if not connector_role_prefix:
            connector_role_prefix = input("Enter your connector role prefix: ") or None
            if not connector_role_prefix:
                raise ValueError("Connector role prefix cannot be empty.")

        if not endpoint_arn:
            endpoint_arn = input(
                "Enter your SageMaker inference endpoint ARN: "
            ).strip()

        id = str(uuid.uuid1())[:8]
        connector_role_name = f"{connector_role_prefix}_sagemaker_connector_{id}"
        create_connector_role_name = (
            f"{connector_role_prefix}_sagemaker_connector_create_{id}"
        )

        connector_role_inline_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": ["sagemaker:InvokeEndpoint"],
                    "Effect": "Allow",
                    "Resource": endpoint_arn,
                }
            ],
        }

        if model_type == "1":
            if not endpoint_url:
                endpoint_url = input(
                    "Enter your SageMaker inference endpoint URL: "
                ).strip()

            if not region:
                region = (
                    input(
                        f"Enter your SageMaker region [{self.opensearch_domain_region}]: "
                    ).strip()
                    or self.opensearch_domain_region
                )

            connector_payload = {
                "name": "DeepSeek R1 model connector",
                "description": "Connector for my Sagemaker DeepSeek model",
                "version": "1.0",
                "protocol": "aws_sigv4",
                "parameters": {
                    "service_name": "sagemaker",
                    "region": region,
                    "do_sample": "true",
                    "top_p": 0.9,
                    "temperature": 0.7,
                    "max_new_tokens": 512,
                },
                "actions": [
                    {
                        "action_type": "PREDICT",
                        "method": "POST",
                        "url": endpoint_url,
                        "headers": {"content-type": "application/json"},
                        "request_body": '{ "inputs": "${parameters.inputs}", "parameters": {"do_sample": ${parameters.do_sample}, "top_p": ${parameters.top_p}, "temperature": ${parameters.temperature}, "max_new_tokens": ${parameters.max_new_tokens}} }',
                        "post_process_function": "\n      if (params.result == null || params.result.length == 0) {\n        throw new Exception('No response available');\n      }\n      \n      def completion = params.result[0].generated_text;\n      return '{' +\n               '\"name\": \"response\",'+\n               '\"dataAsMap\": {' +\n                  '\"completion\":\"' + escape(completion) + '\"}' +\n             '}';\n    ",
                    }
                ],
            }
        elif model_type == "2":
            if not endpoint_url:
                endpoint_url = input(
                    "Enter your SageMaker inference endpoint URL: "
                ).strip()

            if not region:
                region = (
                    input(
                        f"Enter your SageMaker region [{self.opensearch_domain_region}]: "
                    ).strip()
                    or self.opensearch_domain_region
                )

            connector_payload = {
                "name": "SageMaker Embedding Model Connector",
                "description": "Connector for SageMaker embedding model",
                "version": "1.0",
                "protocol": "aws_sigv4",
                "parameters": {"region": region, "service_name": "sagemaker"},
                "actions": [
                    {
                        "action_type": "predict",
                        "method": "POST",
                        "headers": {"Content-Type": "application/json"},
                        "url": endpoint_url,
                        "request_body": "${parameters.input}",
                        "pre_process_function": "connector.pre_process.default.embedding",
                        "post_process_function": "connector.post_process.default.embedding",
                    }
                ],
            }
        elif model_type == "3":
            if not connector_payload:
                connector_payload = self.input_custom_model_details()
        else:
            print(
                f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
            )
            if not connector_payload:
                connector_payload = self.input_custom_model_details()

        # Create connector
        print("\nCreating SageMaker connector...")
        connector_id, connector_role_arn = helper.create_connector_with_role(
            connector_role_inline_policy,
            connector_role_name,
            create_connector_role_name,
            connector_payload,
            sleep_time_in_seconds=10,
        )

        if connector_id:
            print(
                f"{Fore.GREEN}\nSuccessfully created SageMaker connector with ID: {connector_id}{Style.RESET_ALL}"
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
            print(f"{Fore.RED}Failed to create SageMaker connector.{Style.RESET_ALL}")
            return False
