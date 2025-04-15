# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

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

    def _get_connector_body(self, model_type, region, endpoint_url):
        """
        Get the connectory body
        """
        connector_configs = {
            "1": {
                "name": "Amazon SageMaker: DeepSeek R1 model",
                "description": "The connector to SageMaker for DeepSeek R1 model",
                "request_body": '{ "inputs": "${parameters.inputs}", "parameters": {"do_sample": ${parameters.do_sample}, "top_p": ${parameters.top_p}, "temperature": ${parameters.temperature}, "max_new_tokens": ${parameters.max_new_tokens}} }',
                "post_process_function": "\n      if (params.result == null || params.result.length == 0) {\n        throw new Exception('No response available');\n      }\n      \n      def completion = params.result[0].generated_text;\n      return '{' +\n               '\"name\": \"response\",'+\n               '\"dataAsMap\": {' +\n                  '\"completion\":\"' + escape(completion) + '\"}' +\n             '}';\n    ",
                "parameters": {
                    "do_sample": "true",
                    "top_p": 0.9,
                    "temperature": 0.7,
                    "max_new_tokens": 512,
                },
            },
            "2": {
                "name": "Amazon SageMaker: Embedding model",
                "description": "The connector to SageMaker for embedding model",
                "request_body": "${parameters.input}",
                "pre_process_function": "connector.pre_process.default.embedding",
                "post_process_function": "connector.post_process.default.embedding",
                "parameters": {},
            },
            "3": "Custom model",
        }

        # Handle custom model or invalid choice
        if (
            model_type not in connector_configs
            or connector_configs[model_type] == "Custom model"
        ):
            if model_type not in connector_configs:
                print(
                    f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
                )
            return self.input_custom_model_details()

        config = connector_configs[model_type]

        # Base parameters that all connectors need
        base_parameters = {
            "region": region,
            "service_name": "sagemaker",
        }

        # Merge with model-specific parameters if any
        parameters = {**base_parameters, **config.get("parameters", {})}

        # Return the connector body
        return {
            "name": config["name"],
            "description": config["description"],
            "version": "1.0",
            "protocol": "aws_sigv4",
            "parameters": parameters,
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "headers": {"content-type": "application/json"},
                    "url": endpoint_url,
                    "request_body": config["request_body"],
                    **(
                        {"pre_process_function": config["pre_process_function"]}
                        if "pre_process_function" in config
                        else {}
                    ),
                    **(
                        {"post_process_function": config["post_process_function"]}
                        if "post_process_function" in config
                        else {}
                    ),
                }
            ],
        }

    def create_connector(
        self,
        helper,
        save_config_method,
        connector_role_prefix=None,
        region=None,
        model_name=None,
        endpoint_arn=None,
        endpoint_url=None,
        connector_body=None,
    ):
        """
        Create SageMaker connector.
        """
        # Set trusted connector endpoints for SageMaker
        trusted_endpoint = (
            "^https://runtime\\.sagemaker\\..*[a-z0-9-]\\.amazonaws\\.com/.*$"
        )
        self.set_trusted_endpoint(helper, trusted_endpoint)

        # Prompt to choose model
        model_type = self.get_model_details(
            "Amazon SageMaker", self.AMAZON_OPENSEARCH_SERVICE, model_name
        )

        if not endpoint_arn:
            endpoint_arn = input(
                "Enter your SageMaker inference endpoint ARN: "
            ).strip()

        # Create connector role
        connector_role_name, create_connector_role_name = self.create_connector_role(
            connector_role_prefix, "sagemaker"
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

        # Prompt for endpoint URL and region for non-custom model
        if model_type == "1" or model_type == "2":
            region = (
                region
                or input(
                    f"Enter your SageMaker region [{self.opensearch_domain_region}]: "
                ).strip()
                or self.opensearch_domain_region
            )
            endpoint_url = (
                endpoint_url
                or input("Enter your SageMaker inference endpoint URL: ").strip()
            )

        # Get connector body
        connector_body = connector_body or self._get_connector_body(
            model_type, region, endpoint_url
        )

        # Create connector
        print("\nCreating SageMaker connector...")
        connector_id, connector_role_arn, _ = helper.create_connector_with_role(
            connector_role_inline_policy,
            connector_role_name,
            create_connector_role_name,
            connector_body,
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
                connector_role_arn,
            )
            return True
        else:
            print(f"{Fore.RED}Failed to create SageMaker connector.{Style.RESET_ALL}")
            return False
