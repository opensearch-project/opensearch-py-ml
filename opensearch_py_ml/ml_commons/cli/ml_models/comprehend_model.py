# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.


from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.model_base import ModelBase
from opensearch_py_ml.ml_commons.cli.ml_setup import Setup


class ComprehendModel(ModelBase):
    def __init__(self, opensearch_domain_region, service_type):
        """
        Initializes the Comprehend model with necessary configurations
        """
        self.opensearch_domain_region = opensearch_domain_region
        self.service_type = service_type

    def _get_connector_body(self, model_type, region):
        """
        Get the connectory body
        """
        connector_configs = {
            "1": {
                "name": "Amazon Comprehend",
                "description": "The connector for Amazon Comprehend",
                "request_body": '{ "Text": "${parameters.Text}"}',
                "parameters": {
                    "api_version": "20171127",
                    "api_name": "DetectDominantLanguage",
                    "response_filter": "$",
                },
            },
            "2": "Custom model",
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
            "service_name": "comprehend",
            "api": "Comprehend_${parameters.api_version}.${parameters.api_name}",
        }

        # Merge with model-specific parameters if any
        parameters = {**base_parameters, **config.get("parameters", {})}

        # Return the connector body
        return {
            "name": config["name"],
            "description": config["description"],
            "version": 1,
            "protocol": "aws_sigv4",
            "parameters": parameters,
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "headers": {
                        "X-Amz-Target": "${parameters.api}",
                        "content-type": "application/x-amz-json-1.1",
                    },
                    "url": "https://${parameters.service_name}.${parameters.region}.amazonaws.com",
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

    def _get_connector_role_policy(self, model_type, connector_body):
        """
        Get the IAM role policy for the connector
        """
        # Get the model arn
        model_action = (
            input("Enter the Comprehend actions you need: ").strip()
            if model_type == "2"
            else f"comprehend:{connector_body['parameters']['api_name']}"
        )

        return {
            "Version": "2012-10-17",
            "Statement": [
                {"Action": [model_action], "Effect": "Allow", "Resource": "*"}
            ],
        }

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
        Create Comprehend connector.
        """
        # Set trusted connector endpoints for Comprehend
        trusted_endpoint = "^https://comprehend\\..*[a-z0-9-]\\.amazonaws\\.com$"
        self.set_trusted_endpoint(helper, trusted_endpoint)

        # Prompt to choose model
        model_type = self.get_model_details(
            "Amazon Comprehend", self.service_type, model_name
        )

        # Prompt for region
        region = (
            region
            or input(
                f"Enter your AWS region [{self.opensearch_domain_region}]: "
            ).strip()
            or self.opensearch_domain_region
        )

        # Get connector body
        connector_body = connector_body or self._get_connector_body(model_type, region)

        if self.service_type == self.AMAZON_OPENSEARCH_SERVICE:
            # Create connector role
            connector_role_name, create_connector_role_name = (
                self.create_connector_role(connector_role_prefix, "comprehend")
            )
            # Get the connector role inline policy
            connector_role_inline_policy = self._get_connector_role_policy(
                model_type, connector_body
            )

            # Create connector
            print("\nCreating Comprehend connector...")
            connector_id, connector_role_arn, _ = helper.create_connector_with_role(
                connector_role_inline_policy,
                connector_role_name,
                create_connector_role_name,
                connector_body,
                sleep_time_in_seconds=10,
            )
        else:
            # Prompt for AWS credentials
            setup = Setup()
            print("\nPlease enter your AWS credentials:")
            connector_body["credential"] = {
                "access_key": setup.get_password_with_asterisks(
                    "Enter your AWS Access Key ID: "
                ),
                "secret_key": setup.get_password_with_asterisks(
                    "Enter your AWS Secret Access Key: "
                ),
                "session_token": setup.get_password_with_asterisks(
                    "Enter your AWS Session Token: "
                ),
            }

            # Create connector
            print("\nCreating Comprehend connector...")
            connector_id = helper.create_connector(
                create_connector_role_name=None,
                body=connector_body,
            )

        if connector_id:
            print(
                f"{Fore.GREEN}\nSuccessfully created Comprehend connector with ID: {connector_id}{Style.RESET_ALL}"
            )
            connector_output = helper.get_connector(connector_id)
            save_config_method(
                connector_id,
                connector_output,
                (
                    connector_role_name
                    if self.service_type == self.AMAZON_OPENSEARCH_SERVICE
                    else None
                ),
                (
                    connector_role_arn
                    if self.service_type == self.AMAZON_OPENSEARCH_SERVICE
                    else None
                ),
            )
            return True
        else:
            print(f"{Fore.RED}Failed to create Comprehend connector.{Style.RESET_ALL}")
            return False
