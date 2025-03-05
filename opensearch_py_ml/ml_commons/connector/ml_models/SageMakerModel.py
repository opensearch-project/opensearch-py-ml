# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.connector.ml_models.model_base import ModelBase


class SageMakerModel(ModelBase):
    def __init__(
        self,
        opensearch_domain_region,
    ):
        """
        Initializes the SageMaker model with necessary configurations.
        """
        self.opensearch_domain_region = opensearch_domain_region

    def create_sagemaker_connector(self, helper, config, save_config_method):
        """
        Create SageMaker connector.
        """
        # Prompt for necessary inputs
        sagemaker_region = (
            input(
                f"Enter your SageMaker region [{self.opensearch_domain_region}]: "
            ).strip()
            or self.opensearch_domain_region
        )
        sagemaker_endpoint_arn = input(
            "Enter your SageMaker inference endpoint ARN: "
        ).strip()
        sagemaker_endpoint_url = input(
            "Enter your SageMaker inference endpoint URL: "
        ).strip()

        connector_role_name = "sagemaker_connector_role"
        create_connector_role_name = "create_sagemaker_connector_role"

        connector_role_inline_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": ["sagemaker:InvokeEndpoint"],
                    "Effect": "Allow",
                    "Resource": sagemaker_endpoint_arn,
                }
            ],
        }

        default_connector_input = {
            "name": "SageMaker Embedding Model Connector",
            "description": "Connector for SageMaker embedding model",
            "version": "1.0",
            "protocol": "aws_sigv4",
            "parameters": {"region": sagemaker_region, "service_name": "sagemaker"},
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"},
                    "url": sagemaker_endpoint_url,
                    "request_body": "${parameters.input}",
                    "pre_process_function": "connector.pre_process.default.embedding",
                    "post_process_function": "connector.post_process.default.embedding",
                }
            ],
        }

        # Get model details from user
        create_connector_input = self.get_custom_model_details(default_connector_input)
        if not create_connector_input:
            return  # Abort if no valid input

        # Create connector
        print("\nCreating SageMaker connector...")
        connector_id = helper.create_connector_with_role(
            connector_role_inline_policy,
            connector_role_name,
            create_connector_role_name,
            create_connector_input,
            sleep_time_in_seconds=10,
        )

        if connector_id:
            print(
                f"{Fore.GREEN}\nSuccessfully created SageMaker connector with ID: {connector_id}{Style.RESET_ALL}"
            )
            # Update config with connector ID if needed
            config["connector_id"] = connector_id
            save_config_method(config)
            return True
        else:
            print(f"{Fore.RED}Failed to create SageMaker connector.{Style.RESET_ALL}")
            return False
