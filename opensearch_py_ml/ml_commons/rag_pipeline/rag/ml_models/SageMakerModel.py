# SageMakerModel.py
import json
from colorama import Fore, Style

class SageMakerModel:
    def __init__(self, aws_region, opensearch_domain_name, opensearch_username, opensearch_password, iam_role_helper):
        """
        Initializes the SageMakerModel with necessary configurations.
        
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

    def register_sagemaker_model(self, helper, config, save_config_method):
        """
        Register a SageMaker embedding model by creating the necessary connector and model in OpenSearch.

        Args:
            helper (AIConnectorHelper): Instance of AIConnectorHelper.
            config (dict): Configuration dictionary.
            save_config_method (function): Method to save the configuration.
        """
        # Prompt for necessary inputs
        sagemaker_endpoint_arn = input("Enter your SageMaker inference endpoint ARN: ").strip()
        sagemaker_endpoint_url = input("Enter your SageMaker inference endpoint URL: ").strip()
        sagemaker_region = input(f"Enter your SageMaker region [{self.aws_region}]: ").strip() or self.aws_region
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
            "name": "SageMaker Embedding Model Connector",
            "description": "Connector for SageMaker embedding model",
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
                        "Content-Type": "application/json"
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
            print(f"{Fore.RED}Failed to create SageMaker connector. Aborting.{Style.RESET_ALL}")
            return

        # Register model
        model_name = "SageMaker Embedding Model"
        description = "SageMaker embedding model for semantic search"
        model_id = helper.create_model(model_name, description, connector_id, create_connector_role_name)

        if not model_id:
            print(f"{Fore.RED}Failed to create SageMaker model. Aborting.{Style.RESET_ALL}")
            return

        # Save model_id to config
        config['embedding_model_id'] = model_id
        save_config_method(config)
        print(f"{Fore.GREEN}SageMaker model registered successfully. Model ID '{model_id}' saved in configuration.{Style.RESET_ALL}")