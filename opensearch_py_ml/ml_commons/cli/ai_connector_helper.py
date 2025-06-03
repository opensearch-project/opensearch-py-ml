# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import logging
import time
import warnings
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import boto3
import urllib3
from colorama import Fore, Style
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

from opensearch_py_ml.ml_commons.cli.aws_config import AWSConfig
from opensearch_py_ml.ml_commons.cli.iam_role_helper import IAMRoleHelper
from opensearch_py_ml.ml_commons.cli.opensearch_domain_config import (
    OpenSearchDomainConfig,
)
from opensearch_py_ml.ml_commons.cli.secret_helper import SecretHelper

# Disable warnings when verify_certs=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="opensearchpy")

# Configure the logger for this module
logger = logging.getLogger(__name__)


class AIConnectorHelper:
    """
    Helper class for managing AI connectors and models in OpenSearch.
    """

    OPEN_SOURCE = "open-source"
    AMAZON_OPENSEARCH_SERVICE = "amazon-opensearch-service"

    def __init__(
        self,
        service_type: str,
        opensearch_config: OpenSearchDomainConfig,
        aws_config: AWSConfig,
        ssl_check_enabled: bool = True,
    ):
        """
        Initialize the AIConnectorHelper with necessary AWS and OpenSearch configurations.

        Args:
            service_type: Service type to connect to. Either 'open-source' or 'amazon-opensearch-service'.
            opensearch_config: Configuration object containing OpenSearch domain settings, including region, domain name, credentials, and endpoint.
            aws_config: Configuration object containins AWS credentials and settings including user name, role name, access key, secret key, and session token.
            ssl_check_enabled (optional): Whether to verify SSL certificates when connecting to OpenSearch. Defaults to True.
        """
        self.service_type = service_type
        self.ssl_check_enabled = ssl_check_enabled
        self.opensearch_config = opensearch_config
        self.aws_config = aws_config

        if self.service_type == self.OPEN_SOURCE:
            domain_endpoint = self.opensearch_config.opensearch_domain_endpoint
            domain_arn = None
        else:
            # Get domain info for AOS
            domain_endpoint, domain_arn = self.get_opensearch_domain_info(
                self.opensearch_config.opensearch_domain_region,
                self.opensearch_config.opensearch_domain_name,
                self.aws_config.aws_access_key,
                self.aws_config.aws_secret_access_key,
                self.aws_config.aws_session_token,
            )
        self.opensearch_domain_arn = domain_arn

        # Parse the OpenSearch domain URL
        parsed_url = urlparse(self.opensearch_config.opensearch_domain_endpoint)

        # Initialize OpenSearch client
        self.opensearch_client = OpenSearch(
            hosts=[self.opensearch_config.opensearch_domain_endpoint],
            http_auth=(
                self.opensearch_config.opensearch_domain_username,
                self.opensearch_config.opensearch_domain_password,
            ),
            use_ssl=(parsed_url.scheme == "https"),
            verify_certs=self.ssl_check_enabled,
            connection_class=RequestsHttpConnection,
        )

        # Initialize helpers for IAM roles and secrets management
        if self.service_type == self.OPEN_SOURCE:
            self.iam_helper = None
            self.secret_helper = None
        else:
            self.iam_helper = IAMRoleHelper(
                opensearch_config=self.opensearch_config, aws_config=self.aws_config
            )

            self.secret_helper = SecretHelper(
                opensearch_config=self.opensearch_config, aws_config=self.aws_config
            )

    @staticmethod
    def get_opensearch_domain_info(
        region: str,
        domain_name: str,
        aws_access_key: str,
        aws_secret_access_key: str,
        aws_session_token: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Retrieve the OpenSearch domain endpoint and ARN based on the domain name and region.

        Args:
            region: AWS region where the OpenSearch domain is located (e.g., 'us-west-2').
            domain_name: OpenSearch domain name to retrieve information for.
            aws_access_key: AWS access key.
            aws_secret_access_key: AWS secret access key.
            aws_session_token: AWS session token.

        Returns:
            Tuple[Optional[str], Optional[str]]: A tuple containing:
                - domain_endpoint: The endpoint URL for the OpenSearch domain.
                  (None if retrieval fails)
                - domain_arn: The ARN (Amazon Resource Name) of the OpenSearch domain.
                  (None if retrieval fails)
        """
        try:
            # Get current credentials
            session = boto3.Session(
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
            )
            # Check if credentials are valid
            credentials = session.get_credentials()
            if not credentials:
                logger.error(f"{Fore.RED}No valid credentials found.{Style.RESET_ALL}")
                return None, None

            # Get frozen credentials
            frozen_credentials = credentials.get_frozen_credentials()

            # Create client with explicit credentials
            opensearch_client = boto3.client(
                "opensearch",
                region_name=region,
                aws_access_key_id=frozen_credentials.access_key,
                aws_secret_access_key=frozen_credentials.secret_key,
                aws_session_token=frozen_credentials.token,
            )

            response = opensearch_client.describe_domain(DomainName=domain_name)
            domain_status = response["DomainStatus"]
            domain_endpoint = (
                domain_status.get("Endpoint") or domain_status["Endpoints"]["vpc"]
            )
            domain_arn = domain_status["ARN"]
            return domain_endpoint, domain_arn
        except Exception as e:
            logger.error(f"Error retrieving OpenSearch domain info: {e}")
            return None, None

    def get_ml_auth(self, create_connector_role_name: str) -> AWS4Auth:
        """
        Obtain AWS4Auth credentials for ML API calls using the specified IAM role.

        Args:
            create_connector_role_name: Name of the IAM role.

        Returns:
            AWS4Auth: Authentication object containing temporary credentials.
        """
        # Get role ARN
        create_connector_role_arn = self.iam_helper.get_role_arn(
            create_connector_role_name
        )
        if not create_connector_role_arn:
            raise Exception(f"IAM role '{create_connector_role_name}' not found.")

        # Obtain AWS4Auth with temporary credentials
        temp_credentials = self.iam_helper.assume_role(create_connector_role_arn)
        awsauth = AWS4Auth(
            temp_credentials["credentials"]["AccessKeyId"],
            temp_credentials["credentials"]["SecretAccessKey"],
            self.opensearch_config.opensearch_domain_region,
            "es",
            session_token=temp_credentials["credentials"]["SessionToken"],
        )
        return awsauth

    def register_model(
        self,
        model_name: str,
        description: str,
        connector_id: str,
        deploy: bool = True,
    ) -> str:
        """
        Register a remote model in OpenSearch using ML APIs in opensearch-py.

        Args:
            model_name: The name of the model to register with.
            description: The description of the model to register with.
            connector_id: The connector ID to register the model with.
            deploy (optional): Whether to deploy the model immediately after registration. Defaults to True.

        Returns:
            str: The model ID of the registered model.

        Raises:
            KeyError: If the response doesn't contain expected fields (model_id or task_id)
                or if model_id is missing from task response
            Exception: If there's an error during model registration
        """
        try:
            body = {
                "name": model_name,
                "function_name": "remote",
                "description": description,
                "connector_id": connector_id,
            }
            headers = {"Content-Type": "application/json"}
            deploy_str = str(deploy).lower()

            response = self.opensearch_client.plugins.ml.register_model(
                body=body,
                params={"deploy": deploy_str},
                headers=headers,
            )

            if "model_id" in response:
                return response["model_id"]
            elif "task_id" in response:
                task_response = self.opensearch_client.plugins.ml.get_task(
                    response["task_id"],
                )
                print("Task Response:", json.dumps(task_response))
                if "model_id" in task_response:
                    return task_response["model_id"]
                else:
                    raise KeyError(
                        f"'model_id' not found in task response: {task_response}"
                    )
            elif "error" in response:
                raise Exception(f"Error registering model: {response['error']}")
            else:
                raise KeyError(
                    f"The response does not contain 'model_id' or 'task_id'. Response content: {response}"
                )
        except Exception as e:
            logger.error(
                f"{Fore.RED}Error registering model: {str(e)}{Style.RESET_ALL}"
            )
            raise

    # TODO: Replace with self.opensearch_client.plugins.ml.predict() once available in opensearch-py
    # Current workaround uses transport.perform_request directly
    def predict(self, model_id: str, body: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        """
        Make a prediction using the specified model and input body.

        Args:
            model_id: The model ID to use for prediction.
            body: The request body for prediction.

        Returns:
            Tuple[str, Optional[str]]: A tuple containing:
                - response_text: The complete response from the model as a JSON string
                - status: The status code from the inference results, or None if not available
        """
        headers = {"Content-Type": "application/json"}
        response = self.opensearch_client.transport.perform_request(
            method="POST",
            url=f"/_plugins/_ml/models/{model_id}/_predict",
            body=body,
            headers=headers,
        )
        response_text = json.dumps(response)
        status = response.get("inference_results", [{}])[0].get("status_code")
        print(f"Predict Response: {response_text[:200]}...{response_text[-21:]}")
        return response_text, status

    # TODO: Replace with self.opensearch_client.plugins.ml.get_connector() once available in opensearch-py
    # Current workaround uses transport.perform_request directly
    def get_connector(self, connector_id: str) -> str:
        """
        Get a connector information from the connector ID.

        Args:
            connector_id: The connector ID to get information for.

        Returns:
            str: The connector information as a JSON string
        """
        headers = {"Content-Type": "application/json"}
        response = self.opensearch_client.transport.perform_request(
            method="GET",
            url=f"/_plugins/_ml/connectors/{connector_id}",
            headers=headers,
        )
        return json.dumps(response)

    def create_connector(
        self, create_connector_role_name: str, body: Dict[str, Any]
    ) -> str:
        """
        Create a connector in OpenSearch using the specified role and body.

        Args:
            create_connector_role_name: Name of the IAM role.
            body: The connector configuration as a dictionary.

        Returns:
            str: The ID of the created connector.

        Raises:
            ValueError: If body is None or not a dictionary
        """
        # Verify connector body is not empty and in dict format
        if body is None:
            raise ValueError("A 'body' parameter must be provided as a dictionary.")
        if not isinstance(body, dict):
            raise ValueError("'body' needs to be a dictionary.")

        # Store original auth
        original_auth = None

        # For AOS, temporarily modify client auth
        if self.service_type == self.AMAZON_OPENSEARCH_SERVICE:
            # Obtain AWS4Auth credentials and initialize OpenSearch client with the credentials
            temp_awsauth = self.get_ml_auth(create_connector_role_name)
            original_auth = (
                self.opensearch_client.transport.connection_pool.connections[
                    0
                ].session.auth
            )
            self.opensearch_client.transport.connection_pool.connections[
                0
            ].session.auth = temp_awsauth
        try:
            # Create connector using the body parameter
            headers = {"Content-Type": "application/json"}
            response = self.opensearch_client.plugins.ml.create_connector(
                body=body,
                headers=headers,
            )
            connector_id = response.get("connector_id")
            return connector_id
        finally:
            # Restore original auth if it was modified
            if original_auth is not None:
                self.opensearch_client.http_auth = original_auth

    def _create_iam_role(
        self, step_number: int, connector_role_name: str, inline_policy: Dict[str, Any]
    ) -> str:
        """
        Create an IAM role in OpenSearch using the specified inline policy.
        """
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "es.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        print(f"Step {step_number}: Create IAM role configured in connector")
        if not self.iam_helper.role_exists(connector_role_name):
            connector_role_arn = self.iam_helper.create_iam_role(
                connector_role_name, trust_policy, inline_policy
            )
        else:
            print(f"{connector_role_name} role exists, skipping creation.")
            connector_role_arn = self.iam_helper.get_role_arn(connector_role_name)
        print("----------")
        return connector_role_arn

    def _configure_iam_role(
        self, step_number: int, connector_role_arn: str, create_connector_role_name: str
    ) -> str:
        """
        Configure an IAM role in OpenSearch.
        """
        statements = []
        if self.aws_config.aws_user_name:
            statements.append(
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": self.aws_config.aws_user_name},
                    "Action": "sts:AssumeRole",
                }
            )
        if self.aws_config.aws_role_name:
            statements.append(
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": self.aws_config.aws_role_name},
                    "Action": "sts:AssumeRole",
                }
            )
        trust_policy = {"Version": "2012-10-17", "Statement": statements}

        inline_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "iam:PassRole",
                    "Resource": connector_role_arn,
                },
                {
                    "Effect": "Allow",
                    "Action": "es:ESHttpPost",
                    "Resource": self.opensearch_domain_arn,
                },
            ],
        }

        print(f"Step {step_number}: Configure IAM role in OpenSearch")
        print(
            f"Step {step_number}.1: Create IAM role for Signing create connector request"
        )
        if not self.iam_helper.role_exists(create_connector_role_name):
            create_connector_role_arn = self.iam_helper.create_iam_role(
                create_connector_role_name, trust_policy, inline_policy
            )
        else:
            print(f"{create_connector_role_name} role exists, skipping creation.")
            create_connector_role_arn = self.iam_helper.get_role_arn(
                create_connector_role_name
            )
        print("----------")
        return create_connector_role_arn

    def _map_iam_role(
        self,
        step_number: int,
        create_connector_role_arn: str,
        create_connector_role_name: str,
    ) -> None:
        """
        Map IAM role in OpenSearch.
        """
        print(
            f"Step {step_number}.2: Map IAM role {create_connector_role_name} to OpenSearch permission role"
        )
        self.iam_helper.map_iam_role_to_backend_role(create_connector_role_arn)
        print("----------")

    def _create_connector_with_credentials(
        self,
        step_number: int,
        create_connector_input: Dict[str, Any],
        create_connector_role_name: str,
        connector_role_arn: str,
        sleep_time_in_seconds: int,
        secret_arn: Optional[str] = None,
    ) -> Tuple[str, str, Optional[str]]:
        """
        Create connector in OpenSearch with either role ARN only or both secret and role ARN.
        """
        print(f"Step {step_number}: Create connector in OpenSearch")
        print("Waiting for resources to be ready...")
        for remaining in range(sleep_time_in_seconds, 0, -1):
            print(f"\rTime remaining: {remaining} seconds...", end="", flush=True)
            time.sleep(1)
        print("\nWait completed, creating connector...")
        print("Connector role arn: ", connector_role_arn)
        body = create_connector_input
        if secret_arn:
            body["credential"] = {
                "secretArn": secret_arn,
                "roleArn": connector_role_arn,
            }
        else:
            body["credential"] = {"roleArn": connector_role_arn}
        connector_id = self.create_connector(create_connector_role_name, body)
        print("----------")
        return connector_id, connector_role_arn, secret_arn

    def create_connector_with_secret(
        self,
        secret_name: str,
        secret_value: Dict[str, Any],
        connector_role_name: str,
        create_connector_role_name: str,
        create_connector_input: Dict[str, Any],
        sleep_time_in_seconds: int = 10,
    ) -> Tuple[str, str, str]:
        """
        Create a connector in OpenSearch using a secret for credentials.

        Args:
            secret_name: Name for the secret to be created in AWS Secrets Manager.
            secret_value: The secret value to be stored, containing necessary credentials.
            connector_role_name: Name for the IAM role that the connector will use.
            create_connector_role_name: Name for the IAM role used to create the connector.
            create_connector_input: The configuration for the connector.
            sleep_time_in_seconds (optional): Number of seconds to wait before creating the connector. Defaults to 10 seconds.

        Returns:
            Tuple[str, str, str]: A tuple containing:
                - connector_id: The ID of the created connector.
                - connector_role_arn: The ARN of the role used by the connector.
                - secret_arn: The ARN of the created secret.
        """
        # Step 1: Create Secret
        print("Step 1: Create Secret")
        if not self.secret_helper.secret_exists(secret_name):
            secret_arn = self.secret_helper.create_secret(secret_name, secret_value)
        else:
            print(f"{secret_name} secret exists, skipping creation.")
            secret_arn = self.secret_helper.get_secret_arn(secret_name)
        print("----------")

        # Step 2: Create IAM role configured in connector
        inline_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": [
                        "secretsmanager:GetSecretValue",
                        "secretsmanager:DescribeSecret",
                    ],
                    "Effect": "Allow",
                    "Resource": secret_arn,
                }
            ],
        }
        connector_role_arn = self._create_iam_role(
            "2", connector_role_name, inline_policy
        )

        # Step 3: Configure IAM role in OpenSearch
        # 3.1 Create IAM role for signing create connector request
        create_connector_role_arn = self._configure_iam_role(
            "3", connector_role_arn, create_connector_role_name
        )

        # 3.2 Map IAM role to backend role in OpenSearch
        self._map_iam_role("3", create_connector_role_arn, create_connector_role_name)

        # Step 4: Create connector
        return self._create_connector_with_credentials(
            "4",
            create_connector_input,
            create_connector_role_name,
            connector_role_arn,
            sleep_time_in_seconds,
            secret_arn,
        )

    def create_connector_with_role(
        self,
        connector_role_inline_policy: Dict[str, Any],
        connector_role_name: str,
        create_connector_role_name: str,
        create_connector_input: Dict[str, Any],
        sleep_time_in_seconds: int = 10,
    ) -> Tuple[str, str, None]:
        """
        Create a connector in OpenSearch using an IAM role for credentials.

        Args:
            connector_role_inline_policy:
            connector_role_name: Name for the IAM role that the connector will use.
            create_connector_role_name: Name for the IAM role used to create the connector.
            create_connector_input: The configuration for the connector.
            sleep_time_in_seconds (optional): Number of seconds to wait before creating the connector. Defaults to 10 seconds.

        Returns:
            Tuple[str, str, None]: A tuple containing:
                - connector_id (str): The ID of the created connector
                - connector_role_arn (str): The ARN of the role used by the connector
                - None: Placeholder for secret_arn
        """
        # Step 1: Create IAM role configured in connector
        connector_role_arn = self._create_iam_role(
            "1", connector_role_name, connector_role_inline_policy
        )

        # Step 2: Configure IAM role in OpenSearch
        # 2.1 Create IAM role for signing create connector request
        create_connector_role_arn = self._configure_iam_role(
            "2", connector_role_arn, create_connector_role_name
        )

        # 2.2 Map IAM role to backend role in OpenSearch
        self._map_iam_role("2", create_connector_role_arn, create_connector_role_name)

        # Step 3: Create connector
        return self._create_connector_with_credentials(
            "3",
            create_connector_input,
            create_connector_role_name,
            connector_role_arn,
            sleep_time_in_seconds,
        )
