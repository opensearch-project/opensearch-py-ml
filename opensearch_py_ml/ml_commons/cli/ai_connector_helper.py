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
from urllib.parse import urlparse

import boto3
import urllib3
from colorama import Fore, Style
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

from opensearch_py_ml.ml_commons.cli.aws_config import AWSConfig
from opensearch_py_ml.ml_commons.cli.opensearch_domain_config import (
    OpenSearchDomainConfig,
)
from opensearch_py_ml.ml_commons.iam_role_helper import IAMRoleHelper
from opensearch_py_ml.ml_commons.ml_common_utils import TIMEOUT
from opensearch_py_ml.ml_commons.secret_helper import SecretHelper

# Disable warnings when verify_certs=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="opensearchpy")

# Configure the logger for this module
logger = logging.getLogger(__name__)


class AIConnectorHelper:
    """
    Helper class for managing AI connectors and models in OpenSearch.
    """

    def __init__(
        self,
        service_type,
        ssl_check_enabled,
        opensearch_config: OpenSearchDomainConfig,
        aws_config: AWSConfig,
    ):
        """
        Initialize the AIConnectorHelper with necessary AWS and OpenSearch configurations
        """
        self.service_type = service_type
        self.ssl_check_enabled = ssl_check_enabled
        self.opensearch_config = opensearch_config
        self.aws_config = aws_config

        if self.service_type == "open-source":
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

        # Parse the OpenSearch domain URL to extract host and port
        parsed_url = urlparse(self.opensearch_config.opensearch_domain_endpoint)
        host = parsed_url.hostname
        port = parsed_url.port or 9200

        # Initialize OpenSearch client
        self.opensearch_client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=(
                self.opensearch_config.opensearch_domain_username,
                self.opensearch_config.opensearch_domain_password,
            ),
            use_ssl=(parsed_url.scheme == "https"),
            verify_certs=self.ssl_check_enabled,
            connection_class=RequestsHttpConnection,
        )

        # Initialize helpers for IAM roles and secrets management
        if self.service_type == "open-source":
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
        region, domain_name, aws_access_key, aws_secret_access_key, aws_session_token
    ):
        """
        Retrieve the OpenSearch domain endpoint and ARN based on the domain name and region.
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

    def get_ml_auth(self, create_connector_role_name):
        """
        Obtain AWS4Auth credentials for ML API calls using the specified IAM role.
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

    def get_task_info(self, task_id: str, wait_until_task_done: bool = False) -> object:
        """
        Returns information about a task running into opensearch cluster
        Replicating the get_task_info in ml_commons_client.py
        since ML client APIs will be deprecated
        """
        if wait_until_task_done:
            end = time.time() + TIMEOUT  # timeout seconds
            task_flag = False
            while not task_flag and time.time() < end:
                time.sleep(1)
                output = self._get_task_info(task_id)
                if (
                    output["state"] == "COMPLETED"
                    or output["state"] == "FAILED"
                    or output["state"] == "COMPLETED_WITH_ERROR"
                ):
                    task_flag = True
        return self._get_task_info(task_id)

    def _get_task_info(self, task_id: str):
        """
        Perform the get request to get the task status
        Replicating the _get_task_info in ml_commons_client.py
        since ML client APIs will be deprecated
        """
        headers = {"Content-Type": "application/json"}
        response = self.opensearch_client.transport.perform_request(
            method="GET", url=f"/_plugins/_ml/tasks/{task_id}", headers=headers
        )
        return response

    def get_task(self, task_id, wait_until_task_done=False):
        """
        Retrieve the status of a specific task using its ID.
        """
        try:
            response = self.get_task_info(task_id, wait_until_task_done)
            print("Get Task Response:", json.dumps(response))
            return response
        except Exception as e:
            logger.error(f"Error in get_task: {e}")
            raise

    def register_model(
        self,
        model_name,
        description,
        connector_id,
        deploy=True,
    ):
        """
        Register a new model in OpenSearch and optionally deploy it.
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

            response = self.opensearch_client.transport.perform_request(
                method="POST",
                url="/_plugins/_ml/models/_register",
                params={"deploy": deploy_str},
                body=body,
                headers=headers,
            )
            if "model_id" in response:
                return response["model_id"]
            elif "task_id" in response:
                # Handle asynchronous task by leveraging wait_until_task_done
                task_response = self.get_task(
                    response["task_id"],
                    wait_until_task_done=True,
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

    def deploy_model(self, model_id):
        """
        Deploy a specified model in OpenSearch.
        """
        headers = {"Content-Type": "application/json"}
        response = self.opensearch_client.transport.perform_request(
            method="POST",
            url=f"/_plugins/_ml/models/{model_id}/_deploy",
            headers=headers,
        )
        print(f"Deploy Model Response: {response}")
        return response

    def predict(self, model_id, body):
        """
        Make a prediction using the specified model and input body.
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

    def get_connector(self, connector_id):
        """
        Get a connector information from the connector ID.
        """
        headers = {"Content-Type": "application/json"}
        response = self.opensearch_client.transport.perform_request(
            method="GET",
            url=f"/_plugins/_ml/connectors/{connector_id}",
            headers=headers,
        )
        return json.dumps(response)

    def create_connector(self, create_connector_role_name, body):
        """
        Create a connector in OpenSearch using the specified role and body.
        """
        # Verify connector body is not empty and in dict format
        if body is None:
            raise ValueError("A 'body' parameter must be provided as a dictionary.")
        if not isinstance(body, dict):
            raise ValueError("'body' needs to be a dictionary.")

        # Parse the OpenSearch domain URL to extract host and port
        parsed_url = urlparse(self.opensearch_config.opensearch_domain_endpoint)
        host = parsed_url.hostname
        port = parsed_url.port or 9200

        if self.service_type == "amazon-opensearch-service":
            # Obtain AWS4Auth credentials and initialize OpenSearch client with the credentials
            temp_awsauth = self.get_ml_auth(create_connector_role_name)
            temp_os_client = OpenSearch(
                hosts=[{"host": host, "port": port}],
                http_auth=temp_awsauth,
                use_ssl=(parsed_url.scheme == "https"),
                verify_certs=self.ssl_check_enabled,
                connection_class=RequestsHttpConnection,
            )
        else:
            # For open-source, initialize OpenSearch client with domain username and password
            temp_os_client = OpenSearch(
                hosts=[{"host": host, "port": port}],
                http_auth=(
                    self.opensearch_config.opensearch_domain_username,
                    self.opensearch_config.opensearch_domain_password,
                ),
                use_ssl=(parsed_url.scheme == "https"),
                verify_certs=self.ssl_check_enabled,
                connection_class=RequestsHttpConnection,
            )

        # Create connector using the body parameter
        headers = {"Content-Type": "application/json"}
        response = temp_os_client.transport.perform_request(
            method="POST",
            url="/_plugins/_ml/connectors/_create",
            body=body,
            headers=headers,
        )
        connector_id = response.get("connector_id")
        return connector_id

    def _create_iam_role(self, step_number, connector_role_name, inline_policy):
        """
        Create an IAM role in OpenSearch using the specified inline policy
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
        self, step_number, connector_role_arn, create_connector_role_name
    ):
        """
        Configure an IAM role in OpenSearch
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
        self, step_number, create_connector_role_arn, create_connector_role_name
    ):
        """
        Map IAM role in OpenSearch
        """
        print(
            f"Step {step_number}.2: Map IAM role {create_connector_role_name} to OpenSearch permission role"
        )
        self.iam_helper.map_iam_role_to_backend_role(create_connector_role_arn)
        print("----------")

    def _create_connector_with_credentials(
        self,
        step_number,
        create_connector_input,
        create_connector_role_name,
        connector_role_arn,
        sleep_time_in_seconds,
        secret_arn=None,
    ):
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
        return connector_id, connector_role_arn

    def create_connector_with_secret(
        self,
        secret_name,
        secret_value,
        connector_role_name,
        create_connector_role_name,
        create_connector_input,
        sleep_time_in_seconds=10,
    ):
        """
        Create a connector in OpenSearch using a secret for credentials.
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
        connector_role_inline_policy,
        connector_role_name,
        create_connector_role_name,
        create_connector_input,
        sleep_time_in_seconds=10,
    ):
        """
        Create a connector in OpenSearch using an IAM role for credentials.
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
