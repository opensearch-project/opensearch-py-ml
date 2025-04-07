# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import time
from urllib.parse import urlparse

import boto3
import requests
from colorama import Fore, Style
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests.auth import HTTPBasicAuth
from requests_aws4auth import AWS4Auth

from opensearch_py_ml.ml_commons.cli.aws_config import AWSConfig
from opensearch_py_ml.ml_commons.cli.opensearch_domain_config import (
    OpenSearchDomainConfig,
)
from opensearch_py_ml.ml_commons.IAMRoleHelper import IAMRoleHelper
from opensearch_py_ml.ml_commons.ml_commons_client import MLCommonClient
from opensearch_py_ml.ml_commons.model_access_control import ModelAccessControl
from opensearch_py_ml.ml_commons.model_connector import Connector
from opensearch_py_ml.ml_commons.SecretHelper import SecretHelper


class AIConnectorHelper:
    """
    Helper class for managing AI connectors and models in OpenSearch.
    """

    def __init__(
        self,
        service_type,
        opensearch_config: OpenSearchDomainConfig,
        aws_config: AWSConfig,
    ):
        """
        Initialize the AIConnectorHelper with necessary AWS and OpenSearch configurations.
        """
        self.service_type = service_type
        self.opensearch_config = opensearch_config
        self.aws_config = aws_config

        if self.service_type == "open-source":
            domain_endpoint = self.opensearch_config.opensearch_domain_endpoint
            domain_arn = None
        else:
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
        port = parsed_url.port or (443 if parsed_url.scheme == "https" else 9200)

        # Initialize OpenSearch client
        self.opensearch_client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=(
                self.opensearch_config.opensearch_domain_username,
                self.opensearch_config.opensearch_domain_password,
            ),
            use_ssl=(parsed_url.scheme == "https"),
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )

        # Initialize ModelAccessControl for managing model groups
        self.model_access_control = ModelAccessControl(self.opensearch_client)

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

        # Initialize MLCommonClient for reuse of get_task_info
        self.ml_commons_client = MLCommonClient(self.opensearch_client)

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
            credentials = session.get_credentials()
            if not credentials:
                print(f"{Fore.RED}No valid credentials found.{Style.RESET_ALL}")
                return None, None

            # Get frozen credentials (this converts to accessible format)
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
            print(f"Error retrieving OpenSearch domain info: {e}")
            return None, None

    def get_ml_auth(self, create_connector_role_name):
        """
        Obtain AWS4Auth credentials for ML API calls using the specified IAM role.
        """
        create_connector_role_arn = self.iam_helper.get_role_arn(
            create_connector_role_name
        )
        if not create_connector_role_arn:
            raise Exception(f"IAM role '{create_connector_role_name}' not found.")

        temp_credentials = self.iam_helper.assume_role(create_connector_role_arn)
        awsauth = AWS4Auth(
            temp_credentials["AccessKeyId"],
            temp_credentials["SecretAccessKey"],
            self.opensearch_config.opensearch_domain_region,
            "es",
            session_token=temp_credentials["SessionToken"],
        )
        return awsauth

    def create_connector(self, create_connector_role_name, payload):
        """
        Create a connector in OpenSearch using the specified role and payload.
        Reusing create_standalone_connector from Connector class.
        """
        if self.service_type == "amazon-opensearch-service":
            create_connector_role_arn = self.iam_helper.get_role_arn(
                create_connector_role_name
            )
            temp_credentials = self.iam_helper.assume_role(create_connector_role_arn)
            temp_awsauth = AWS4Auth(
                temp_credentials["credentials"]["AccessKeyId"],
                temp_credentials["credentials"]["SecretAccessKey"],
                self.opensearch_config.opensearch_domain_region,
                "es",
                session_token=temp_credentials["credentials"]["SessionToken"],
            )

            parsed_url = urlparse(self.opensearch_config.opensearch_domain_endpoint)
            host = parsed_url.hostname
            port = parsed_url.port or (443 if parsed_url.scheme == "https" else 9200)

            temp_os_client = OpenSearch(
                hosts=[{"host": host, "port": port}],
                http_auth=temp_awsauth,
                use_ssl=(parsed_url.scheme == "https"),
                verify_certs=True,
                connection_class=RequestsHttpConnection,
            )
        else:
            parsed_url = urlparse(self.opensearch_config.opensearch_domain_endpoint)
            host = parsed_url.hostname
            port = parsed_url.port or (443 if parsed_url.scheme == "https" else 9200)

            temp_os_client = OpenSearch(
                hosts=[{"host": host, "port": port}],
                http_auth=(
                    self.opensearch_config.opensearch_domain_username,
                    self.opensearch_config.opensearch_domain_password,
                ),
                use_ssl=(parsed_url.scheme == "https"),
                verify_certs=True,
                connection_class=RequestsHttpConnection,
            )

        temp_connector = Connector(temp_os_client)
        response = temp_connector.create_standalone_connector(payload)
        connector_id = response.get("connector_id")
        return connector_id

    def get_task(self, task_id, wait_until_task_done=False):
        """
        Retrieve the status of a specific task using its ID.
        Reusing the get_task_info method from MLCommonClient and allowing
        optional wait until the task completes.
        """
        try:
            # No need to re-authenticate here; ml_commons_client uses self.opensearch_client
            task_response = self.ml_commons_client.get_task_info(
                task_id, wait_until_task_done
            )
            print("Get Task Response:", json.dumps(task_response))
            return task_response
        except Exception as e:
            print(f"Error in get_task: {e}")
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
            payload = {
                "name": model_name,
                "function_name": "remote",
                "description": description,
                "connector_id": connector_id,
            }
            headers = {"Content-Type": "application/json"}
            deploy_str = str(deploy).lower()

            response = requests.post(
                f"{self.opensearch_config.opensearch_domain_endpoint}/_plugins/_ml/models/_register?deploy={deploy_str}",
                auth=HTTPBasicAuth(
                    self.opensearch_config.opensearch_domain_username,
                    self.opensearch_config.opensearch_domain_password,
                ),
                json=payload,
                headers=headers,
            )

            response_data = json.loads(response.text)

            if "model_id" in response_data:
                return response_data["model_id"]
            elif "task_id" in response_data:
                # Handle asynchronous task by leveraging wait_until_task_done
                task_response = self.get_task(
                    response_data["task_id"],
                    wait_until_task_done=True,
                )
                print("Task Response:", json.dumps(task_response))
                if "model_id" in task_response:
                    return task_response["model_id"]
                else:
                    raise KeyError(
                        f"'model_id' not found in task response: {task_response}"
                    )
            elif "error" in response_data:
                raise Exception(f"Error registering model: {response_data['error']}")
            else:
                raise KeyError(
                    f"The response does not contain 'model_id' or 'task_id'. Response content: {response_data}"
                )
        except Exception as e:
            print(f"{Fore.RED}Error registering model: {str(e)}{Style.RESET_ALL}")
            raise

    def deploy_model(self, model_id):
        """
        Deploy a specified model in OpenSearch.
        """
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f"{self.opensearch_config.opensearch_domain_endpoint}/_plugins/_ml/models/{model_id}/_deploy",
            auth=HTTPBasicAuth(
                self.opensearch_config.opensearch_domain_username,
                self.opensearch_config.opensearch_domain_password,
            ),
            headers=headers,
        )
        print(f"Deploy Model Response: {response.text}")
        return response

    def predict(self, model_id, payload):
        """
        Make a prediction using the specified model and input payload.
        """
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f"{self.opensearch_config.opensearch_domain_endpoint}/_plugins/_ml/models/{model_id}/_predict",
            auth=HTTPBasicAuth(
                self.opensearch_config.opensearch_domain_username,
                self.opensearch_config.opensearch_domain_password,
            ),
            json=payload,
            headers=headers,
        )
        response_json = response.json()
        status = response_json.get("inference_results", [{}])[0].get("status_code")
        print("Predict Response:", response.text[:200] + "..." + response.text[-21:])
        return response.text, status

    def get_connector(self, connector_id):
        """
        Get a connector information from the connector ID.
        """
        headers = {"Content-Type": "application/json"}
        response = requests.get(
            f"{self.opensearch_config.opensearch_domain_endpoint}/_plugins/_ml/connectors/{connector_id}",
            auth=HTTPBasicAuth(
                self.opensearch_config.opensearch_domain_username,
                self.opensearch_config.opensearch_domain_password,
            ),
            headers=headers,
        )
        return response.text

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

        print("Step 2: Create IAM role configured in connector")
        if not self.iam_helper.role_exists(connector_role_name):
            connector_role_arn = self.iam_helper.create_iam_role(
                connector_role_name, trust_policy, inline_policy
            )
        else:
            print(f"{connector_role_name} role exists, skipping creation.")
            connector_role_arn = self.iam_helper.get_role_arn(connector_role_name)
        print("----------")

        # Step 3: Configure IAM role in OpenSearch
        # 3.1 Create IAM role for signing create connector request
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

        print("Step 3: Configure IAM role in OpenSearch")
        print("Step 3.1: Create IAM role for Signing create connector request")
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

        # 3.2 Map IAM role to backend role in OpenSearch
        print(
            f"Step 3.2: Map IAM role {create_connector_role_name} to OpenSearch permission role"
        )
        self.iam_helper.map_iam_role_to_backend_role(create_connector_role_arn)
        print("----------")

        # Step 4: Create connector
        print("Step 4: Create connector in OpenSearch")
        print("Waiting for resources to be ready...")
        for remaining in range(sleep_time_in_seconds, 0, -1):
            print(f"\rTime remaining: {remaining} seconds...", end="", flush=True)
            time.sleep(1)
        print("\nWait completed, creating connector...")
        payload = create_connector_input
        payload["credential"] = {"secretArn": secret_arn, "roleArn": connector_role_arn}
        connector_id = self.create_connector(create_connector_role_name, payload)
        print("----------")
        return connector_id, connector_role_arn

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

        print("Step 1: Create IAM role configured in connector")
        if not self.iam_helper.role_exists(connector_role_name):
            connector_role_arn = self.iam_helper.create_iam_role(
                connector_role_name, trust_policy, connector_role_inline_policy
            )
        else:
            print(f"{connector_role_name} role exists, skipping creation.")
            connector_role_arn = self.iam_helper.get_role_arn(connector_role_name)
        print("----------")

        # Step 2: Configure IAM role in OpenSearch
        # 2.1 Create IAM role for signing create connector request
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

        print("Step 2: Configure IAM role in OpenSearch")
        print("Step 2.1: Create IAM role for Signing create connector request")
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

        # 2.2 Map IAM role to backend role in OpenSearch
        print(
            f"Step 2.2: Map IAM role {create_connector_role_name} to OpenSearch permission role"
        )
        self.iam_helper.map_iam_role_to_backend_role(create_connector_role_arn)
        print("----------")

        # Step 3: Create connector
        print("Step 3: Create connector in OpenSearch")
        print("Waiting for resources to be ready...")
        for remaining in range(sleep_time_in_seconds, 0, -1):
            print(f"\rTime remaining: {remaining} seconds...", end="", flush=True)
            time.sleep(1)
        print("\nWait completed, creating connector...")
        payload = create_connector_input
        print("Connector role arn: ", connector_role_arn)
        payload["credential"] = {"roleArn": connector_role_arn}
        connector_id = self.create_connector(create_connector_role_name, payload)
        print("----------")
        return connector_id, connector_role_arn
