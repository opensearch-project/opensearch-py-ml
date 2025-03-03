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
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests.auth import HTTPBasicAuth
from requests_aws4auth import AWS4Auth

from opensearch_py_ml.ml_commons.IAMRoleHelper import IAMRoleHelper
from opensearch_py_ml.ml_commons.ml_commons_client import MLCommonClient
from opensearch_py_ml.ml_commons.model_access_control import ModelAccessControl
from opensearch_py_ml.ml_commons.model_connector import Connector
from opensearch_py_ml.ml_commons.SecretsHelper import SecretHelper


class AIConnectorHelper:
    """
    Helper class for managing AI connectors and models in OpenSearch.
    """

    def __init__(
        self,
        opensearch_domain_region,
        opensearch_domain_name,
        opensearch_domain_username,
        opensearch_domain_password,
        aws_user_name,
        aws_role_name,
        opensearch_domain_url,
    ):
        """
        Initialize the AIConnectorHelper with necessary AWS and OpenSearch configurations.
        """
        self.opensearch_domain_region = opensearch_domain_region
        self.opensearch_domain_name = opensearch_domain_name
        self.opensearch_domain_username = opensearch_domain_username
        self.opensearch_domain_password = opensearch_domain_password
        self.aws_user_name = aws_user_name
        self.aws_role_name = aws_role_name
        self.opensearch_domain_url = opensearch_domain_url

        # Retrieve OpenSearch domain information
        domain_endpoint, domain_arn = self.get_opensearch_domain_info(
            self.opensearch_domain_region, self.opensearch_domain_name
        )
        if domain_arn:
            self.opensearch_domain_arn = domain_arn
        else:
            print("Warning: Could not retrieve OpenSearch domain ARN.")
            self.opensearch_domain_arn = None

        # Parse the OpenSearch domain URL to extract host and port
        parsed_url = urlparse(self.opensearch_domain_url)
        host = parsed_url.hostname
        port = parsed_url.port or (443 if parsed_url.scheme == "https" else 9200)

        # Initialize OpenSearch client
        self.opensearch_client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=(
                self.opensearch_domain_username,
                self.opensearch_domain_password,
            ),
            use_ssl=(parsed_url.scheme == "https"),
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )

        # Initialize ModelAccessControl for managing model groups
        self.model_access_control = ModelAccessControl(self.opensearch_client)

        # Initialize helpers for IAM roles and secrets management
        self.iam_helper = IAMRoleHelper(
            opensearch_domain_region=self.opensearch_domain_region,
            opensearch_domain_url=self.opensearch_domain_url,
            opensearch_domain_username=self.opensearch_domain_username,
            opensearch_domain_password=self.opensearch_domain_password,
            aws_user_name=self.aws_user_name,
            aws_role_name=self.aws_role_name,
            opensearch_domain_arn=self.opensearch_domain_arn,
        )

        self.secret_helper = SecretHelper(self.opensearch_domain_region)

        # Initialize MLCommonClient for reuse of get_task_info
        self.ml_commons_client = MLCommonClient(self.opensearch_client)

    @staticmethod
    def get_opensearch_domain_info(region, domain_name):
        """
        Retrieve the OpenSearch domain endpoint and ARN based on the domain name and region.
        """
        try:
            opensearch_client = boto3.client("opensearch", region_name=region)
            response = opensearch_client.describe_domain(DomainName=domain_name)
            domain_status = response["DomainStatus"]
            domain_endpoint = domain_status.get("Endpoint")
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
            self.opensearch_domain_region,
            "es",
            session_token=temp_credentials["SessionToken"],
        )
        return awsauth

    def create_connector(self, create_connector_role_name, payload):
        """
        Create a connector in OpenSearch using the specified role and payload.
        Reusing create_standalone_connector from Connector class.
        """
        # Assume role and create a temporary authenticated OS client
        create_connector_role_arn = self.iam_helper.get_role_arn(
            create_connector_role_name
        )
        temp_credentials = self.iam_helper.assume_role(create_connector_role_arn)
        temp_awsauth = AWS4Auth(
            temp_credentials["AccessKeyId"],
            temp_credentials["SecretAccessKey"],
            self.opensearch_domain_region,
            "es",
            session_token=temp_credentials["SessionToken"],
        )

        parsed_url = urlparse(self.opensearch_domain_url)
        host = parsed_url.hostname
        port = parsed_url.port or (443 if parsed_url.scheme == "https" else 9200)

        temp_os_client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=temp_awsauth,
            use_ssl=(parsed_url.scheme == "https"),
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )

        temp_connector = Connector(temp_os_client)
        response = temp_connector.create_standalone_connector(payload)

        print(response)
        connector_id = response.get("connector_id")
        return connector_id

    def get_task(self, task_id, create_connector_role_name, wait_until_task_done=False):
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

    def create_model(
        self,
        model_name,
        description,
        connector_id,
        create_connector_role_name,
        deploy=True,
    ):
        """
        Create a new model in OpenSearch and optionally deploy it.
        """
        try:
            # Use ModelAccessControl methods directly without wrapping
            model_group_id = self.model_access_control.get_model_group_id_by_name(
                model_name
            )
            if not model_group_id:
                self.model_access_control.register_model_group(
                    name=model_name, description=description
                )
                model_group_id = self.model_access_control.get_model_group_id_by_name(
                    model_name
                )
                if not model_group_id:
                    raise Exception("Failed to create model group.")

            payload = {
                "name": model_name,
                "function_name": "remote",
                "description": description,
                "model_group_id": model_group_id,
                "connector_id": connector_id,
            }
            headers = {"Content-Type": "application/json"}
            deploy_str = str(deploy).lower()

            awsauth = self.get_ml_auth(create_connector_role_name)

            response = requests.post(
                f"{self.opensearch_domain_url}/_plugins/_ml/models/_register?deploy={deploy_str}",
                auth=awsauth,
                json=payload,
                headers=headers,
            )

            print("Create Model Response:", response.text)
            response_data = json.loads(response.text)

            if "model_id" in response_data:
                return response_data["model_id"]
            elif "task_id" in response_data:
                # Handle asynchronous task by leveraging wait_until_task_done
                task_response = self.get_task(
                    response_data["task_id"],
                    create_connector_role_name,
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
                raise Exception(f"Error creating model: {response_data['error']}")
            else:
                raise KeyError(
                    f"The response does not contain 'model_id' or 'task_id'. Response content: {response_data}"
                )
        except Exception as e:
            print(f"Error in create_model: {e}")
            raise

    def deploy_model(self, model_id):
        """
        Deploy a specified model in OpenSearch.
        """
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f"{self.opensearch_domain_url}/_plugins/_ml/models/{model_id}/_deploy",
            auth=HTTPBasicAuth(
                self.opensearch_domain_username, self.opensearch_domain_password
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
            f"{self.opensearch_domain_url}/_plugins/_ml/models/{model_id}/_predict",
            auth=HTTPBasicAuth(
                self.opensearch_domain_username, self.opensearch_domain_password
            ),
            json=payload,
            headers=headers,
        )
        print("Predict Response:", response.text)
        return response

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
        print("Step1: Create Secret")
        if not self.secret_helper.secret_exists(secret_name):
            secret_arn = self.secret_helper.create_secret(secret_name, secret_value)
        else:
            print("Secret exists, skipping creation.")
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

        print("Step2: Create IAM role configured in connector")
        if not self.iam_helper.role_exists(connector_role_name):
            connector_role_arn = self.iam_helper.create_iam_role(
                connector_role_name, trust_policy, inline_policy
            )
        else:
            print("Role exists, skipping creation.")
            connector_role_arn = self.iam_helper.get_role_arn(connector_role_name)
        print("----------")

        # Step 3: Configure IAM role in OpenSearch
        # 3.1 Create IAM role for signing create connector request
        user_arn = self.iam_helper.get_user_arn(self.aws_user_name)
        role_arn = self.iam_helper.get_role_arn(self.aws_role_name)
        statements = []
        if user_arn:
            statements.append(
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": user_arn},
                    "Action": "sts:AssumeRole",
                }
            )
        if role_arn:
            statements.append(
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": role_arn},
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
            print("Role exists, skipping creation.")
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
        time.sleep(sleep_time_in_seconds)
        payload = create_connector_input
        payload["credential"] = {"secretArn": secret_arn, "roleArn": connector_role_arn}
        connector_id = self.create_connector(create_connector_role_name, payload)
        print("----------")
        return connector_id

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

        print("Step1: Create IAM role configured in connector")
        if not self.iam_helper.role_exists(connector_role_name):
            connector_role_arn = self.iam_helper.create_iam_role(
                connector_role_name, trust_policy, connector_role_inline_policy
            )
        else:
            print("Role exists, skipping creation.")
            connector_role_arn = self.iam_helper.get_role_arn(connector_role_name)
        print("----------")

        # Step 2: Configure IAM role in OpenSearch
        # 2.1 Create IAM role for signing create connector request
        user_arn = self.iam_helper.get_user_arn(self.aws_user_name)
        role_arn = self.iam_helper.get_role_arn(self.aws_role_name)
        statements = []
        if user_arn:
            statements.append(
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": user_arn},
                    "Action": "sts:AssumeRole",
                }
            )
        if role_arn:
            statements.append(
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": role_arn},
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
            print("Role exists, skipping creation.")
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
        time.sleep(sleep_time_in_seconds)
        payload = create_connector_input
        payload["credential"] = {"roleArn": connector_role_arn}
        connector_id = self.create_connector(create_connector_role_name, payload)
        print("----------")
        return connector_id
