# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

#  Licensed to Elasticsearch B.V. under one or more contributor
#  license agreements. See the NOTICE file distributed with
#  this work for additional information regarding copyright
#  ownership. Elasticsearch B.V. licenses this file to you under
#  the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

import boto3
import json
import requests
from requests.auth import HTTPBasicAuth
from requests_aws4auth import AWS4Auth
import time
from opensearchpy import OpenSearch, RequestsHttpConnection

from IAMRoleHelper import IAMRoleHelper
from SecretsHelper import SecretHelper
from opensearch_py_ml.ml_commons.model_access_control import ModelAccessControl

class AIConnectorHelper:
    def __init__(self, region, opensearch_domain_name, opensearch_domain_username,
                 opensearch_domain_password, aws_user_name, aws_role_name):
        """
        Initialize the AIConnectorHelper with necessary AWS and OpenSearch configurations.
        """
        self.region = region
        self.opensearch_domain_name = opensearch_domain_name
        self.opensearch_domain_username = opensearch_domain_username
        self.opensearch_domain_password = opensearch_domain_password
        self.aws_user_name = aws_user_name
        self.aws_role_name = aws_role_name

        # Retrieve the OpenSearch domain endpoint and ARN
        domain_endpoint, domain_arn = self.get_opensearch_domain_info(self.region, self.opensearch_domain_name)
        if domain_arn:
            self.opensearch_domain_arn = domain_arn
        else:
            print("Warning: Could not retrieve OpenSearch domain ARN.")
            self.opensearch_domain_arn = None

        if domain_endpoint:
            # Construct the full domain URL
            self.opensearch_domain_url = f'https://{domain_endpoint}'
        else:
            print("Warning: Could not retrieve OpenSearch domain endpoint.")
            self.opensearch_domain_url = None

        # Initialize the OpenSearch client
        self.opensearch_client = OpenSearch(
            hosts=[{'host': domain_endpoint, 'port': 443}],
            http_auth=(self.opensearch_domain_username, self.opensearch_domain_password),
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )

        # Initialize ModelAccessControl
        self.model_access_control = ModelAccessControl(self.opensearch_client)

        # Initialize IAMRoleHelper and SecretHelper
        self.iam_helper = IAMRoleHelper(
            region=self.region,
            opensearch_domain_url=self.opensearch_domain_url,
            opensearch_domain_username=self.opensearch_domain_username,
            opensearch_domain_password=self.opensearch_domain_password,
            aws_user_name=self.aws_user_name,
            aws_role_name=self.aws_role_name,
            opensearch_domain_arn=self.opensearch_domain_arn
        )

        self.secret_helper = SecretHelper(self.region)

    @staticmethod
    def get_opensearch_domain_info(region, domain_name):
        """
        Retrieve the OpenSearch domain endpoint and ARN based on the domain name and region.
        """
        try:
            opensearch_client = boto3.client('opensearch', region_name=region)
            response = opensearch_client.describe_domain(DomainName=domain_name)
            domain_status = response['DomainStatus']
            domain_endpoint = domain_status.get('Endpoint')
            domain_arn = domain_status['ARN']
            return domain_endpoint, domain_arn
        except Exception as e:
            print(f"Error retrieving OpenSearch domain info: {e}")
            return None, None

    def get_ml_auth(self, create_connector_role_name):
        """
        Obtain AWS4Auth credentials for ML API calls using the specified IAM role.
        """
        create_connector_role_arn = self.iam_helper.get_role_arn(create_connector_role_name)
        if not create_connector_role_arn:
            raise Exception(f"IAM role '{create_connector_role_name}' not found.")

        temp_credentials = self.iam_helper.assume_role(create_connector_role_arn)
        awsauth = AWS4Auth(
            temp_credentials["AccessKeyId"],
            temp_credentials["SecretAccessKey"],
            self.region,
            'es',
            session_token=temp_credentials["SessionToken"],
        )
        return awsauth

    def create_connector(self, create_connector_role_name, payload):
        create_connector_role_arn = self.iam_helper.get_role_arn(create_connector_role_name)
        temp_credentials = self.iam_helper.assume_role(create_connector_role_arn)
        awsauth = AWS4Auth(
            temp_credentials["AccessKeyId"],
            temp_credentials["SecretAccessKey"],
            self.region,
            'es',
            session_token=temp_credentials["SessionToken"],
        )

        path = '/_plugins/_ml/connectors/_create'
        url = self.opensearch_domain_url + path

        headers = {"Content-Type": "application/json"}

        r = requests.post(url, auth=awsauth, json=payload, headers=headers)
        print(r.text)
        connector_id = json.loads(r.text)['connector_id']
        return connector_id

    def search_model_group(self, model_group_name, create_connector_role_name):
        """
        Utilize ModelAccessControl to search for a model group by name.
        """
        response = self.model_access_control.search_model_group_by_name(model_group_name, size=1)
        return response

    def create_model_group(self, model_group_name, description, create_connector_role_name):
        """
        Utilize ModelAccessControl to create or retrieve an existing model group.
        """
        model_group_id = self.model_access_control.get_model_group_id_by_name(model_group_name)
        print("Search Model Group Response:", model_group_id)

        if model_group_id:
            return model_group_id

        # Use ModelAccessControl to register model group
        self.model_access_control.register_model_group(name=model_group_name, description=description)
        
        # Retrieve the newly created model group id
        model_group_id = self.model_access_control.get_model_group_id_by_name(model_group_name)
        if model_group_id:
            return model_group_id
        else:
            raise Exception("Failed to create model group.")

    def get_task(self, task_id, create_connector_role_name):
        try:
            awsauth = self.get_ml_auth(create_connector_role_name)
            r = requests.get(
                f'{self.opensearch_domain_url}/_plugins/_ml/tasks/{task_id}',
                auth=awsauth
            )
            print("Get Task Response:", r.text)
            return r
        except Exception as e:
            print(f"Error in get_task: {e}")
            raise

    def create_model(self, model_name, description, connector_id, create_connector_role_name, deploy=True):
        try:
            model_group_id = self.create_model_group(model_name, description, create_connector_role_name)
            payload = {
                "name": model_name,
                "function_name": "remote",
                "description": description,
                "model_group_id": model_group_id,
                "connector_id": connector_id
            }
            headers = {"Content-Type": "application/json"}
            deploy_str = str(deploy).lower()

            awsauth = self.get_ml_auth(create_connector_role_name)

            r = requests.post(
                f'{self.opensearch_domain_url}/_plugins/_ml/models/_register?deploy={deploy_str}',
                auth=awsauth,
                json=payload,
                headers=headers
            )

            print("Create Model Response:", r.text)
            response = json.loads(r.text)

            if 'model_id' in response:
                return response['model_id']
            elif 'task_id' in response:
                # Handle asynchronous task
                time.sleep(2)  # Wait for task to complete
                task_response = self.get_task(response['task_id'], create_connector_role_name)
                print("Task Response:", task_response.text)
                task_result = json.loads(task_response.text)
                if 'model_id' in task_result:
                    return task_result['model_id']
                else:
                    raise KeyError(f"'model_id' not found in task response: {task_result}")
            elif 'error' in response:
                raise Exception(f"Error creating model: {response['error']}")
            else:
                raise KeyError(f"The response does not contain 'model_id' or 'task_id'. Response content: {response}")
        except Exception as e:
            print(f"Error in create_model: {e}")
            raise

    def deploy_model(self, model_id):
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f'{self.opensearch_domain_url}/_plugins/_ml/models/{model_id}/_deploy',
            auth=HTTPBasicAuth(self.opensearch_domain_username, self.opensearch_domain_password),
            headers=headers
        )
        print(f"Deploy Model Response: {response.text}")
        return response

    def predict(self, model_id, payload):
        headers = {"Content-Type": "application/json"}
        r = requests.post(
            f'{self.opensearch_domain_url}/_plugins/_ml/models/{model_id}/_predict',
            auth=HTTPBasicAuth(self.opensearch_domain_username, self.opensearch_domain_password),
            json=payload,
            headers=headers
        )
        print("Predict Response:", r.text)
        return r

    def create_connector_with_secret(self, secret_name, secret_value, connector_role_name, create_connector_role_name,
                                     create_connector_input, sleep_time_in_seconds=10):
        # Step1: Create Secret
        print('Step1: Create Secret')
        if not self.secret_helper.secret_exists(secret_name):
            secret_arn = self.secret_helper.create_secret(secret_name, secret_value)
        else:
            print('Secret exists, skipping creation.')
            secret_arn = self.secret_helper.get_secret_arn(secret_name)
        print('----------')

        # Step2: Create IAM role configured in connector
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "es.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }

        inline_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": [
                        "secretsmanager:GetSecretValue",
                        "secretsmanager:DescribeSecret"
                    ],
                    "Effect": "Allow",
                    "Resource": secret_arn
                }
            ]
        }

        print('Step2: Create IAM role configured in connector')
        if not self.iam_helper.role_exists(connector_role_name):
            connector_role_arn = self.iam_helper.create_iam_role(connector_role_name, trust_policy, inline_policy)
        else:
            print('Role exists, skipping creation.')
            connector_role_arn = self.iam_helper.get_role_arn(connector_role_name)
        print('----------')

        # Step 3: Configure IAM role in OpenSearch
        # 3.1 Create IAM role for Signing create connector request
        user_arn = self.iam_helper.get_user_arn(self.aws_user_name)
        role_arn = self.iam_helper.get_role_arn(self.aws_role_name)
        statements = []
        if user_arn:
            statements.append({
                "Effect": "Allow",
                "Principal": {
                    "AWS": user_arn
                },
                "Action": "sts:AssumeRole"
            })
        if role_arn:
            statements.append({
                "Effect": "Allow",
                "Principal": {
                    "AWS": role_arn
                },
                "Action": "sts:AssumeRole"
            })
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": statements
        }

        inline_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "iam:PassRole",
                    "Resource": connector_role_arn
                },
                {
                    "Effect": "Allow",
                    "Action": "es:ESHttpPost",
                    "Resource": self.opensearch_domain_arn
                }
            ]
        }

        print('Step 3: Configure IAM role in OpenSearch')
        print('Step 3.1: Create IAM role for Signing create connector request')
        if not self.iam_helper.role_exists(create_connector_role_name):
            create_connector_role_arn = self.iam_helper.create_iam_role(create_connector_role_name, trust_policy,
                                                                       inline_policy)
        else:
            print('Role exists, skipping creation.')
            create_connector_role_arn = self.iam_helper.get_role_arn(create_connector_role_name)
        print('----------')

        # 3.2 Map backend role
        print(f'Step 3.2: Map IAM role {create_connector_role_name} to OpenSearch permission role')
        self.iam_helper.map_iam_role_to_backend_role(create_connector_role_arn)
        print('----------')

        # Step 4: Create connector
        print('Step 4: Create connector in OpenSearch')
        # When you create an IAM role, it can take some time for the changes to propagate across AWS systems.
        # During this time, some services might not immediately recognize the new role or its permissions.
        # So we wait for some time before creating connector.
        # If you see such error: ClientError: An error occurred (AccessDenied) when calling the AssumeRole operation
        # you can rerun this function.

        # Wait for some time
        time.sleep(sleep_time_in_seconds)
        payload = create_connector_input
        payload['credential'] = {
            "secretArn": secret_arn,
            "roleArn": connector_role_arn
        }
        connector_id = self.create_connector(create_connector_role_name, payload)
        print('----------')
        return connector_id

    def create_connector_with_role(self, connector_role_inline_policy, connector_role_name, create_connector_role_name,
                                   create_connector_input, sleep_time_in_seconds=10):
        # Step1: Create IAM role configured in connector
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "es.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }

        print('Step1: Create IAM role configured in connector')
        if not self.iam_helper.role_exists(connector_role_name):
            connector_role_arn = self.iam_helper.create_iam_role(connector_role_name, trust_policy,
                                                                 connector_role_inline_policy)
        else:
            print('Role exists, skipping creation.')
            connector_role_arn = self.iam_helper.get_role_arn(connector_role_name)
        print('----------')

        # Step 2: Configure IAM role in OpenSearch
        # 2.1 Create IAM role for Signing create connector request
        user_arn = self.iam_helper.get_user_arn(self.aws_user_name)
        role_arn = self.iam_helper.get_role_arn(self.aws_role_name)
        statements = []
        if user_arn:
            statements.append({
                "Effect": "Allow",
                "Principal": {
                    "AWS": user_arn
                },
                "Action": "sts:AssumeRole"
            })
        if role_arn:
            statements.append({
                "Effect": "Allow",
                "Principal": {
                    "AWS": role_arn
                },
                "Action": "sts:AssumeRole"
            })
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": statements
        }

        inline_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "iam:PassRole",
                    "Resource": connector_role_arn
                },
                {
                    "Effect": "Allow",
                    "Action": "es:ESHttpPost",
                    "Resource": self.opensearch_domain_arn
                }
            ]
        }

        print('Step 2: Configure IAM role in OpenSearch')
        print('Step 2.1: Create IAM role for Signing create connector request')
        if not self.iam_helper.role_exists(create_connector_role_name):
            create_connector_role_arn = self.iam_helper.create_iam_role(create_connector_role_name, trust_policy,
                                                                       inline_policy)
        else:
            print('Role exists, skipping creation.')
            create_connector_role_arn = self.iam_helper.get_role_arn(create_connector_role_name)
        print('----------')

        # 2.2 Map backend role
        print(f'Step 2.2: Map IAM role {create_connector_role_name} to OpenSearch permission role')
        self.iam_helper.map_iam_role_to_backend_role(create_connector_role_arn)
        print('----------')

        # Step 3: Create connector
        print('Step 3: Create connector in OpenSearch')
        # When you create an IAM role, it can take some time for the changes to propagate across AWS systems.
        # During this time, some services might not immediately recognize the new role or its permissions.
        # So we wait for some time before creating connector.
        # If you see such error: ClientError: An error occurred (AccessDenied) when calling the AssumeRole operation
        # you can rerun this function.

        # Wait for some time
        time.sleep(sleep_time_in_seconds)
        payload = create_connector_input
        payload['credential'] = {
            "roleArn": connector_role_arn
        }
        connector_id = self.create_connector(create_connector_role_name, payload)
        print('----------')
        return connector_id