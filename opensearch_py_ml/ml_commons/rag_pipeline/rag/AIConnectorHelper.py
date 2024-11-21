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
from botocore.exceptions import BotoCoreError
import json
import requests
from requests.auth import HTTPBasicAuth
from requests_aws4auth import AWS4Auth
import time
from opensearchpy import OpenSearch, RequestsHttpConnection

# This Python code is compatible with AWS OpenSearch versions 2.9 and higher.
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
        
    def get_user_arn(self, username):
        if not username:
            return None
        # Create a boto3 client for IAM
        iam_client = boto3.client('iam')

        try:
            # Get information about the IAM user
            response = iam_client.get_user(UserName=username)
            user_arn = response['User']['Arn']
            return user_arn
        except iam_client.exceptions.NoSuchEntityException:
            print(f"IAM user '{username}' not found.")
            return None

    def secret_exists(self, secret_name):
        secretsmanager = boto3.client('secretsmanager', region_name=self.region)
        try:
            # Try to get the secret
            secretsmanager.get_secret_value(SecretId=secret_name)
            # If no exception was raised by get_secret_value, the secret exists
            return True
        except secretsmanager.exceptions.ResourceNotFoundException:
            # If a ResourceNotFoundException was raised, the secret does not exist
            return False

    def get_secret_arn(self, secret_name):
        secretsmanager = boto3.client('secretsmanager', region_name=self.region)
        try:
            response = secretsmanager.describe_secret(SecretId=secret_name)
            # Return ARN of the secret 
            return response['ARN']
        except secretsmanager.exceptions.ResourceNotFoundException:
            print(f"The requested secret {secret_name} was not found")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
    def get_secret(self, secret_name):
        secretsmanager = boto3.client('secretsmanager', region_name=self.region)
        try:
            response = secretsmanager.get_secret_value(SecretId=secret_name)
        except secretsmanager.exceptions.NoSuchEntityException:
            print("The requested secret was not found")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        else:
            return response.get('SecretString')

    def create_secret(self, secret_name, secret_value):
        secretsmanager = boto3.client('secretsmanager', region_name=self.region)

        try:
            response = secretsmanager.create_secret(
                Name=secret_name,
                SecretString=json.dumps(secret_value),
            )
            print(f'Secret {secret_name} created successfully.')
            return response['ARN']  # Return the ARN of the created secret
        except BotoCoreError as e:
            print(f'Error creating secret: {e}')
            return None


    def role_exists(self, role_name):
        iam_client = boto3.client('iam')

        try:
            iam_client.get_role(RoleName=role_name)
            return True
        except iam_client.exceptions.NoSuchEntityException:
            return False

    def delete_role(self, role_name):
        iam_client = boto3.client('iam')

        try:
            # Detach managed policies
            policies = iam_client.list_attached_role_policies(RoleName=role_name)['AttachedPolicies']
            for policy in policies:
                iam_client.detach_role_policy(RoleName=role_name, PolicyArn=policy['PolicyArn'])
            print(f'All managed policies detached from role {role_name}.')

            # Delete inline policies
            inline_policies = iam_client.list_role_policies(RoleName=role_name)['PolicyNames']
            for policy_name in inline_policies:
                iam_client.delete_role_policy(RoleName=role_name, PolicyName=policy_name)
            print(f'All inline policies deleted from role {role_name}.')

            # Now, delete the role
            iam_client.delete_role(RoleName=role_name)
            print(f'Role {role_name} deleted.')

        except iam_client.exceptions.NoSuchEntityException:
            print(f'Role {role_name} does not exist.')


    def create_iam_role(self, role_name, trust_policy_json, inline_policy_json):
        iam_client = boto3.client('iam')

        try:
            # Create the role with the trust policy
            create_role_response = iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy_json),
                Description='Role with custom trust and inline policies',
            )

            # Get the ARN of the newly created role
            role_arn = create_role_response['Role']['Arn']

            # Attach the inline policy to the role
            iam_client.put_role_policy(
                RoleName=role_name,
                PolicyName='InlinePolicy',  # you can replace this with your preferred policy name
                PolicyDocument=json.dumps(inline_policy_json)
            )

            print(f'Created role: {role_name}')
            return role_arn

        except Exception as e:
            print(f"Error creating the role: {e}")
            return None

    def get_role_arn(self, role_name):
        if not role_name:
            return None
        iam_client = boto3.client('iam')
        try:
            response = iam_client.get_role(RoleName=role_name)
            # Return ARN of the role
            return response['Role']['Arn']
        except iam_client.exceptions.NoSuchEntityException:
            print(f"The requested role {role_name} does not exist")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    
    
    def get_role_details(self, role_name):
        iam = boto3.client('iam')

        try:
            response = iam.get_role(RoleName=role_name)
            role = response['Role']

            print(f"Role Name: {role['RoleName']}")
            print(f"Role ID: {role['RoleId']}")
            print(f"ARN: {role['Arn']}")
            print(f"Creation Date: {role['CreateDate']}")
            print("Assume Role Policy Document:")
            print(json.dumps(role['AssumeRolePolicyDocument'], indent=4, sort_keys=True))

            list_role_policies_response = iam.list_role_policies(RoleName=role_name)

            for policy_name in list_role_policies_response['PolicyNames']:
                get_role_policy_response = iam.get_role_policy(RoleName=role_name, PolicyName=policy_name)
                print(f"Role Policy Name: {get_role_policy_response['PolicyName']}")
                print("Role Policy Document:")
                print(json.dumps(get_role_policy_response['PolicyDocument'], indent=4, sort_keys=True))

        except iam.exceptions.NoSuchEntityException:
            print(f'Role {role_name} does not exist.')

    def map_iam_role_to_backend_role(self, iam_role_arn):
        os_security_role = 'ml_full_access'  # Changed from 'all_access' to 'ml_full_access'
        url = f'{self.opensearch_domain_url}/_plugins/_security/api/rolesmapping/{os_security_role}'

        payload = {
            "backend_roles": [iam_role_arn]
        }
        headers = {'Content-Type': 'application/json'}

        response = requests.put(url, auth=(self.opensearch_domain_username, self.opensearch_domain_password),
                                json=payload, headers=headers, verify=True)

        if response.status_code == 200:
            print(f"Successfully mapped IAM role to OpenSearch role '{os_security_role}'.")
        else:
            print(f"Failed to map IAM role to OpenSearch role '{os_security_role}'. Status code: {response.status_code}")
            print(f"Response: {response.text}")

    def assume_role(self, create_connector_role_arn, role_session_name="your_session_name"):
        sts_client = boto3.client('sts')

        #role_arn = f"arn:aws:iam::{aws_account_id}:role/{role_name}"
        assumed_role_object = sts_client.assume_role(
            RoleArn=create_connector_role_arn,
            RoleSessionName=role_session_name,
        )

        # Obtain the temporary credentials from the assumed role 
        temp_credentials = assumed_role_object["Credentials"]

        return temp_credentials
    
    def get_ml_auth(self, create_connector_role_name):
        """
        Obtain AWS4Auth credentials for ML API calls using the specified IAM role.
        """
        create_connector_role_arn = self.get_role_arn(create_connector_role_name)
        if not create_connector_role_arn:
            raise Exception(f"IAM role '{create_connector_role_name}' not found.")

        temp_credentials = self.assume_role(create_connector_role_arn)
        awsauth = AWS4Auth(
            temp_credentials["AccessKeyId"],
            temp_credentials["SecretAccessKey"],
            self.region,
            'es',
            session_token=temp_credentials["SessionToken"],
        )
        return awsauth

    def create_connector(self, create_connector_role_name, payload):
        create_connector_role_arn = self.get_role_arn(create_connector_role_name)
        temp_credentials = self.assume_role(create_connector_role_arn)
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
    
    def search_model_group(self, model_group_name):
        payload = {
            "query": {
                "term": {
                    "name.keyword": {
                        "value": model_group_name
                    }
                }
            }
        }
        headers = {"Content-Type": "application/json"}
        
        # Obtain temporary credentials
        create_connector_role_arn = self.get_role_arn('my_test_create_bedrock_connector_role')  # Replace with actual role name
        temp_credentials = self.assume_role(create_connector_role_arn)
        awsauth = AWS4Auth(
            temp_credentials["AccessKeyId"],
            temp_credentials["SecretAccessKey"],
            self.region,
            'es',
            session_token=temp_credentials["SessionToken"],
        )
        
        r = requests.post(
            f'{self.opensearch_domain_url}/_plugins/_ml/model_groups/_search',
            auth=awsauth,
            json=payload,
            headers=headers
        )
        
        response = json.loads(r.text)
        return response
    
    def create_model_group(self, model_group_name, description, create_connector_role_name):
        search_model_group_response = self.search_model_group(model_group_name)
        print("Search Model Group Response:", search_model_group_response)
        
        if 'hits' in search_model_group_response and search_model_group_response['hits']['total']['value'] > 0:
            return search_model_group_response['hits']['hits'][0]['_id']
        
        payload = {
            "name": model_group_name,
            "description": description
        }
        headers = {"Content-Type": "application/json"}
        
        # Obtain temporary credentials using the provided role name
        create_connector_role_arn = self.get_role_arn(create_connector_role_name)
        temp_credentials = self.assume_role(create_connector_role_arn)
        awsauth = AWS4Auth(
            temp_credentials["AccessKeyId"],
            temp_credentials["SecretAccessKey"],
            self.region,
            'es',
            session_token=temp_credentials["SessionToken"],
        )
        
        r = requests.post(
            f'{self.opensearch_domain_url}/_plugins/_ml/model_groups/_register',
            auth=awsauth,
            json=payload,
            headers=headers
        )
        
        print(r.text)
        response = json.loads(r.text)
        
        if 'model_group_id' in response:
            return response['model_group_id']
        else:
            # Handle error gracefully
            raise KeyError("The response does not contain 'model_group_id'. Response content: {}".format(response))
    
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
        return requests.post(f'{self.opensearch_domain_url}/_plugins/_ml/models/{model_id}/_deploy',
                             auth=HTTPBasicAuth(self.opensearch_domain_username, self.opensearch_domain_password),
                             headers=headers)
    
    def predict(self, model_id, payload):
        headers = {"Content-Type": "application/json"}
        r = requests.post(
            f'{self.opensearch_domain_url}/_plugins/_ml/models/{model_id}/_predict',
            auth=HTTPBasicAuth(self.opensearch_domain_username, self.opensearch_domain_password),
            json=payload,
            headers=headers
        )
    
    def create_connector_with_secret(self, secret_name, secret_value, connector_role_name, create_connector_role_name, create_connector_input, sleep_time_in_seconds=10):
        # Step1: Create Secret
        print('Step1: Create Secret')
        if not self.secret_exists(secret_name):
            secret_arn = self.create_secret(secret_name, secret_value)
        else:
            print('secret exists, skip creating')
            secret_arn = self.get_secret_arn(secret_name)
        #print(secret_arn)
        print('----------')
        
        # Step2: Create IAM role configued in connector
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

        print('Step2: Create IAM role configued in connector')
        if not self.role_exists(connector_role_name):
            connector_role_arn = self.create_iam_role(connector_role_name, trust_policy, inline_policy)
        else:
            print('role exists, skip creating')
            connector_role_arn = self.get_role_arn(connector_role_name)
        #print(connector_role_arn)
        print('----------')
        
        # Step 3: Configure IAM role in OpenSearch
        # 3.1 Create IAM role for Signing create connector request
        user_arn = self.get_user_arn(self.aws_user_name)
        role_arn = self.get_role_arn(self.aws_role_name)
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
        if not self.role_exists(create_connector_role_name):
            create_connector_role_arn = self.create_iam_role(create_connector_role_name, trust_policy, inline_policy)
        else:
            print('role exists, skip creating')
            create_connector_role_arn = self.get_role_arn(create_connector_role_name)
        #print(create_connector_role_arn)
        print('----------')
        
        # 3.2 Map backend role
        print(f'Step 3.2: Map IAM role {create_connector_role_name} to OpenSearch permission role')
        self.map_iam_role_to_backend_role(create_connector_role_arn)
        print('----------')
        
        # 4. Create connector
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
        #print(connector_id)
        print('----------')
        return connector_id
    
    def create_connector_with_role(self, connector_role_inline_policy, connector_role_name, create_connector_role_name, create_connector_input, sleep_time_in_seconds=10):
            # Step1: Create IAM role configued in connector
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

            print('Step1: Create IAM role configued in connector')
            if not self.role_exists(connector_role_name):
                connector_role_arn = self.create_iam_role(connector_role_name, trust_policy, connector_role_inline_policy)
            else:
                print('role exists, skip creating')
                connector_role_arn = self.get_role_arn(connector_role_name)
            #print(connector_role_arn)
            print('----------')

            # Step 2: Configure IAM role in OpenSearch
            # 2.1 Create IAM role for Signing create connector request
            user_arn = self.get_user_arn(self.aws_user_name)
            role_arn = self.get_role_arn(self.aws_role_name)
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
            if not self.role_exists(create_connector_role_name):
                create_connector_role_arn = self.create_iam_role(create_connector_role_name, trust_policy, inline_policy)
            else:
                print('role exists, skip creating')
                create_connector_role_arn = self.get_role_arn(create_connector_role_name)
            #print(create_connector_role_arn)
            print('----------')

            # 2.2 Map backend role
            print(f'Step 2.2: Map IAM role {create_connector_role_name} to OpenSearch permission role')
            self.map_iam_role_to_backend_role(create_connector_role_arn)
            print('----------')

            # 3. Create connector
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
            #print(connector_id)
            print('----------')
            return connector_id