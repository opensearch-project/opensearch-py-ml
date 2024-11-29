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
from botocore.exceptions import BotoCoreError
import requests

class IAMRoleHelper:
    def __init__(self, region, opensearch_domain_url=None, opensearch_domain_username=None,
                 opensearch_domain_password=None, aws_user_name=None, aws_role_name=None, opensearch_domain_arn=None):
        self.region = region
        self.opensearch_domain_url = opensearch_domain_url
        self.opensearch_domain_username = opensearch_domain_username
        self.opensearch_domain_password = opensearch_domain_password
        self.aws_user_name = aws_user_name
        self.aws_role_name = aws_role_name
        self.opensearch_domain_arn = opensearch_domain_arn

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

    def assume_role(self, role_arn, role_session_name="your_session_name"):
        sts_client = boto3.client('sts')

        assumed_role_object = sts_client.assume_role(
            RoleArn=role_arn,
            RoleSessionName=role_session_name,
        )

        # Obtain the temporary credentials from the assumed role 
        temp_credentials = assumed_role_object["Credentials"]

        return temp_credentials

    def map_iam_role_to_backend_role(self, iam_role_arn):
        os_security_role = 'ml_full_access'  # Changed from 'all_access' to 'ml_full_access'
        url = f'{self.opensearch_domain_url}/_plugins/_security/api/rolesmapping/{os_security_role}'

        payload = {
            "backend_roles": [iam_role_arn]
        }
        headers = {'Content-Type': 'application/json'}

        response = requests.put(
            url,
            auth=(self.opensearch_domain_username, self.opensearch_domain_password),
            json=payload,
            headers=headers,
            verify=True
        )

        if response.status_code == 200:
            print(f"Successfully mapped IAM role to OpenSearch role '{os_security_role}'.")
        else:
            print(f"Failed to map IAM role to OpenSearch role '{os_security_role}'. Status code: {response.status_code}")
            print(f"Response: {response.text}")