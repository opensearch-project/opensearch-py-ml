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

class SecretHelper:
    def __init__(self, region):
        self.region = region

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