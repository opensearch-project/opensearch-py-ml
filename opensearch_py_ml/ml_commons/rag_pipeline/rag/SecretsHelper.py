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

import logging
import boto3
import json
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class SecretHelper:
    def __init__(self, region):
        self.region = region

    def secret_exists(self, secret_name):
        secretsmanager = boto3.client('secretsmanager', region_name=self.region)
        try:
            secretsmanager.get_secret_value(SecretId=secret_name)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                return False
            else:
                logger.error(f"An error occurred: {e}")
                return False

    def get_secret_arn(self, secret_name):
        secretsmanager = boto3.client('secretsmanager', region_name=self.region)
        try:
            response = secretsmanager.describe_secret(SecretId=secret_name)
            return response['ARN']
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.warning(f"The requested secret {secret_name} was not found")
                return None
            else:
                logger.error(f"An error occurred: {e}")
                return None

    def get_secret(self, secret_name):
        secretsmanager = boto3.client('secretsmanager', region_name=self.region)
        try:
            response = secretsmanager.get_secret_value(SecretId=secret_name)
            return response.get('SecretString')
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.warning("The requested secret was not found")
                return None
            else:
                logger.error(f"An error occurred: {e}")
                return None

    def create_secret(self, secret_name, secret_value):
        secretsmanager = boto3.client('secretsmanager', region_name=self.region)
        try:
            response = secretsmanager.create_secret(
                Name=secret_name,
                SecretString=json.dumps(secret_value),
            )
            logger.info(f'Secret {secret_name} created successfully.')
            return response['ARN']
        except ClientError as e:
            logger.error(f'Error creating secret: {e}')
            return None