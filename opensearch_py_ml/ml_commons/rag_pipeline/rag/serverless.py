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
import botocore
import json
import time
from urllib.parse import urlparse
from colorama import Fore, Style

class Serverless:
    def __init__(self, aoss_client, collection_name, iam_principal, aws_region):
        """
        Initialize the Serverless class with necessary AWS clients and configuration.

        :param aoss_client: Boto3 client for OpenSearch Serverless
        :param collection_name: Name of the OpenSearch collection
        :param iam_principal: IAM Principal ARN
        :param aws_region: AWS Region
        """
        self.aoss_client = aoss_client
        self.collection_name = collection_name
        self.iam_principal = iam_principal
        self.aws_region = aws_region

    def create_security_policies(self):
        """
        Create security policies for serverless OpenSearch.
        """
        encryption_policy = json.dumps({
            "Rules": [{"Resource": [f"collection/{self.collection_name}"], "ResourceType": "collection"}],
            "AWSOwnedKey": True
        })
        
        network_policy = json.dumps([{
            "Rules": [{"Resource": [f"collection/{self.collection_name}"], "ResourceType": "collection"}],
            "AllowFromPublic": True
        }])
        
        data_access_policy = json.dumps([{
            "Rules": [
                {"Resource": ["collection/*"], "Permission": ["aoss:*"], "ResourceType": "collection"},
                {"Resource": ["index/*/*"], "Permission": ["aoss:*"], "ResourceType": "index"}
            ],
            "Principal": [self.iam_principal],
            "Description": f"Data access policy for {self.collection_name}"
        }])
        
        encryption_policy_name = self.get_truncated_name(f"{self.collection_name}-enc-policy")
        self.create_security_policy("encryption", encryption_policy_name, f"{self.collection_name} encryption security policy", encryption_policy)
        self.create_security_policy("network", f"{self.collection_name}-net-policy", f"{self.collection_name} network security policy", network_policy)
        self.create_access_policy(self.get_truncated_name(f"{self.collection_name}-access-policy"), f"{self.collection_name} data access policy", data_access_policy)

    def create_security_policy(self, policy_type, name, description, policy_body):
        """
        Create a specific security policy (encryption or network).

        :param policy_type: Type of policy ('encryption' or 'network')
        :param name: Name of the policy
        :param description: Description of the policy
        :param policy_body: JSON string of the policy
        """
        try:
            if policy_type.lower() == "encryption":
                self.aoss_client.create_security_policy(
                    description=description,
                    name=name,
                    policy=policy_body,
                    type="encryption"
                )
            elif policy_type.lower() == "network":
                self.aoss_client.create_security_policy(
                    description=description,
                    name=name,
                    policy=policy_body,
                    type="network"
                )
            else:
                raise ValueError("Invalid policy type specified.")
            print(f"{Fore.GREEN}{policy_type.capitalize()} Policy '{name}' created successfully.{Style.RESET_ALL}")
        except self.aoss_client.exceptions.ConflictException:
            print(f"{Fore.YELLOW}{policy_type.capitalize()} Policy '{name}' already exists.{Style.RESET_ALL}")
        except Exception as ex:
            print(f"{Fore.RED}Error creating {policy_type} policy '{name}': {ex}{Style.RESET_ALL}")

    def create_access_policy(self, name, description, policy_body):
        """
        Create a data access policy.

        :param name: Name of the access policy
        :param description: Description of the access policy
        :param policy_body: JSON string of the access policy
        """
        try:
            self.aoss_client.create_access_policy(
                description=description,
                name=name,
                policy=policy_body,
                type="data"
            )
            print(f"{Fore.GREEN}Data Access Policy '{name}' created successfully.{Style.RESET_ALL}\n")
        except self.aoss_client.exceptions.ConflictException:
            print(f"{Fore.YELLOW}Data Access Policy '{name}' already exists.{Style.RESET_ALL}\n")
        except Exception as ex:
            print(f"{Fore.RED}Error creating data access policy '{name}': {ex}{Style.RESET_ALL}\n")

    def create_collection(self, collection_name, max_retries=3):
        """
        Create an OpenSearch serverless collection.

        :param collection_name: Name of the collection to create
        :param max_retries: Maximum number of retries for creation
        :return: Collection ID if successful, None otherwise
        """
        for attempt in range(max_retries):
            try:
                response = self.aoss_client.create_collection(
                    description=f"{collection_name} collection",
                    name=collection_name,
                    type="VECTORSEARCH"
                )
                print(f"{Fore.GREEN}Collection '{collection_name}' creation initiated.{Style.RESET_ALL}")
                return response['createCollectionDetail']['id']
            except self.aoss_client.exceptions.ConflictException:
                print(f"{Fore.YELLOW}Collection '{collection_name}' already exists.{Style.RESET_ALL}")
                return self.get_collection_id(collection_name)
            except Exception as ex:
                print(f"{Fore.RED}Error creating collection '{collection_name}' (Attempt {attempt+1}/{max_retries}): {ex}{Style.RESET_ALL}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(5)
        return None

    def get_collection_id(self, collection_name):
        """
        Retrieve the ID of an existing collection.

        :param collection_name: Name of the collection
        :return: Collection ID if found, None otherwise
        """
        try:
            response = self.aoss_client.list_collections()
            for collection in response.get('collectionSummaries', []):
                if collection.get('name') == collection_name:
                    return collection.get('id')
        except Exception as ex:
            print(f"{Fore.RED}Error getting collection ID: {ex}{Style.RESET_ALL}")
        return None

    def wait_for_collection_active(self, collection_id, max_wait_minutes=30):
        """
        Wait for the collection to become active.

        :param collection_id: ID of the collection
        :param max_wait_minutes: Maximum wait time in minutes
        :return: True if active, False otherwise
        """
        print(f"Waiting for collection '{self.collection_name}' to become active...")
        start_time = time.time()
        while time.time() - start_time < max_wait_minutes * 60:
            try:
                response = self.aoss_client.batch_get_collection(ids=[collection_id])
                status = response['collectionDetails'][0]['status']
                if status == 'ACTIVE':
                    print(f"{Fore.GREEN}Collection '{self.collection_name}' is now active.{Style.RESET_ALL}\n")
                    return True
                elif status in ['FAILED', 'DELETED']:
                    print(f"{Fore.RED}Collection creation failed or was deleted. Status: {status}{Style.RESET_ALL}\n")
                    return False
                else:
                    print(f"Collection status: {status}. Waiting...")
                    time.sleep(30)
            except Exception as ex:
                print(f"{Fore.RED}Error checking collection status: {ex}{Style.RESET_ALL}")
                time.sleep(30)
        print(f"{Fore.RED}Timed out waiting for collection to become active after {max_wait_minutes} minutes.{Style.RESET_ALL}\n")
        return False

    def get_collection_endpoint(self):
        """
        Retrieve the endpoint URL for the OpenSearch collection.

        :return: Collection endpoint URL if available, None otherwise
        """
        try:
            collection_id = self.get_collection_id(self.collection_name)
            if not collection_id:
                print(f"{Fore.RED}Collection '{self.collection_name}' not found.{Style.RESET_ALL}\n")
                return None
            
            batch_get_response = self.aoss_client.batch_get_collection(ids=[collection_id])
            collection_details = batch_get_response.get('collectionDetails', [])
            
            if not collection_details:
                print(f"{Fore.RED}No details found for collection ID '{collection_id}'.{Style.RESET_ALL}\n")
                return None
            
            endpoint = collection_details[0].get('collectionEndpoint')
            if endpoint:
                print(f"Collection '{self.collection_name}' has endpoint URL: {endpoint}\n")
                return endpoint
            else:
                print(f"{Fore.RED}No endpoint URL found in collection '{self.collection_name}'.{Style.RESET_ALL}\n")
                return None
        except Exception as ex:
            print(f"{Fore.RED}Error retrieving collection endpoint: {ex}{Style.RESET_ALL}\n")
            return None

    @staticmethod
    def get_truncated_name(base_name, max_length=32):
        """
        Truncate a name to fit within a specified length.

        :param base_name: Original name
        :param max_length: Maximum allowed length
        :return: Truncated name
        """
        if len(base_name) <= max_length:
            return base_name
        return base_name[:max_length-3] + "..."