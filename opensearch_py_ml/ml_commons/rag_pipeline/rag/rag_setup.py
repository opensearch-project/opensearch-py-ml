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
from botocore.config import Config
import configparser
import subprocess
import os
import json
import time
import termios
import tty
import sys
from urllib.parse import urlparse
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

class Setup:
    CONFIG_FILE = 'config.ini'
    SERVICE_AOSS = 'opensearchserverless'
    SERVICE_BEDROCK = 'bedrock-runtime'

    def __init__(self):
        self.aws_region = None
        self.iam_principal = None
        self.index_name = None
        self.collection_name = None
        self.opensearch_endpoint = None
        self.is_serverless = None
        self.opensearch_username = None
        self.opensearch_password = None
        self.aoss_client = None
        self.bedrock_client = None
        self.opensearch_client = None

    def check_and_configure_aws(self):
        try:
            session = boto3.Session()
            credentials = session.get_credentials()

            if credentials is None:
                print("AWS credentials are not configured.")
                self.configure_aws()
            else:
                print("AWS credentials are already configured.")
                reconfigure = input("Do you want to reconfigure? (yes/no): ").lower()
                if reconfigure == 'yes':
                    self.configure_aws()
        except Exception as e:
            print(f"An error occurred while checking AWS credentials: {e}")
            self.configure_aws()

    def configure_aws(self):
        print("Let's configure your AWS credentials.")

        aws_access_key_id = input("Enter your AWS Access Key ID: ")
        aws_secret_access_key = input("Enter your AWS Secret Access Key: ")
        aws_region_input = input("Enter your preferred AWS region (e.g., us-west-2): ")

        try:
            subprocess.run([
                'aws', 'configure', 'set', 
                'aws_access_key_id', aws_access_key_id
            ], check=True)
            
            subprocess.run([
                'aws', 'configure', 'set', 
                'aws_secret_access_key', aws_secret_access_key
            ], check=True)
            
            subprocess.run([
                'aws', 'configure', 'set', 
                'region', aws_region_input
            ], check=True)
            
            print("AWS credentials have been successfully configured.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while configuring AWS credentials: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def load_config(self):
        config = configparser.ConfigParser()
        if os.path.exists(self.CONFIG_FILE):
            config.read(self.CONFIG_FILE)
            return dict(config['DEFAULT'])
        return {}


    def save_config(self, config):
        parser = configparser.ConfigParser()
        parser['DEFAULT'] = config
        with open(self.CONFIG_FILE, 'w') as f:
            parser.write(f)

    def get_password_with_asterisks(self, prompt="Enter password: "):  # Accept 'prompt'
        import sys
        if sys.platform == 'win32':
            import msvcrt
            print(prompt, end='', flush=True)
            password = ""
            while True:
                key = msvcrt.getch()
                if key == b'\r':  # Enter key
                    sys.stdout.write('\n')
                    return password
                elif key == b'\x08':  # Backspace key
                    if len(password) > 0:
                        password = password[:-1]
                        sys.stdout.write('\b \b')  # Erase the last asterisk
                        sys.stdout.flush()
                else:
                    password += key.decode('utf-8')
                    sys.stdout.write('*')  # Mask input with '*'
                    sys.stdout.flush()
        else:
            import termios, tty
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                sys.stdout.write(prompt)
                sys.stdout.flush()
                password = ""
                while True:
                    ch = sys.stdin.read(1)
                    if ch in ('\r', '\n'):  # Enter key
                        sys.stdout.write('\n')
                        return password
                    elif ch == '\x7f':  # Backspace key
                        if len(password) > 0:
                            password = password[:-1]
                            sys.stdout.write('\b \b')  # Erase the last asterisk
                            sys.stdout.flush()
                    else:
                        password += ch
                        sys.stdout.write('*')  # Mask input with '*'
                        sys.stdout.flush()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def setup_configuration(self):
        config = self.load_config()

        self.aws_region = input(f"Enter your AWS Region [{config.get('region', 'us-west-2')}]: ") or config.get('region', 'us-west-2')
        self.iam_principal = input(f"Enter your IAM Principal ARN [{config.get('iam_principal', '')}]: ") or config.get('iam_principal', '')

        service_type = input("Choose OpenSearch service type (1 for Serverless, 2 for Managed): ")
        self.is_serverless = service_type == '1'

        if self.is_serverless:
            self.index_name = input("Enter a name for your KNN index in OpenSearch: ")
            self.collection_name = input("Enter the name for your OpenSearch collection: ")
            self.opensearch_endpoint = None
            self.opensearch_username = None
            self.opensearch_password = None
        else:
            self.index_name = input("Enter a name for your KNN index in OpenSearch: ")
            self.opensearch_endpoint = input("Enter your OpenSearch domain endpoint: ")
            self.opensearch_username = input("Enter your OpenSearch username: ")
            self.opensearch_password = self.get_password_with_asterisks("Enter your OpenSearch password: ")
            self.collection_name = ''

        self.config = {
            'region': self.aws_region,
            'iam_principal': self.iam_principal,
            'index_name': self.index_name,
            'collection_name': self.collection_name if self.collection_name else '',
            'is_serverless': str(self.is_serverless),
            'opensearch_endpoint': self.opensearch_endpoint if self.opensearch_endpoint else '',
            'opensearch_username': self.opensearch_username if self.opensearch_username else '',
            'opensearch_password': self.opensearch_password if self.opensearch_password else ''
        }
        self.save_config(self.config)
        print("Configuration saved successfully.")

    def initialize_clients(self):
        try:
            boto_config = Config(
                region_name=self.aws_region,
                signature_version='v4',
                retries={'max_attempts': 10, 'mode': 'standard'}
            )
            if self.is_serverless:
                self.aoss_client = boto3.client(self.SERVICE_AOSS, config=boto_config)
            self.bedrock_client = boto3.client(self.SERVICE_BEDROCK, region_name=self.aws_region)
            
            time.sleep(7)
            print("AWS clients initialized successfully.")
            return True
        except Exception as e:
            print(f"Failed to initialize AWS clients: {e}")
            return False

    def create_security_policies(self):
        if not self.is_serverless:
            print("Security policies are not applicable for managed OpenSearch domains.")
            return
        
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
        try:
            if policy_type.lower() == "encryption":
                self.aoss_client.create_security_policy(description=description, name=name, policy=policy_body, type="encryption")
            elif policy_type.lower() == "network":
                self.aoss_client.create_security_policy(description=description, name=name, policy=policy_body, type="network")
            else:
                raise ValueError("Invalid policy type specified.")
            print(f"{policy_type.capitalize()} Policy '{name}' created successfully.")
        except self.aoss_client.exceptions.ConflictException:
            print(f"{policy_type.capitalize()} Policy '{name}' already exists.")
        except Exception as ex:
            print(f"Error creating {policy_type} policy '{name}': {ex}")

    def create_access_policy(self, name, description, policy_body):
        try:
            self.aoss_client.create_access_policy(description=description, name=name, policy=policy_body, type="data")
            print(f"Data Access Policy '{name}' created successfully.")
        except self.aoss_client.exceptions.ConflictException:
            print(f"Data Access Policy '{name}' already exists.")
        except Exception as ex:
            print(f"Error creating data access policy '{name}': {ex}")

    def create_collection(self, collection_name, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = self.aoss_client.create_collection(
                    description=f"{collection_name} collection",
                    name=collection_name,
                    type="VECTORSEARCH"
                )
                print(f"Collection '{collection_name}' creation initiated.")
                return response['createCollectionDetail']['id']
            except self.aoss_client.exceptions.ConflictException:
                print(f"Collection '{collection_name}' already exists.")
                return self.get_collection_id(collection_name)
            except Exception as ex:
                print(f"Error creating collection '{collection_name}' (Attempt {attempt+1}/{max_retries}): {ex}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(5)
        return None

    def get_collection_id(self, collection_name):
        try:
            response = self.aoss_client.list_collections()
            for collection in response['collectionSummaries']:
                if collection['name'] == collection_name:
                    return collection['id']
        except Exception as ex:
            print(f"Error getting collection ID: {ex}")
        return None

    def wait_for_collection_active(self, collection_id, max_wait_minutes=30):
        print(f"Waiting for collection '{self.collection_name}' to become active...")
        start_time = time.time()
        while time.time() - start_time < max_wait_minutes * 60:
            try:
                response = self.aoss_client.batch_get_collection(ids=[collection_id])
                status = response['collectionDetails'][0]['status']
                if status == 'ACTIVE':
                    print(f"Collection '{self.collection_name}' is now active.")
                    return True
                elif status in ['FAILED', 'DELETED']:
                    print(f"Collection creation failed or was deleted. Status: {status}")
                    return False
                else:
                    print(f"Collection status: {status}. Waiting...")
                    time.sleep(30)
            except Exception as ex:
                print(f"Error checking collection status: {ex}")
                time.sleep(30)
        print(f"Timed out waiting for collection to become active after {max_wait_minutes} minutes.")
        return False

    def get_collection_endpoint(self):
        if not self.is_serverless:
            return self.opensearch_endpoint
        
        try:
            collection_id = self.get_collection_id(self.collection_name)
            if not collection_id:
                print(f"Collection '{self.collection_name}' not found.")
                return None
            
            batch_get_response = self.aoss_client.batch_get_collection(ids=[collection_id])
            collection_details = batch_get_response.get('collectionDetails', [])
            
            if not collection_details:
                print(f"No details found for collection ID '{collection_id}'.")
                return None
            
            self.opensearch_endpoint = collection_details[0].get('collectionEndpoint')
            if self.opensearch_endpoint:
                print(f"Collection '{self.collection_name}' has endpoint URL: {self.opensearch_endpoint}")
                return self.opensearch_endpoint
            else:
                print(f"No endpoint URL found in collection '{self.collection_name}'.")
                return None
        except Exception as ex:
            print(f"Error retrieving collection endpoint: {ex}")
            return None

    def initialize_opensearch_client(self):
        if not self.opensearch_endpoint:
            print("OpenSearch endpoint not set. Please run setup first.")
            return False
        
        parsed_url = urlparse(self.opensearch_endpoint)
        host = parsed_url.hostname
        port = parsed_url.port or 443

        if self.is_serverless:
            credentials = boto3.Session().get_credentials()
            auth = AWSV4SignerAuth(credentials, self.aws_region, 'aoss')
        else:
            if not self.opensearch_username or not self.opensearch_password:
                print("OpenSearch username or password not set. Please run setup first.")
                return False
            auth = (self.opensearch_username, self.opensearch_password)

        try:
            self.opensearch_client = OpenSearch(
                hosts=[{'host': host, 'port': port}],
                http_auth=auth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                pool_maxsize=20
            )
            print(f"Initialized OpenSearch client with host: {host} and port: {port}")
            return True
        except Exception as ex:
            print(f"Error initializing OpenSearch client: {ex}")
            return False

    def get_knn_index_details(self):
        # Simplified dimension input
        dimension_input = input("Press Enter to use the default embedding size (768), or type a custom size: ")
        
        if dimension_input.strip() == "":
            embedding_dimension = 768
        else:
            try:
                embedding_dimension = int(dimension_input)
            except ValueError:
                print("Invalid input. Using default dimension of 768.")
                embedding_dimension = 768

        print(f"\nEmbedding dimension set to: {embedding_dimension}")

        # Space type selection
        print("\nChoose the space type for KNN:")
        print("1. L2 (Euclidean distance)")
        print("2. Cosine similarity")
        print("3. Inner product")
        space_choice = input("Enter your choice (1-3), or press Enter for default (L2): ")

        if space_choice == "" or space_choice == "1":
            space_type = "l2"
        elif space_choice == "2":
            space_type = "cosinesimil"
        elif space_choice == "3":
            space_type = "innerproduct"
        else:
            print("Invalid choice. Using default space type of L2 (Euclidean distance).")
            space_type = "l2"

        print(f"Space type set to: {space_type}")

        return embedding_dimension, space_type


    def create_index(self, embedding_dimension, space_type):
        index_body = {
            "mappings": {
                "properties": {
                    "nominee_text": {"type": "text"},
                    "nominee_vector": {
                        "type": "knn_vector",
                        "dimension": embedding_dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": space_type,
                            "engine": "nmslib",
                            "parameters": {"ef_construction": 512, "m": 16},
                        },
                    },
                }
            },
            "settings": {
                "index": {
                    "number_of_shards": 2,
                    "knn.algo_param": {"ef_search": 512},
                    "knn": True,
                }
            },
        }
        try:
            self.opensearch_client.indices.create(index=self.index_name, body=index_body)
            print(f"KNN index '{self.index_name}' created successfully with dimension {embedding_dimension} and space type {space_type}.")
        except Exception as e:
            if 'resource_already_exists_exception' in str(e).lower():
                print(f"Index '{self.index_name}' already exists.")
            else:
                print(f"Error creating index '{self.index_name}': {e}")

    def verify_and_create_index(self, embedding_dimension, space_type):
        try:
            index_exists = self.opensearch_client.indices.exists(index=self.index_name)
            if index_exists:
                print(f"KNN index '{self.index_name}' already exists.")
            else:
                self.create_index(embedding_dimension, space_type)
            return True
        except Exception as ex:
            print(f"Error verifying or creating index: {ex}")
            return False

    def get_truncated_name(self, base_name, max_length=32):
        if len(base_name) <= max_length:
            return base_name
        return base_name[:max_length-3] + "..."

    def setup_command(self):
        self.check_and_configure_aws()
        self.setup_configuration()
        
        if not self.initialize_clients():
            print("Failed to initialize AWS clients. Setup incomplete.")
            return
        
        if self.is_serverless:
            self.create_security_policies()
            collection_id = self.get_collection_id(self.collection_name)
            if not collection_id:
                print(f"Collection '{self.collection_name}' not found. Attempting to create it...")
                collection_id = self.create_collection(self.collection_name)
            
            if collection_id:
                if self.wait_for_collection_active(collection_id):
                    self.opensearch_endpoint = self.get_collection_endpoint()
                    if not self.opensearch_endpoint:
                        print("Failed to retrieve OpenSearch endpoint. Setup incomplete.")
                        return
                    else:
                        self.config['opensearch_endpoint'] = self.opensearch_endpoint
                else:
                    print("Collection is not active. Setup incomplete.")
                    return
        else:
            if not self.opensearch_endpoint:
                print("OpenSearch endpoint not set. Setup incomplete.")
                return
        
        if self.initialize_opensearch_client():
            embedding_dimension, space_type = self.get_knn_index_details()
            if self.verify_and_create_index(embedding_dimension, space_type):
                print("Setup completed successfully.")
                self.config['embedding_dimension'] = str(embedding_dimension)
                self.config['space_type'] = space_type
            else:
                print("Index verification failed. Please check your index name and permissions.")
        else:
            print("Failed to initialize OpenSearch client. Setup incomplete.")