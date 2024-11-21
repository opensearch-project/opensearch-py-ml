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
from AIConnectorHelper import AIConnectorHelper

# [Existing license and import statements remain unchanged]

class Setup:
    CONFIG_FILE = 'config.ini'
    SERVICE_AOSS = 'opensearchserverless'
    SERVICE_BEDROCK = 'bedrock-runtime'

    def __init__(self):
            # Initialize setup variables
            self.config = self.load_config()
            self.aws_region = self.config.get('region')
            self.iam_principal = self.config.get('iam_principal')
            self.index_name = self.config.get('index_name')
            self.collection_name = self.config.get('collection_name', '')
            self.opensearch_endpoint = self.config.get('opensearch_endpoint', '')
            self.is_serverless = self.config.get('is_serverless', 'False') == 'True'
            self.opensearch_username = self.config.get('opensearch_username', '')
            self.opensearch_password = self.config.get('opensearch_password', '')
            self.aoss_client = None
            self.bedrock_client = None
            self.opensearch_client = None

            # Initialize opensearch_domain_name
            self.opensearch_domain_name = self.get_opensearch_domain_name()

    def check_and_configure_aws(self):
        # Check if AWS credentials are configured and offer to reconfigure if needed
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
        # Configure AWS credentials using user input
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
        # Load configuration from the config file
        config = configparser.ConfigParser()
        if os.path.exists(self.CONFIG_FILE):
            config.read(self.CONFIG_FILE)
            return dict(config['DEFAULT'])
        return {}

    def save_config(self, config):
        # Save configuration to the config file
        parser = configparser.ConfigParser()
        parser['DEFAULT'] = config
        with open(self.CONFIG_FILE, 'w') as f:
            parser.write(f)

    def get_password_with_asterisks(self, prompt="Enter password: "):
        # Get password input from user, masking it with asterisks
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
        # Set up the configuration by prompting the user for various settings
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
        # Initialize AWS clients (AOSS and Bedrock)
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
        # Create security policies for serverless OpenSearch
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
        # Create a specific security policy (encryption or network)
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
        # Create a data access policy
        try:
            self.aoss_client.create_access_policy(description=description, name=name, policy=policy_body, type="data")
            print(f"Data Access Policy '{name}' created successfully.")
        except self.aoss_client.exceptions.ConflictException:
            print(f"Data Access Policy '{name}' already exists.")
        except Exception as ex:
            print(f"Error creating data access policy '{name}': {ex}")

    def create_collection(self, collection_name, max_retries=3):
        # Create an OpenSearch serverless collection
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
        # Retrieve the ID of an existing collection
        try:
            response = self.aoss_client.list_collections()
            for collection in response['collectionSummaries']:
                if collection['name'] == collection_name:
                    return collection['id']
        except Exception as ex:
            print(f"Error getting collection ID: {ex}")
        return None
    def get_opensearch_domain_name(self):
        """
        Extract the domain name from the OpenSearch endpoint URL.
        """
        if self.opensearch_endpoint:
            parsed_url = urlparse(self.opensearch_endpoint)
            hostname = parsed_url.hostname  # e.g., 'search-your-domain-name-uniqueid.region.es.amazonaws.com'
            if hostname:
                # Split the hostname into parts
                parts = hostname.split('.')
                domain_part = parts[0]  # e.g., 'search-your-domain-name-uniqueid'
                # Remove the 'search-' prefix if present
                if domain_part.startswith('search-'):
                    domain_part = domain_part[len('search-'):]
                # Remove the unique ID suffix after the domain name
                domain_name = domain_part.rsplit('-', 1)[0]
                print(f"Extracted domain name: {domain_name}")
                return domain_name
        return None
    def get_opensearch_domain_info(region, domain_name):
        """
        Retrieve the OpenSearch domain endpoint and ARN based on the domain name and region.
        """
        try:
            client = boto3.client('opensearch', region_name=region)
            response = client.describe_domain(DomainName=domain_name)
            domain_status = response['DomainStatus']
            domain_endpoint = domain_status.get('Endpoint') or domain_status.get('Endpoints', {}).get('vpc')
            domain_arn = domain_status['ARN']
            return domain_endpoint, domain_arn
        except Exception as e:
            print(f"Error retrieving OpenSearch domain info: {e}")
            return None, None

    def wait_for_collection_active(self, collection_id, max_wait_minutes=30):
        # Wait for the collection to become active
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
        # Retrieve the endpoint URL for the OpenSearch collection
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
    def get_iam_user_name_from_arn(self, iam_principal_arn):
        """
        Extract the IAM user name from the IAM principal ARN.
        """
        # IAM user ARN format: arn:aws:iam::123456789012:user/user-name
        if iam_principal_arn and ':user/' in iam_principal_arn:
            return iam_principal_arn.split(':user/')[-1]
        else:
            return None
        
    def initialize_opensearch_client(self):
        # Initialize the OpenSearch client
        if not self.opensearch_endpoint:
            print("OpenSearch endpoint not set. Please run setup first.")
            return False
        
        parsed_url = urlparse(self.opensearch_endpoint)
        host = parsed_url.hostname
        port = 443

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
        # Prompt user for KNN index details (embedding dimension, space type, and ef_construction)
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

        # New prompt for ef_construction
        ef_construction_input = input("\nPress Enter to use the default ef_construction value (512), or type a custom value: ")
        
        if ef_construction_input.strip() == "":
            ef_construction = 512
        else:
            try:
                ef_construction = int(ef_construction_input)
            except ValueError:
                print("Invalid input. Using default ef_construction of 512.")
                ef_construction = 512

        print(f"ef_construction set to: {ef_construction}")

        return embedding_dimension, space_type, ef_construction


    def create_index(self, embedding_dimension, space_type, ef_construction):
        # Create the KNN index in OpenSearch
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
                            "parameters": {"ef_construction": ef_construction, "m": 16},
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
            print(f"KNN index '{self.index_name}' created successfully with dimension {embedding_dimension}, space type {space_type}, and ef_construction {ef_construction}.")
        except Exception as e:
            if 'resource_already_exists_exception' in str(e).lower():
                print(f"Index '{self.index_name}' already exists.")
            else:
                print(f"Error creating index '{self.index_name}': {e}")



    def verify_and_create_index(self, embedding_dimension, space_type, ef_construction):
        try:
            print(f"Attempting to verify index '{self.index_name}'...")
            index_exists = self.opensearch_client.indices.exists(index=self.index_name)
            if index_exists:
                print(f"KNN index '{self.index_name}' already exists.")
            else:
                print(f"Index '{self.index_name}' does not exist. Attempting to create...")
                self.create_index(embedding_dimension, space_type, ef_construction)
            return True
        except Exception as ex:
            print(f"Error verifying or creating index: {ex}")
            print(f"OpenSearch client config: {self.opensearch_client.transport.hosts}")
            return False


    def get_truncated_name(self, base_name, max_length=32):
        # Truncate a name to fit within a specified length
        if len(base_name) <= max_length:
            return base_name
        return base_name[:max_length-3] + "..."

    def setup_command(self):
        # Main setup command that orchestrates the entire setup process
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
                        self.save_config(self.config)
                        # Initialize opensearch_domain_name after setting opensearch_endpoint
                        self.opensearch_domain_name = self.get_opensearch_domain_name()
                else:
                    print("Collection is not active. Setup incomplete.")
                    return
        else:
            if not self.opensearch_endpoint:
                print("OpenSearch endpoint not set. Setup incomplete.")
                return
            else:
                # Initialize opensearch_domain_name after setting opensearch_endpoint
                self.opensearch_domain_name = self.get_opensearch_domain_name()
        
        if self.initialize_opensearch_client():
            print("OpenSearch client initialized successfully. Proceeding with index creation...")
            embedding_dimension, space_type, ef_construction = self.get_knn_index_details()
            if self.verify_and_create_index(embedding_dimension, space_type, ef_construction):
                print("Setup completed successfully.")
                self.config['embedding_dimension'] = str(embedding_dimension)
                self.config['space_type'] = space_type
                self.config['ef_construction'] = str(ef_construction)
                self.save_config(self.config)
            else:
                print("Index verification failed. Please check your index name and permissions.")
        else:
            print("Failed to initialize OpenSearch client. Setup incomplete.")

    def register_model_command(self):
        """
        Command method to register a new embedding model.
        Prompts the user to select a model and gathers necessary inputs.
        """
        # Load existing config
        self.config = self.load_config()

        # Initialize clients
        if not self.initialize_clients():
            print("Failed to initialize AWS clients. Cannot proceed.")
            return

        # Ensure opensearch_endpoint is set
        if not self.opensearch_endpoint:
            self.opensearch_endpoint = self.config.get('opensearch_endpoint')
            if not self.opensearch_endpoint:
                print("OpenSearch endpoint not set. Please run 'setup' command first.")
                return

        # Initialize opensearch_domain_name
        self.opensearch_domain_name = self.get_opensearch_domain_name()

        # Extract the IAM user name from the IAM principal ARN
        aws_user_name = self.get_iam_user_name_from_arn(self.iam_principal)

        if not aws_user_name:
            print("Could not extract IAM user name from IAM principal ARN.")
            aws_user_name = input("Enter your AWS IAM user name: ")

        # Instantiate AIConnectorHelper
        helper = AIConnectorHelper(
            region=self.aws_region,
            opensearch_domain_name=self.opensearch_domain_name,
            opensearch_domain_username=self.opensearch_username,
            opensearch_domain_password=self.opensearch_password,
            aws_user_name=aws_user_name,
            aws_role_name=None  # Set to None or provide if applicable
        )
        

        # Prompt user to select a model
        print("Please select an embedding model to register:")
        print("1. Bedrock Titan Embedding Model")
        print("2. SageMaker Embedding Model")
        print("3. Cohere Embedding Model")
        print("4. OpenAI Embedding Model")
        model_choice = input("Enter your choice (1-4): ")

        # Call the appropriate method based on the user's choice
        if model_choice == '1':
            self.register_bedrock_model(helper)
        elif model_choice == '2':
            self.register_sagemaker_model(helper)
        elif model_choice == '3':
            self.register_cohere_model(helper)
        elif model_choice == '4':
            self.register_openai_model(helper)
        else:
            print("Invalid choice. Exiting.")
            return
        
    def register_bedrock_model(self, helper):
        """
        Register a Bedrock embedding model by creating the necessary connector and model in OpenSearch.
        """
        # Prompt for necessary inputs
        bedrock_region = input(f"Enter your Bedrock region [{self.aws_region}]: ") or self.aws_region
        connector_role_name = "my_test_bedrock_connector_role"
        create_connector_role_name = "my_test_create_bedrock_connector_role"

        # Set up connector role inline policy
        connector_role_inline_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": ["bedrock:InvokeModel"],
                    "Effect": "Allow",
                    "Resource": "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v1"
                }
            ]
        }

        # Create connector input
        create_connector_input = {
            "name": "Amazon Bedrock Connector: titan embedding v1",
            "description": "The connector to Bedrock Titan embedding model",
            "version": 1,
            "protocol": "aws_sigv4",
            "parameters": {
                "region": bedrock_region,
                "service_name": "bedrock"
            },
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": f"https://bedrock-runtime.{bedrock_region}.amazonaws.com/model/amazon.titan-embed-text-v1/invoke",
                    "headers": {
                        "content-type": "application/json",
                        "x-amz-content-sha256": "required"
                    },
                    "request_body": "{ \"inputText\": \"${parameters.inputText}\" }",
                    "pre_process_function": "\n    StringBuilder builder = new StringBuilder();\n    builder.append(\"\\\"\");\n    String first = params.text_docs[0];\n    builder.append(first);\n    builder.append(\"\\\"\");\n    def parameters = \"{\" +\"\\\"inputText\\\":\" + builder + \"}\";\n    return  \"{\" +\"\\\"parameters\\\":\" + parameters + \"}\";",
                    "post_process_function": "\n      def name = \"sentence_embedding\";\n      def dataType = \"FLOAT32\";\n      if (params.embedding == null || params.embedding.length == 0) {\n        return params.message;\n      }\n      def shape = [params.embedding.length];\n      def json = \"{\" +\n                 \"\\\"name\\\":\\\"\" + name + \"\\\",\" +\n                 \"\\\"data_type\\\":\\\"\" + dataType + \"\\\",\" +\n                 \"\\\"shape\\\":\" + shape + \",\" +\n                 \"\\\"data\\\":\" + params.embedding +\n                 \"}\";\n      return json;\n    "
                }
            ]
        }

        # Create connector
        connector_id = helper.create_connector_with_role(
            connector_role_inline_policy,
            connector_role_name,
            create_connector_role_name,
            create_connector_input,
            sleep_time_in_seconds=10
        )

        if not connector_id:
            print("Failed to create connector. Aborting.")
            return

        # Register model
# Register model
        model_name = 'Bedrock embedding model'
        description = 'Bedrock embedding model for semantic search'
        model_id = helper.create_model(model_name, description, connector_id, create_connector_role_name)

        if not model_id:
            print("Failed to create model. Aborting.")
            return

        # Save model_id to config
        self.config['embedding_model_id'] = model_id
        self.save_config(self.config)
        print(f"Model registered successfully. Model ID: {model_id}")

    def register_sagemaker_model(self, helper):
        """
        Register a SageMaker embedding model by creating the necessary connector and model in OpenSearch.
        """
        # Prompt for necessary inputs
        sagemaker_endpoint_arn = input("Enter your SageMaker inference endpoint ARN: ")
        sagemaker_endpoint_url = input("Enter your SageMaker inference endpoint URL: ")
        sagemaker_region = input(f"Enter your SageMaker region [{self.aws_region}]: ") or self.aws_region
        connector_role_name = "my_test_sagemaker_connector_role"
        create_connector_role_name = "my_test_create_sagemaker_connector_role"

        # Set up connector role inline policy
        connector_role_inline_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": ["sagemaker:InvokeEndpoint"],
                    "Effect": "Allow",
                    "Resource": sagemaker_endpoint_arn
                }
            ]
        }

        # Create connector input
        create_connector_input = {
            "name": "SageMaker embedding model connector",
            "description": "Connector for my SageMaker embedding model",
            "version": "1.0",
            "protocol": "aws_sigv4",
            "parameters": {
                "region": sagemaker_region,
                "service_name": "sagemaker"
            },
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "headers": {
                        "content-type": "application/json"
                    },
                    "url": sagemaker_endpoint_url,
                    "request_body": "${parameters.input}",
                    "pre_process_function": "connector.pre_process.default.embedding",
                    "post_process_function": "connector.post_process.default.embedding"
                }
            ]
        }

        # Create connector
        connector_id = helper.create_connector_with_role(
            connector_role_inline_policy,
            connector_role_name,
            create_connector_role_name,
            create_connector_input,
            sleep_time_in_seconds=10
        )

        if not connector_id:
            print("Failed to create connector. Aborting.")
            return

        # Register model
        model_name = 'SageMaker embedding model'
        description = 'SageMaker embedding model for semantic search'
        model_id = helper.create_model(model_name, description, connector_id, create_connector_role_name)

        if not model_id:
            print("Failed to create model. Aborting.")
            return

        # Save model_id to config
        self.config['embedding_model_id'] = model_id
        self.save_config(self.config)
        print(f"Model registered successfully. Model ID: {model_id}")

    def register_cohere_model(self, helper):
        """
        Register a Cohere embedding model by creating the necessary connector and model in OpenSearch.
        """
        # Prompt for necessary inputs
        secret_name = input("Enter a name for the AWS Secrets Manager secret: ")
        secret_key = 'cohere_api_key'
        cohere_api_key = input("Enter your Cohere API key: ")
        secret_value = {secret_key: cohere_api_key}

        connector_role_name = "my_test_cohere_connector_role"
        create_connector_role_name = "my_test_create_cohere_connector_role"

        # Create connector input
        create_connector_input = {
            "name": "Cohere Embedding Model Connector",
            "description": "Connector for Cohere embedding model",
            "version": "1.0",
            "protocol": "http",
            "parameters": {
                "model": "embed-english-v3.0",
                "input_type": "search_document",
                "truncate": "END"
            },
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": "https://api.cohere.ai/v1/embed",
                    "headers": {
                        "Authorization": f"Bearer ${{credential.secretArn.{secret_key}}}",
                        "Request-Source": "unspecified:opensearch"
                    },
                    "request_body": "{ \"texts\": ${parameters.texts}, \"truncate\": \"${parameters.truncate}\", \"model\": \"${parameters.model}\", \"input_type\": \"${parameters.input_type}\" }",
                    "pre_process_function": "connector.pre_process.cohere.embedding",
                    "post_process_function": "connector.post_process.cohere.embedding"
                }
            ]
        }

        # Create connector
        connector_id = helper.create_connector_with_secret(
            secret_name,
            secret_value,
            connector_role_name,
            create_connector_role_name,
            create_connector_input,
            sleep_time_in_seconds=10
        )

        if not connector_id:
            print("Failed to create connector. Aborting.")
            return

        # Register model
        model_name = 'Cohere embedding model'
        description = 'Cohere embedding model for semantic search'
        model_id = helper.create_model(model_name, description, connector_id, create_connector_role_name)

        if not model_id:
            print("Failed to create model. Aborting.")
            return

        # Save model_id to config
        self.config['embedding_model_id'] = model_id
        self.save_config(self.config)
        print(f"Model registered successfully. Model ID: {model_id}")

    def register_openai_model(self, helper):
        """
        Register an OpenAI embedding model by creating the necessary connector and model in OpenSearch.
        """
        # Prompt for necessary inputs
        secret_name = input("Enter a name for the AWS Secrets Manager secret: ")
        secret_key = 'openai_api_key'
        openai_api_key = input("Enter your OpenAI API key: ")
        secret_value = {secret_key: openai_api_key}

        connector_role_name = "my_test_openai_connector_role"
        create_connector_role_name = "my_test_create_openai_connector_role"

        # Create connector input
        create_connector_input = {
            "name": "OpenAI Embedding Model Connector",
            "description": "Connector for OpenAI embedding model",
            "version": "1.0",
            "protocol": "http",
            "parameters": {
                "model": "text-embedding-ada-002"
            },
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": "https://api.openai.com/v1/embeddings",
                    "headers": {
                        "Authorization": f"Bearer ${{credential.secretArn.{secret_key}}}",
                    },
                    "request_body": "{ \"input\": ${parameters.input}, \"model\": \"${parameters.model}\" }",
                    "pre_process_function": "connector.pre_process.openai.embedding",
                    "post_process_function": "connector.post_process.openai.embedding"
                }
            ]
        }

        # Create connector
        connector_id = helper.create_connector_with_secret(
            secret_name,
            secret_value,
            connector_role_name,
            create_connector_role_name,
            create_connector_input,
            sleep_time_in_seconds=10
        )

        if not connector_id:
            print("Failed to create connector. Aborting.")
            return

        # Register model
        model_name = 'OpenAI embedding model'
        description = 'OpenAI embedding model for semantic search'
        model_id = helper.create_model(model_name, description, connector_id, create_connector_role_name)

        if not model_id:
            print("Failed to create model. Aborting.")
            return

        # Save model_id to config
        self.config['embedding_model_id'] = model_id
        self.save_config(self.config)
        print(f"Model registered successfully. Model ID: {model_id}")