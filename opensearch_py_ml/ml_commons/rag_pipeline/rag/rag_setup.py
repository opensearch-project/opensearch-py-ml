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
from colorama import Fore, Style, init
import ssl

from serverless import Serverless  
from AIConnectorHelper import AIConnectorHelper 
from model_register import ModelRegister

init(autoreset=True)

class Setup:
    CONFIG_FILE = 'config.ini'
    SERVICE_AOSS = 'opensearchserverless'
    SERVICE_BEDROCK = 'bedrock-runtime'

    def __init__(self):
        # Initialize setup variables
        self.config = self.load_config()
        self.aws_region = self.config.get('region', 'us-west-2')
        self.iam_principal = self.config.get('iam_principal', '')
        self.collection_name = self.config.get('collection_name', '')
        self.opensearch_endpoint = self.config.get('opensearch_endpoint', '')
        self.service_type = self.config.get('service_type', 'managed')
        self.is_serverless = self.service_type == 'serverless'
        self.opensearch_username = self.config.get('opensearch_username', '')
        self.opensearch_password = self.config.get('opensearch_password', '')
        self.aoss_client = None
        self.bedrock_client = None
        self.opensearch_client = None
        self.opensearch_domain_name = self.get_opensearch_domain_name()
        self.model_register = None
        self.serverless = None  # Will be initialized if service_type is 'serverless'

    def check_and_configure_aws(self):
        """
        Check if AWS credentials are configured and offer to reconfigure if needed.
        """
        try:
            session = boto3.Session()
            credentials = session.get_credentials()

            if credentials is None:
                print(f"{Fore.YELLOW}AWS credentials are not configured.{Style.RESET_ALL}")
                self.configure_aws()
            else:
                print("AWS credentials are already configured.")
                reconfigure = input("Do you want to reconfigure? (yes/no): ").lower()
                if reconfigure == 'yes':
                    self.configure_aws()
        except Exception as e:
            print(f"{Fore.RED}An error occurred while checking AWS credentials: {e}{Style.RESET_ALL}")
            self.configure_aws()

    def configure_aws(self):
        """
        Configure AWS credentials using user input.
        """
        print("Let's configure your AWS credentials.")

        aws_access_key_id = input("Enter your AWS Access Key ID: ").strip()
        aws_secret_access_key = input("Enter your AWS Secret Access Key: ").strip()
        aws_region_input = input("Enter your preferred AWS region (e.g., us-west-2): ").strip()

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
            
            print(f"{Fore.GREEN}AWS credentials have been successfully configured.{Style.RESET_ALL}")
        except subprocess.CalledProcessError as e:
            print(f"{Fore.RED}An error occurred while configuring AWS credentials: {e}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}An unexpected error occurred: {e}{Style.RESET_ALL}")

    def load_config(self):
        """
        Load configuration from the config file.

        :return: Dictionary of configuration parameters
        """
        config = configparser.ConfigParser()
        if os.path.exists(self.CONFIG_FILE):
            config.read(self.CONFIG_FILE)
            return dict(config['DEFAULT'])
        return {}

    def get_password_with_asterisks(self, prompt="Enter password: "):
        """
        Get password input from user, masking it with asterisks.

        :param prompt: Prompt message
        :return: Entered password as a string
        """
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
        """
        Set up the configuration by prompting the user for various settings.
        """
        config = self.load_config()

        print("\nStarting setup process...")

        # First, prompt for service type
        print("\nChoose OpenSearch service type:")
        print("1. Serverless")
        print("2. Managed")
        print("3. Open-source")
        service_choice = input("Enter your choice (1-3): ").strip()

        if service_choice == '1':
            self.service_type = 'serverless'
        elif service_choice == '2':
            self.service_type = 'managed'
        elif service_choice == '3':
            self.service_type = 'open-source'
        else:
            print(f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'managed'.{Style.RESET_ALL}")
            self.service_type = 'managed'

        # Based on service type, prompt for different configurations
        if self.service_type in ['serverless', 'managed']:
            # For 'serverless' and 'managed', prompt for AWS credentials and related info
            self.check_and_configure_aws()

            self.aws_region = input(f"\nEnter your AWS Region [{self.aws_region}]: ").strip() or self.aws_region
            self.iam_principal = input(f"Enter your IAM Principal ARN [{self.iam_principal}]: ").strip() or self.iam_principal

            if self.service_type == 'serverless':
                self.collection_name = input("\nEnter the name for your OpenSearch collection: ").strip()
                self.opensearch_endpoint = None
                self.opensearch_username = None
                self.opensearch_password = None
            elif self.service_type == 'managed':
                self.opensearch_endpoint = input("\nEnter your OpenSearch domain endpoint: ").strip()
                self.opensearch_username = input("Enter your OpenSearch username: ").strip()
                self.opensearch_password = self.get_password_with_asterisks("Enter your OpenSearch password: ")
                self.collection_name = ''
        elif self.service_type == 'open-source':
            # For 'open-source', skip AWS configurations
            print("\n--- Open-source OpenSearch Setup ---")
            default_endpoint = 'https://localhost:9200'
            self.opensearch_endpoint = input(f"\nPress Enter to use the default endpoint (or type your custom endpoint) [{default_endpoint}]: ").strip() or default_endpoint
            auth_required = input("Does your OpenSearch instance require authentication? (yes/no): ").strip().lower()
            if auth_required == 'yes':
                self.opensearch_username = input("Enter your OpenSearch username: ").strip()
                self.opensearch_password = self.get_password_with_asterisks("Enter your OpenSearch password: ")
            else:
                self.opensearch_username = None
                self.opensearch_password = None
            self.collection_name = ''
            # For open-source, AWS region and IAM principal are not needed
            self.aws_region = ''
            self.iam_principal = ''

        # Update configuration dictionary
        self.config = {
            'service_type': self.service_type,
            'region': self.aws_region,
            'iam_principal': self.iam_principal,
            'collection_name': self.collection_name if self.collection_name else '',
            'opensearch_endpoint': self.opensearch_endpoint if self.opensearch_endpoint else '',
            'opensearch_username': self.opensearch_username if self.opensearch_username else '',
            'opensearch_password': self.opensearch_password if self.opensearch_password else ''
        }

        # Now, prompt for default search method
        print("\nChoose the default search method:")
        print("1. Neural Search")
        print("2. Semantic Search")
        search_choice = input("Enter your choice (1-2): ").strip()

        if search_choice == '1':
            default_search_method = 'neural'
        elif search_choice == '2':
            default_search_method = 'semantic'
        else:
            print(f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'neural'.{Style.RESET_ALL}")
            default_search_method = 'neural'

        self.config['default_search_method'] = default_search_method

        if default_search_method == 'semantic':
            # Prompt the user to select an LLM model for semantic search
            print("\nSelect an LLM model for semantic search:")
            available_models = [
                ("amazon.titan-text-lite-v1", "Bedrock Titan Text Lite V1"),
                ("amazon.titan-text-express-v1", "Bedrock Titan Text Express V1"),
                ("anthropic.claude-3-5-sonnet-20240620-v1:0", "Anthropic Claude 3.5 Sonnet"),
                ("anthropic.claude-3-opus-20240229-v1:0", "Anthropic Claude 3 Opus"),
                ("cohere.command-r-plus-v1:0", "Cohere Command R Plus V1"),
                ("cohere.command-r-v1:0", "Cohere Command R V1")
            ]
            for idx, (model_id, model_name) in enumerate(available_models, start=1):
                print(f"{idx}. {model_name} ({model_id})")
            model_choice = input(f"\nEnter the number of your chosen model (1-{len(available_models)}): ").strip()
            try:
                model_choice_idx = int(model_choice) - 1
                if 0 <= model_choice_idx < len(available_models):
                    selected_model_id = available_models[model_choice_idx][0]
                    self.config['llm_model_id'] = selected_model_id
                    print(f"\nSelected LLM Model ID: {selected_model_id}")
                else:
                    print(f"\n{Fore.YELLOW}Invalid choice. Defaulting to '{available_models[0][0]}'.{Style.RESET_ALL}")
                    self.config['llm_model_id'] = available_models[0][0]
            except ValueError:
                print(f"\n{Fore.YELLOW}Invalid input. Defaulting to '{available_models[0][0]}'.{Style.RESET_ALL}")
                self.config['llm_model_id'] = available_models[0][0]

            # Prompt for LLM configurations
            print("\nConfigure LLM settings:")
            try:
                maxTokenCount = int(input("Enter max token count [1000]: ").strip() or "1000")
            except ValueError:
                maxTokenCount = 1000
            try:
                temperature = float(input("Enter temperature [0.7]: ").strip() or "0.7")
            except ValueError:
                temperature = 0.7
            try:
                topP = float(input("Enter topP [0.9]: ").strip() or "0.9")
            except ValueError:
                topP = 0.9
            stopSequences_input = input("Enter stop sequences (comma-separated) or press Enter for none: ").strip()
            if stopSequences_input:
                stopSequences = [s.strip() for s in stopSequences_input.split(',')]
            else:
                stopSequences = []

            # Save LLM configurations to config
            self.config['llm_max_token_count'] = str(maxTokenCount)
            self.config['llm_temperature'] = str(temperature)
            self.config['llm_top_p'] = str(topP)
            self.config['llm_stop_sequences'] = ','.join(stopSequences)

        # Prompt for ingest pipeline name
        default_pipeline_name = 'text-chunking-ingest-pipeline'
        pipeline_name = input(f"\nEnter the name of the ingest pipeline to use [{default_pipeline_name}]: ").strip()
        if not pipeline_name:
            pipeline_name = default_pipeline_name

        # Save the pipeline name to config
        self.config['ingest_pipeline_name'] = pipeline_name

        # Save the configuration
        self.save_config(self.config)
        print(f"\n{Fore.GREEN}Configuration saved successfully to {os.path.abspath(self.CONFIG_FILE)}.{Style.RESET_ALL}\n")


    def initialize_clients(self):
        """
        Initialize AWS clients (AOSS and Bedrock) only if not open-source.

        :return: True if clients initialized successfully or open-source, False otherwise
        """
        if self.service_type == 'open-source':
            return True  # No AWS clients needed

        try:
            boto_config = Config(
                region_name=self.aws_region,
                signature_version='v4',
                retries={'max_attempts': 10, 'mode': 'standard'}
            )
            if self.is_serverless:
                self.aoss_client = boto3.client(self.SERVICE_AOSS, config=boto_config)
            self.bedrock_client = boto3.client(self.SERVICE_BEDROCK, region_name=self.aws_region)
            
            time.sleep(7)  # Wait for clients to initialize
            print(f"{Fore.GREEN}AWS clients initialized successfully.{Style.RESET_ALL}\n")
            return True
        except Exception as e:
            print(f"{Fore.RED}Failed to initialize AWS clients: {e}{Style.RESET_ALL}")
            return False

    def get_opensearch_domain_name(self):
        """
        Extract the domain name from the OpenSearch endpoint URL.

        :return: Domain name if extraction is successful, None otherwise
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
                print(f"Extracted domain name: {domain_name}\n")
                return domain_name
        return None

    @staticmethod
    def get_opensearch_domain_info(region, domain_name):
        """
        Retrieve the OpenSearch domain endpoint and ARN based on the domain name and region.

        :param region: AWS region
        :param domain_name: Name of the OpenSearch domain
        :return: Tuple of (domain_endpoint, domain_arn) if successful, (None, None) otherwise
        """
        try:
            client = boto3.client('opensearch', region_name=region)
            response = client.describe_domain(DomainName=domain_name)
            domain_status = response['DomainStatus']
            domain_endpoint = domain_status.get('Endpoint') or domain_status.get('Endpoints', {}).get('vpc')
            domain_arn = domain_status['ARN']
            return domain_endpoint, domain_arn
        except Exception as e:
            print(f"{Fore.RED}Error retrieving OpenSearch domain info: {e}{Style.RESET_ALL}")
            return None, None

    def initialize_opensearch_client(self):
        if not self.opensearch_endpoint:
            print(f"{Fore.RED}OpenSearch endpoint not set. Please run setup first.{Style.RESET_ALL}\n")
            return False

        parsed_url = urlparse(self.opensearch_endpoint)
        host = parsed_url.hostname
        port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 9200)  # Default ports

        if self.service_type == 'serverless':
            credentials = boto3.Session().get_credentials()
            auth = AWSV4SignerAuth(credentials, self.aws_region, 'aoss')
        elif self.service_type == 'managed':
            if not self.opensearch_username or not self.opensearch_password:
                print(f"{Fore.RED}OpenSearch username or password not set. Please run setup first.{Style.RESET_ALL}\n")
                return False
            auth = (self.opensearch_username, self.opensearch_password)
        elif self.service_type == 'open-source':
            if self.opensearch_username and self.opensearch_password:
                auth = (self.opensearch_username, self.opensearch_password)
            else:
                auth = None  # No authentication
        else:
            print("Invalid service type. Please check your configuration.")
            return False

        use_ssl = parsed_url.scheme == 'https'
        verify_certs = False if use_ssl else True

        # Create an SSL context that does not verify certificates if needed
        if use_ssl and not verify_certs:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        else:
            ssl_context = None

        try:
            self.opensearch_client = OpenSearch(
                hosts=[{'host': host, 'port': port}],
                http_auth=auth,
                use_ssl=use_ssl,
                verify_certs=verify_certs,
                ssl_show_warn=False,          # Suppress SSL warnings
                ssl_context=ssl_context,      # Use the custom SSL context
                connection_class=RequestsHttpConnection,
                pool_maxsize=20
            )
            print(f"{Fore.GREEN}Initialized OpenSearch client with host: {host} and port: {port}{Style.RESET_ALL}\n")
            return True
        except Exception as ex:
            print(f"{Fore.RED}Error initializing OpenSearch client: {ex}{Style.RESET_ALL}\n")
            return False
        
    def get_knn_index_details(self):
        """
        Prompt user for KNN index details (embedding dimension, space type, and ef_construction).

        :return: Tuple of (embedding_dimension, space_type, ef_construction)
        """
        dimension_input = input("Press Enter to use the default embedding size (768), or type a custom size: ").strip()
        
        if dimension_input == "":
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
        space_choice = input("Enter your choice (1-3), or press Enter for default (L2): ").strip()

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
        ef_construction_input = input("\nPress Enter to use the default ef_construction value (512), or type a custom value: ").strip()
        
        if ef_construction_input == "":
            ef_construction = 512
        else:
            try:
                ef_construction = int(ef_construction_input)
            except ValueError:
                print("Invalid input. Using default ef_construction of 512.")
                ef_construction = 512

        print(f"ef_construction set to: {ef_construction}\n")

        return embedding_dimension, space_type, ef_construction
    def save_config(self, config):
        """
        Save configuration to the config file.

        :param config: Dictionary of configuration parameters
        """
        parser = configparser.ConfigParser()
        parser['DEFAULT'] = config
        config_path = os.path.abspath(self.CONFIG_FILE)
        with open(self.CONFIG_FILE, 'w') as f:
            parser.write(f)
        # Removed duplicate message
        # Only one message is printed where this method is called
    def create_index(self, embedding_dimension, space_type, ef_construction):
        """
        Create the KNN index in OpenSearch.

        :param embedding_dimension: Dimension of the embedding vectors
        :param space_type: Type of space for KNN
        :param ef_construction: ef_construction parameter for KNN
        """
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
            print(f"\n{Fore.GREEN}KNN index '{self.index_name}' created successfully with dimension {embedding_dimension}, space type {space_type}, and ef_construction {ef_construction}.{Style.RESET_ALL}\n")
        except Exception as e:
            if 'resource_already_exists_exception' in str(e).lower():
                print(f"\n{Fore.YELLOW}Index '{self.index_name}' already exists.{Style.RESET_ALL}\n")
            else:
                print(f"\n{Fore.RED}Error creating index '{self.index_name}': {e}{Style.RESET_ALL}\n")

    def verify_and_create_index(self, embedding_dimension, space_type, ef_construction):
        """
        Verify if the index exists; if not, create it.

        :param embedding_dimension: Dimension of the embedding vectors
        :param space_type: Type of space for KNN
        :param ef_construction: ef_construction parameter for KNN
        :return: True if index exists or is created successfully, False otherwise
        """
        try:
            print(f"Attempting to verify index '{self.index_name}'...")
            index_exists = self.opensearch_client.indices.exists(index=self.index_name)
            if index_exists:
                print(f"{Fore.GREEN}KNN index '{self.index_name}' already exists.{Style.RESET_ALL}\n")
            else:
                print(f"{Fore.YELLOW}Index '{self.index_name}' does not exist. Attempting to create...{Style.RESET_ALL}\n")
                self.create_index(embedding_dimension, space_type, ef_construction)
            return True
        except Exception as ex:
            print(f"{Fore.RED}Error verifying or creating index: {ex}{Style.RESET_ALL}")
            print(f"OpenSearch client config: {self.opensearch_client.transport.hosts}\n")
            return False

    def get_truncated_name(self, base_name, max_length=32):
        """
        Truncate a name to fit within a specified length.

        :param base_name: Original name
        :param max_length: Maximum allowed length
        :return: Truncated name
        """
        if len(base_name) <= max_length:
            return base_name
        return base_name[:max_length-3] + "..."

    def setup_command(self):
        """
        Main setup command that orchestrates the entire setup process.
        """
        self.setup_configuration()

        if self.service_type != 'open-source' and not self.initialize_clients():
            print(f"\n{Fore.RED}Failed to initialize AWS clients. Setup incomplete.{Style.RESET_ALL}\n")
            return

        if self.service_type == 'serverless':

            pass
        elif self.service_type == 'managed':
            if not self.opensearch_endpoint:
                print(f"\n{Fore.RED}OpenSearch endpoint not set. Setup incomplete.{Style.RESET_ALL}\n")
                return
            else:
                self.opensearch_domain_name = self.get_opensearch_domain_name()
        elif self.service_type == 'open-source':
            # Open-source setup
            if not self.opensearch_endpoint:
                print(f"\n{Fore.RED}OpenSearch endpoint not set. Setup incomplete.{Style.RESET_ALL}\n")
                return
            else:
                self.opensearch_domain_name = None  # Not required for open-source

        # Initialize OpenSearch client
        if self.initialize_opensearch_client():

            # Prompt user to choose between creating a new index or using an existing one
            print("Do you want to create a new KNN index or use an existing one?")
            print("1. Create a new KNN index")
            print("2. Use an existing KNN index")
            index_choice = input("Enter your choice (1-2): ").strip()

            if index_choice == '1':
                # Proceed to create a new index
                self.index_name = input("\nEnter a name for your new KNN index in OpenSearch: ").strip()

                # Save the index name in the configuration
                self.config['index_name'] = self.index_name
                self.save_config(self.config)

                print("\nProceeding with index creation...\n")
                embedding_dimension, space_type, ef_construction = self.get_knn_index_details()

                if self.verify_and_create_index(embedding_dimension, space_type, ef_construction):
                    print(f"\n{Fore.GREEN}KNN index '{self.index_name}' created successfully.{Style.RESET_ALL}\n")
                    # Save index details to config
                    self.config['embedding_dimension'] = str(embedding_dimension)
                    self.config['space_type'] = space_type
                    self.config['ef_construction'] = str(ef_construction)
                    self.save_config(self.config)
                else:
                    print(f"\n{Fore.RED}Index creation failed. Please check your permissions and try again.{Style.RESET_ALL}\n")
                    return
            elif index_choice == '2':
                # Use existing index
                existing_index_name = input("\nEnter the name of your existing KNN index: ").strip()
                if not existing_index_name:
                    print(f"\n{Fore.RED}Index name cannot be empty. Aborting.{Style.RESET_ALL}\n")
                    return
                self.index_name = existing_index_name
                self.config['index_name'] = self.index_name
                self.save_config(self.config)
                # Load index details from config or prompt for them
                if 'embedding_dimension' in self.config and 'space_type' in self.config and 'ef_construction' in self.config:
                    try:
                        embedding_dimension = int(self.config['embedding_dimension'])
                        space_type = self.config['space_type']
                        ef_construction = int(self.config['ef_construction'])
                        print(f"\nUsing existing index '{self.index_name}' with embedding dimension {embedding_dimension}, space type '{space_type}', and ef_construction {ef_construction}.\n")
                    except ValueError:
                        print("\nInvalid index details in configuration. Prompting for details again.\n")
                        embedding_dimension, space_type, ef_construction = self.get_knn_index_details()
                        # Save index details to config
                        self.config['embedding_dimension'] = str(embedding_dimension)
                        self.config['space_type'] = space_type
                        self.config['ef_construction'] = str(ef_construction)
                        self.save_config(self.config)
                else:
                    print("\nIndex details not found in configuration. Prompting for details.\n")
                    embedding_dimension, space_type, ef_construction = self.get_knn_index_details()
                    # Save index details to config
                    self.config['embedding_dimension'] = str(embedding_dimension)
                    self.config['space_type'] = space_type
                    self.config['ef_construction'] = str(ef_construction)
                    self.save_config(self.config)
                # Verify that the index exists
                if not self.opensearch_client.indices.exists(index=self.index_name):
                    print(f"\n{Fore.RED}Index '{self.index_name}' does not exist in OpenSearch. Aborting.{Style.RESET_ALL}\n")
                    return
            else:
                print(f"\n{Fore.RED}Invalid choice. Please run setup again and select a valid option.{Style.RESET_ALL}\n")
                return

            # Proceed with model registration
            # Initialize ModelRegister now that OpenSearch client and domain name are available
            self.model_register = ModelRegister(
                self.config,
                self.opensearch_client,
                self.opensearch_domain_name
            )

            # Model Registration
            if self.service_type != 'open-source':
                # AWS-managed OpenSearch: Proceed with model registration
                self.model_register.prompt_model_registration()
            else:
                # Open-source OpenSearch: Provide instructions or automate model registration
                self.model_register.prompt_opensource_model_registration()
        else:
            print(f"\n{Fore.RED}Failed to initialize OpenSearch client. Setup incomplete.{Style.RESET_ALL}\n")