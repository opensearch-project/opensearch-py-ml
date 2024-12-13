# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from urllib.parse import urlparse

import boto3
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection
from opensearchpy import exceptions as opensearch_exceptions
from opensearchpy import helpers as opensearch_helpers


class OpenSearchConnector:
    """
    Manages the connection and interactions with the OpenSearch cluster.
    Provides methods to initialize the client, create indices, perform bulk indexing, and execute searches.
    """

    def __init__(self, config):
        """
        Initialize the OpenSearchConnector with the provided configuration.

        :param config: Dictionary containing configuration parameters.
        """
        # Store the configuration
        self.config = config
        self.opensearch_client = None
        self.aws_region = config.get("region")
        self.index_name = config.get("index_name")
        self.is_serverless = config.get("is_serverless", "False") == "True"
        self.opensearch_endpoint = config.get("opensearch_endpoint")
        self.opensearch_username = config.get("opensearch_username")
        self.opensearch_password = config.get("opensearch_password")
        self.service_type = config.get("service_type")

    def initialize_opensearch_client(self) -> bool:
        """
        Initialize the OpenSearch client based on the service type and configuration.

        :return: True if the client is initialized successfully, False otherwise.
        """
        # Check if the OpenSearch endpoint is provided
        if not self.opensearch_endpoint:
            print("OpenSearch endpoint not set. Please run setup first.")
            return False

        # Parse the OpenSearch endpoint URL
        parsed_url = urlparse(self.opensearch_endpoint)
        host = parsed_url.hostname
        port = parsed_url.port or (
            443 if parsed_url.scheme == "https" else 9200
        )  # Default ports

        # Determine the authentication method based on the service type
        if self.service_type == "serverless":
            # Use AWS V4 Signer Authentication for serverless
            credentials = boto3.Session().get_credentials()
            auth = AWSV4SignerAuth(credentials, self.aws_region, "aoss")
        elif self.service_type == "managed":
            # Use basic authentication for managed services
            if not self.opensearch_username or not self.opensearch_password:
                print(
                    "OpenSearch username or password not set. Please run setup first."
                )
                return False
            auth = (self.opensearch_username, self.opensearch_password)
        elif self.service_type == "open-source":
            # Use basic authentication if credentials are provided, else no authentication
            if self.opensearch_username and self.opensearch_password:
                auth = (self.opensearch_username, self.opensearch_password)
            else:
                auth = None  # No authentication
        else:
            # Invalid service type
            print("Invalid service type. Please check your configuration.")
            return False

        # Determine SSL settings based on the endpoint scheme
        use_ssl = parsed_url.scheme == "https"
        verify_certs = (
            True  # Always verify certificates unless you have a specific reason not to
        )

        try:
            # Initialize the OpenSearch client
            self.opensearch_client = OpenSearch(
                hosts=[{"host": host, "port": port}],
                http_auth=auth,
                use_ssl=use_ssl,
                verify_certs=verify_certs,
                ssl_show_warn=False,  # Suppress SSL warnings
                # ssl_context=ssl_context,      # Not needed unless you have custom certificates
                connection_class=RequestsHttpConnection,
                pool_maxsize=20,
            )
            print(f"Initialized OpenSearch client with host: {host} and port: {port}")
            return True
        except Exception as ex:
            # Handle initialization errors
            print(f"Error initializing OpenSearch client: {ex}")
            return False

    def create_index(
        self,
        embedding_dimension: int,
        space_type: str,
        ef_construction: int,
        number_of_shards: int,
        number_of_replicas: int,
        passage_text_field: str,
        passage_chunk_field: str,
        embedding_field: str,
    ):
        """
        Create a KNN index in OpenSearch with the specified parameters.

        :param embedding_dimension: The dimension of the embedding vectors.
        :param space_type: The space type for the KNN algorithm (e.g., 'cosinesimil', 'l2').
        :param ef_construction: ef_construction parameter for KNN
        :param number_of_shards: Number of shards for the index
        :param number_of_replicas: Number of replicas for the index
        :param nominee_text_field: Field name for nominee text
        """
        # Define the index mapping and settings
        index_body = {
            "mappings": {
                "properties": {
                    passage_text_field: {"type": "text"},
                    passage_chunk_field: {"type": "text"},
                    embedding_field: {
                        "type": "nested",
                        "properties": {
                            "knn": {
                                "type": "knn_vector",
                                "dimension": embedding_dimension,
                            }
                        },
                    },
                }
            },
            "settings": {
                "index": {
                    "number_of_shards": number_of_shards,
                    "number_of_replicas": number_of_replicas,
                    "knn.algo_param": {"ef_search": 512},
                    "knn": True,
                }
            },
        }

        try:
            # Attempt to create the index
            self.opensearch_client.indices.create(
                index=self.index_name, body=index_body
            )
            print(
                f"KNN index '{self.index_name}' created successfully with the following settings:"
            )
            print(f"Embedding Dimension: {embedding_dimension}")
            print(f"Space Type: {space_type}")
            print(f"ef_construction: {ef_construction}")
            print(f"Number of Shards: {number_of_shards}")
            print(f"Number of Replicas: {number_of_replicas}")
            print(f"Text Field: '{passage_text_field}'")
            print(f"Passage Chunk Field: '{passage_chunk_field}'")
            print(f"Embedding Field: '{embedding_field}'")
        except opensearch_exceptions.RequestError as e:
            # Handle cases where the index already exists
            if "resource_already_exists_exception" in str(e).lower():
                print(f"Index '{self.index_name}' already exists.")
            else:
                # Handle other index creation errors
                print(f"Error creating index '{self.index_name}': {e}")

    def verify_and_create_index(
        self,
        embedding_dimension: int,
        space_type: str,
        ef_construction: int,
        number_of_shards: int,
        number_of_replicas: int,
        passage_text_field: str,
        passage_chunk_field: str,
        embedding_field: str,
    ) -> bool:
        """
        Verify if the index exists; if not, create it.

        :param embedding_dimension: The dimension of the embedding vectors.
        :param space_type: The space type for the KNN algorithm.
        :param ef_construction: ef_construction parameter for KNN
        :param number_of_shards: Number of shards for the index
        :param number_of_replicas: Number of replicas for the index
        :param nominee_text_field: Field name for nominee text
        :return: True if the index exists or was successfully created, False otherwise.
        """
        try:
            # Check if the index already exists
            index_exists = self.opensearch_client.indices.exists(index=self.index_name)
            if index_exists:
                print(f"KNN index '{self.index_name}' already exists.")
            else:
                # Create the index if it doesn't exist
                self.create_index(
                    embedding_dimension,
                    space_type,
                    ef_construction,
                    number_of_shards,
                    number_of_replicas,
                    passage_text_field,
                    passage_chunk_field,
                    embedding_field,
                )
            return True
        except Exception as ex:
            # Handle errors during verification or creation
            print(f"Error verifying or creating index: {ex}")
            return False

    def bulk_index(self, actions: list) -> tuple:
        """
        Perform bulk indexing of documents into OpenSearch.

        :param actions: List of indexing actions to perform.
        :return: A tuple containing the number of successfully indexed documents and the number of failures.
        """
        try:
            # Execute bulk indexing using OpenSearch helpers
            success_count, error_info = opensearch_helpers.bulk(
                self.opensearch_client, actions
            )
            error_count = len(error_info)
            print(
                f"Indexed {success_count} documents successfully. Failed to index {error_count} documents."
            )
            return success_count, error_count
        except Exception as e:
            # Handle bulk indexing errors
            print(f"Error during bulk indexing: {e}")
            return 0, len(actions)

    def search(self, query_text: str, model_id: str, k: int = 5) -> list:
        """
        Perform a neural search based on the query text and model ID.
        """
        embedding_field = self.config.get("embedding_field", "passage_embedding")

        try:
            # Execute the search query using nested query
            response = self.opensearch_client.search(
                index=self.index_name,
                body={
                    "size": k,
                    "_source": ["passage_chunk"],
                    "query": {
                        "nested": {
                            "score_mode": "max",
                            "path": embedding_field,
                            "query": {
                                "neural": {
                                    f"{embedding_field}.knn": {
                                        "query_text": query_text,
                                        "model_id": model_id,
                                        "k": k,
                                    }
                                }
                            },
                        }
                    },
                },
            )
            return response["hits"]["hits"]
        except Exception as e:
            # Handle search errors
            print(f"Error during search: {e}")
            return []

    def check_connection(self) -> bool:
        """
        Check the connection to the OpenSearch cluster.

        :return: True if the connection is successful, False otherwise.
        """
        try:
            # Retrieve cluster information to verify connection
            self.opensearch_client.info()
            return True
        except Exception as e:
            # Handle connection errors
            print(f"Error connecting to OpenSearch: {e}")
            return False

    def search_by_vector(self, vector: list, k: int = 5) -> list:
        """
        Perform a vector-based search using the provided embedding vector.

        :param vector: The embedding vector to search with.
        :param k: The number of top results to retrieve.
        :return: A list of search hits.
        """
        # Retrieve field names from the config
        embedding_field = self.config.get("embedding_field", "passage_embedding")
        passage_text_field = self.config.get("passage_text_field", "passage_text")
        passage_chunk_field = self.config.get("passage_chunk_field", "passage_chunk")

        try:
            # Execute the KNN search query using the correct field name
            response = self.opensearch_client.search(
                index=self.index_name,
                body={
                    "size": k,
                    "_source": [passage_text_field, passage_chunk_field],
                    "query": {
                        "nested": {
                            "path": embedding_field,
                            "query": {
                                "knn": {
                                    f"{embedding_field}.knn": {"vector": vector, "k": k}
                                }
                            },
                        }
                    },
                },
            )
            return response["hits"]["hits"]
        except Exception as e:
            # Handle search errors
            print(f"Error during search: {e}")
            return []
