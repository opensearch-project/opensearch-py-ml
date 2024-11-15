# opensearch_class.py

from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, exceptions as opensearch_exceptions
import boto3
from urllib.parse import urlparse
from opensearchpy import helpers as opensearch_helpers

class OpenSearchClass:
    def __init__(self, config):
        self.config = config
        self.opensearch_client = None
        self.aws_region = config.get('region')
        self.index_name = config.get('index_name')
        self.is_serverless = config.get('is_serverless', 'False') == 'True'
        self.opensearch_endpoint = config.get('opensearch_endpoint')
        self.opensearch_username = config.get('opensearch_username')
        self.opensearch_password = config.get('opensearch_password')

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
        except opensearch_exceptions.RequestError as e:
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

    def bulk_index(self, actions):
        try:
            success, failed = opensearch_helpers.bulk(self.opensearch_client, actions)
            print(f"Indexed {success} documents successfully. Failed to index {failed} documents.")
            return success, failed
        except Exception as e:
            print(f"Error during bulk indexing: {e}")
            return 0, len(actions)

    def search(self, vector, k=5):
        try:
            response = self.opensearch_client.search(
                index=self.index_name,
                body={
                    "size": k,
                    "_source": ["nominee_text"],
                    "query": {
                        "knn": {
                            "nominee_vector": {
                                "vector": vector,
                                "k": k
                            }
                        }
                    }
                }
            )
            return response['hits']['hits']
        except Exception as e:
            print(f"Error during search: {e}")
            return []
