import pytest

import os
from opensearchpy import OpenSearch
from sklearn.datasets import load_iris

@pytest.fixture
def opensearch_client():
    opensearch_host = os.environ.get("OPENSEARCH_HOST", "https://localhost:9200")
    opensearch_admin_user = os.environ.get("OPENSEARCH_ADMIN_USER", "admin")
    opensearch_admin_password = os.environ.get("OPENSEARCH_ADMIN_PASSWORD", "admin")
    client = OpenSearch(
        hosts=[opensearch_host],
        http_auth=(opensearch_admin_user, opensearch_admin_password),
        verify_certs=False,
    )
    yield client
    
    # tear down
    client.transport.close()
    

@pytest.fixture
def iris_index_client(opensearch_client: OpenSearch):
    index_name = "test__index__iris_data"
    index_mapping = {
        "mappings": {
            "properties": {
                "sepal_length": {"type": "float"},
                "sepal_width": {"type": "float"},
                "petal_length": {"type": "float"},
                "petal_width": {"type": "float"},
                "species": {"type": "keyword"}
            }
        }
    }
    if opensearch_client.indices.exists(index=index_name):
        opensearch_client.indices.delete(index=index_name)
    opensearch_client.indices.create(index=index_name, body=index_mapping)
    
    iris = load_iris()
    iris_data = iris.data
    iris_target = iris.target
    iris_species = [iris.target_names[i] for i in iris_target]
    
    actions = [
        {   '_index': index_name,
            "_source":{
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width,
                "species": species
            }
        }
        for (sepal_length, sepal_width, petal_length, petal_width), species in zip(iris_data, iris_species)
    ]
    
    
    opensearch_client.bulk(actions)
    
    yield opensearch_client, index_name
    
    opensearch_client.indices.delete(index=index_name)
    