from opensearchpy import OpenSearch
from opensearchpy.exceptions import RequestError
from .common import OPENSEARCH_TEST_CLIENT, MODEL_GROUP_PREFIX
import time




def delete_test_model_groups(os: OpenSearch):
    model_group_query = {"query": {"match_phrase_prefix": {"name": MODEL_GROUP_PREFIX}}}

    try:
        result = os.transport.perform_request(
            method="GET", url="/_plugins/_ml/model_groups/_search", body=model_group_query
        )
        
        for each in result["hits"]["hits"]:
            try:
                os.transport.perform_request(
                    method="DELETE", url=f"/_plugins/_ml/model_groups/{each['_id']}"
                )
                time.sleep(0.2)
            except Exception as ex:
                print(f"Failed to delete model group id: {each['_id']}")
    except RequestError as ex:
        print(f"Failed due to request error: {ex}")
    except Exception as ex:
        print(f"Unexpected Model group deletion failure due to: {ex}")
