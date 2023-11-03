from opensearchpy import OpenSearch
from typing import List
from opensearch_py_ml.ml_commons.ml_common_utils import ML_BASE_URI
from opensearch_py_ml.exceptions.ml_commons import InvalidParameterError

ACCESS_MODES = ["public", "private", "restricted"]


class ModelAccessControl:
    API_ENDPOINT = "model_group"

    def __init__(self, os_client: OpenSearch):
        self.client = os_client

    def register_model_group(
        self,
        name: str,
        description: str,
        access_mode: str,
        backend_roles: List[str],
        add_all_backend_roles=False,
    ):
        # if not isinstance(name, str):
        #     raise TypeError(f"name should be of type string. Not {type(name)}")

        # if not isinstance(description, str):
        #     raise TypeError(
        #         f"description should be of type string. Not {type(description)}"
        #     )

        # if not isinstance(access_mode, str):
        #     if not access_mode in ACCESS_MODES:
        #         raise InvalidParameterError(
        #             "access_mode", "unexpected value", ACCESS_MODES, access_mode
        #         )
        #     if access_mode == "restricted":
        #         if add_all_backend_roles:
        #             if isinstance(backend_roles, list) and len(backend_roles) > 0:
        #                 raise Exception(
        #                     "If access_mode is restricted and add_all_backend_roles is True,"
        #                     "backend_roles should not be given"
        #                 )
        #         else:
        #             if isinstance(backend_roles, list) and len(backend_roles) == 0:
        #                 raise Exception(
        #                     "If access_mode is restricted and add_all_backend_roles is False,"
        #                     "backend_roles parameter is mandatory"
        #                 )

        body = {
            "name": name,
            "description": description,
            "access_mode": access_mode,
            "backend_roles": backend_roles,
            "add_all_backend_roles": add_all_backend_roles
        }

        return self.client.transport.perform_request(
            method="POST", url=f"{ML_BASE_URI}/{self.API_ENDPOINT}/_register", body=body
        )
    
    def update_model_group(
        self,
        name: str,
        description: str,
        access_mode: str,
        backend_roles: List[str],
        add_all_backend_roles=False,
    ):
        
        body = {
            "name": name,
            "description": description,
            "access_mode": access_mode,
            "backend_roles": backend_roles,
            "add_all_backend_roles": add_all_backend_roles
        }

        return self.client.transport.perform_request(
            method="PUT", url=f"{ML_BASE_URI}/{self.API_ENDPOINT}/_update", body=body
        )
    
    def delete_model_group(self, model_group_id):

        return self.client.transport.perform_request(
            method="DELETE", url=f"{ML_BASE_URI}/{self.API_ENDPOINT}/{model_group_id}"
        )
