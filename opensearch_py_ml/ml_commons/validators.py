# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

""" Module for validating model access control parameters """

from typing import List, Optional

ACCESS_MODES = ["public", "private", "restricted"]

NoneType = type(None)


def _validate_model_group_name(name: str):
    if not name or not isinstance(name, str):
        raise ValueError("name is required and needs to be a string")


def _validate_model_group_description(description: Optional[str]):
    if not isinstance(description, (NoneType, str)):
        raise ValueError("description needs to be a string")


def _validate_model_group_access_mode(access_mode: Optional[str]):
    if access_mode is None:
        return
    if access_mode not in ACCESS_MODES:
        raise ValueError(f"access_mode can must be in {ACCESS_MODES} or None")


def _validate_model_group_backend_roles(backend_roles: Optional[List[str]]):
    if not isinstance(backend_roles, (NoneType, list)):
        raise ValueError("backend_roles should either be None or a list of roles names")


def _validate_model_group_add_all_backend_roles(add_all_backend_roles: Optional[bool]):
    if not isinstance(add_all_backend_roles, (NoneType, bool)):
        raise ValueError("add_all_backend_roles should be a boolean")


def _validate_model_group_query(query: dict, operation: Optional[str] = None):
    if not isinstance(query, dict):
        raise ValueError("query needs to be a dictionary")

    if operation and not isinstance(operation, str):
        raise ValueError("operation needs to be a string")


def validate_create_model_group_parameters(
    name: str,
    description: Optional[str] = None,
    access_mode: Optional[str] = "private",
    backend_roles: Optional[List[str]] = None,
    add_all_backend_roles: Optional[bool] = False,
):
    _validate_model_group_name(name)
    _validate_model_group_description(description)
    _validate_model_group_access_mode(access_mode)
    _validate_model_group_backend_roles(backend_roles)
    _validate_model_group_add_all_backend_roles(add_all_backend_roles)

    if access_mode == "restricted":
        if not backend_roles and not add_all_backend_roles:
            raise ValueError(
                "You must specify either backend_roles or add_all_backend_roles=True for restricted access_mode"
            )

        if backend_roles and add_all_backend_roles:
            raise ValueError(
                "You cannot specify both backend_roles and add_all_backend_roles=True at the same time"
            )

    elif access_mode == "private":
        if backend_roles or add_all_backend_roles:
            raise ValueError(
                "You must not specify backend_roles or add_all_backend_roles=True for a private model group"
            )


def validate_update_model_group_parameters(update_query: dict, model_group_id: str):
    if not isinstance(model_group_id, str):
        raise ValueError("Invalid model_group_id. model_group_id needs to be a string")

    if not isinstance(update_query, dict):
        raise ValueError("Invalid update_query. update_query needs to be a dictionary")


def validate_delete_model_group_parameters(model_group_id: str):
    if not isinstance(model_group_id, str):
        raise ValueError("Invalid model_group_id. model_group_id needs to be a string")


def validate_search_model_group_parameters(query: dict):
    _validate_model_group_query(query)


def validate_profile_input(path_parameter, payload):
    if path_parameter is not None and not isinstance(path_parameter, str):
        raise ValueError("path_parameter needs to be a string or None")

    if payload is not None and not isinstance(payload, dict):
        raise ValueError("payload needs to be a dictionary or None")
