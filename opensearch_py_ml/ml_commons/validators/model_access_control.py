# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

""" Module for validating model access control parameters """


ACCESS_MODES = ["public", "private", "restricted"]

NoneType = type(None)


def _validate_model_group_name(name):
    if not name or not isinstance(name, str):
        raise ValueError("name is required and needs to be a string")


def _validate_model_group_description(description):
    if not isinstance(description, (NoneType, str)):
        raise ValueError("description needs to be a string")


def _validate_model_group_access_mode(access_mode):
    if access_mode not in ACCESS_MODES:
        raise ValueError(f"access_mode must be in {ACCESS_MODES}")


def _validate_model_group_backend_roles(backend_roles):
    if not isinstance(backend_roles, (NoneType, list)):
        raise ValueError("backend_roles should either be None or a list of roles names")


def _validate_model_group_add_all_backend_roles(add_all_backend_roles):
    if not isinstance(add_all_backend_roles, bool):
        raise ValueError("add_all_backend_roles should be a boolean")


def _validate_model_group_query(query, operation=None):
    if not isinstance(query, dict):
        raise ValueError("query needs to be a dictionary")

    if operation and not isinstance(operation, str):
        raise ValueError("operation needs to be a string")


def validate_create_model_group_parameters(
    name,
    description,
    access_mode,
    backend_roles,
    add_all_backend_roles,
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


def validate_update_model_group_parameters(
    update_query, model_group_id, model_group_name
):
    if model_group_id and model_group_name:
        raise ValueError(
            "You cannot specify both model_group_id and model_group_name at the same time"
        )

    if not isinstance(model_group_id, (NoneType, str)):
        raise ValueError("Invalid model_group_id. model_group_id needs to be a string")

    if not isinstance(model_group_name, (NoneType, str)):
        raise ValueError(
            "Invalid model_group_name. model_group_name needs to be a string"
        )

    if not isinstance(update_query, dict):
        raise ValueError("Invalid update_query. update_query needs to be a dictionary")


def validate_delete_model_group_parameters(model_group_id, model_group_name):
    if model_group_id and model_group_name:
        raise ValueError(
            "You cannot specify both model_group_id and model_group_name at the same time"
        )

    if not isinstance(model_group_id, (NoneType, str)):
        raise ValueError("Invalid model_group_id. model_group_id needs to be a string")

    if not isinstance(model_group_name, (NoneType, str)):
        raise ValueError(
            "Invalid model_group_name. model_group_name needs to be a string"
        )


def validate_search_model_group_parameters(query):
    _validate_model_group_query(query)
