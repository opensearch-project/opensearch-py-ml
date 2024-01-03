# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Module for validating Profile API parameters """


def validate_profile_input(path_parameter, payload):
    if path_parameter is not None and not isinstance(path_parameter, str):
        raise ValueError("path_parameter needs to be a string or None")

    if payload is not None and not isinstance(payload, dict):
        raise ValueError("payload needs to be a dictionary or None")
