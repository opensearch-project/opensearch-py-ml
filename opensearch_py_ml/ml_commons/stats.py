# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Module for validating Stats API parameters """


def validate_stats_input(node_id, stat_id, payload):
    if payload:
        if node_id or stat_id:
            raise ValueError(
                "Stats API does not accept node_id or stat_id with payload"
            )
    if payload is not None and not isinstance(payload, dict):
        raise ValueError("payload needs to be a dictionary or None")
    if node_id is not None and not isinstance(node_id, str):
        raise ValueError("node_id needs to be a string or None")
    if stat_id is not None and not isinstance(stat_id, str):
        raise ValueError("stat_id needs to be a string or None")
