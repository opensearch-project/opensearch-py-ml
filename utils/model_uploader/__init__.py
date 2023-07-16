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

# This file follows the cluster setup of tests
# in opensearch-py-ml/tests/__init__.py
import opensearchpy
from opensearchpy import OpenSearch

from opensearch_py_ml.common import os_version

OPENSEARCH_HOST = "https://instance:9200"
OPENSEARCH_ADMIN_USER, OPENSEARCH_ADMIN_PASSWORD = "admin", "admin"

# Define client to use in workflow
OPENSEARCH_TEST_CLIENT = OpenSearch(
    hosts=[OPENSEARCH_HOST],
    http_auth=(OPENSEARCH_ADMIN_USER, OPENSEARCH_ADMIN_PASSWORD),
    verify_certs=False,
)
# in github automated workflow, host url is: https://instance:9200
# in development, usually host url is: https://localhost:9200
# it's hard to remember changing the host url. So applied a try catch so that we don't have to keep change this config
try:
    OS_VERSION = os_version(OPENSEARCH_TEST_CLIENT)
except opensearchpy.exceptions.ConnectionError:
    OPENSEARCH_HOST = "https://localhost:9200"
    # Define client to use in tests
    OPENSEARCH_TEST_CLIENT = OpenSearch(
        hosts=[OPENSEARCH_HOST],
        http_auth=(OPENSEARCH_ADMIN_USER, OPENSEARCH_ADMIN_PASSWORD),
        verify_certs=False,
    )
    OS_VERSION = os_version(OPENSEARCH_TEST_CLIENT)
