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

# File called _pytest for PyCharm compatability

import pytest

import opensearch_py_ml as oml
from opensearch_py_ml.query_compiler import QueryCompiler
from tests import FLIGHTS_INDEX_NAME, OPENSEARCH_TEST_CLIENT


class TestDataFrameInit:
    def test_init(self):
        # Construct empty DataFrame (throws)
        with pytest.raises(ValueError):
            oml.DataFrame()

        # Construct invalid DataFrame (throws)
        with pytest.raises(ValueError):
            oml.DataFrame(os_client=OPENSEARCH_TEST_CLIENT)

        # Construct invalid DataFrame (throws)
        with pytest.raises(ValueError):
            oml.DataFrame(os_index_pattern=FLIGHTS_INDEX_NAME)

        # Good constructors
        oml.DataFrame(OPENSEARCH_TEST_CLIENT, FLIGHTS_INDEX_NAME)
        oml.DataFrame(
            os_client=OPENSEARCH_TEST_CLIENT, os_index_pattern=FLIGHTS_INDEX_NAME
        )

        qc = QueryCompiler(
            client=OPENSEARCH_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME
        )
        oml.DataFrame(_query_compiler=qc)
