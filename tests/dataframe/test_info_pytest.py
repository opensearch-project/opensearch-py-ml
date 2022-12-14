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
from io import StringIO

import opensearch_py_ml as oml
from tests import OPENSEARCH_TEST_CLIENT
from tests.common import TestData


class TestDataFrameInfo(TestData):
    def test_flights_info(self):
        oml_flights = self.oml_flights()
        pd_flights = self.pd_flights()

        oml_buf = StringIO()
        pd_buf = StringIO()

        # Ignore memory_usage and first line (class name)
        oml_flights.info(buf=oml_buf, memory_usage=False)
        pd_flights.info(buf=pd_buf, memory_usage=False)

        oml_buf_lines = oml_buf.getvalue().split("\n")
        pd_buf_lines = pd_buf.getvalue().split("\n")

        assert pd_buf_lines[1:] == oml_buf_lines[1:]

        # NOTE: info does not work on truncated data frames (e.g. head/tail) TODO

        print(self.oml_ecommerce().info())

    def test_empty_info(self):
        mapping = {"mappings": {"properties": {}}}

        for i in range(0, 10):
            field_name = "field_name_" + str(i)
            mapping["mappings"]["properties"][field_name] = {"type": "float"}

        OPENSEARCH_TEST_CLIENT.indices.delete(index="empty_index", ignore=[400, 404])
        OPENSEARCH_TEST_CLIENT.indices.create(index="empty_index", body=mapping)

        oml_df = oml.DataFrame(OPENSEARCH_TEST_CLIENT, "empty_index")
        oml_df.info()

        OPENSEARCH_TEST_CLIENT.indices.delete(index="empty_index")
