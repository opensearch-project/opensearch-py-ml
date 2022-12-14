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
from pandas.testing import assert_series_equal

from opensearch_py_ml.field_mappings import FieldMappings
from tests import FLIGHTS_INDEX_NAME, FLIGHTS_MAPPING
from tests.common import OPENSEARCH_TEST_CLIENT, TestData


class TestFieldNamePDDType(TestData):
    def test_all_formats(self):
        oml_field_mappings = FieldMappings(
            client=OPENSEARCH_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME
        )

        pd_flights = self.pd_flights()

        assert_series_equal(pd_flights.dtypes, oml_field_mappings.dtypes())

        for es_field_name in FLIGHTS_MAPPING["mappings"]["properties"].keys():
            pd_dtype = oml_field_mappings.field_name_pd_dtype(es_field_name)

            assert pd_flights[es_field_name].dtype == pd_dtype

    def test_non_existant(self):
        oml_field_mappings = FieldMappings(
            client=OPENSEARCH_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME
        )

        with pytest.raises(KeyError):
            oml_field_mappings.field_name_pd_dtype("unknown")
