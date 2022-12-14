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

from opensearch_py_ml.field_mappings import FieldMappings
from tests import FLIGHTS_INDEX_NAME, OPENSEARCH_TEST_CLIENT
from tests.common import TestData


class TestRename(TestData):
    def test_single_rename(self):
        oml_field_mappings = FieldMappings(
            client=OPENSEARCH_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME
        )

        pd_flights_column_series = self.pd_flights().columns.to_series()

        assert (
            pd_flights_column_series.index.to_list() == oml_field_mappings.display_names
        )

        renames = {"DestWeather": "renamed_DestWeather"}

        # inplace rename
        oml_field_mappings.rename(renames)

        assert (
            pd_flights_column_series.rename(renames).index.to_list()
            == oml_field_mappings.display_names
        )

        get_renames = oml_field_mappings.get_renames()

        assert renames == get_renames

    def test_non_exists_rename(self):
        oml_field_mappings = FieldMappings(
            client=OPENSEARCH_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME
        )

        pd_flights_column_series = self.pd_flights().columns.to_series()

        assert (
            pd_flights_column_series.index.to_list() == oml_field_mappings.display_names
        )

        renames = {"unknown": "renamed_unknown"}

        # inplace rename - in this case it has no effect
        oml_field_mappings.rename(renames)

        assert (
            pd_flights_column_series.index.to_list() == oml_field_mappings.display_names
        )

        get_renames = oml_field_mappings.get_renames()

        assert not get_renames

    def test_exists_and_non_exists_rename(self):
        oml_field_mappings = FieldMappings(
            client=OPENSEARCH_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME
        )

        pd_flights_column_series = self.pd_flights().columns.to_series()

        assert (
            pd_flights_column_series.index.to_list() == oml_field_mappings.display_names
        )

        renames = {
            "unknown": "renamed_unknown",
            "DestWeather": "renamed_DestWeather",
            "unknown2": "renamed_unknown2",
            "Carrier": "renamed_Carrier",
        }

        # inplace rename - only real names get renamed
        oml_field_mappings.rename(renames)

        assert (
            pd_flights_column_series.rename(renames).index.to_list()
            == oml_field_mappings.display_names
        )

        get_renames = oml_field_mappings.get_renames()

        assert {
            "Carrier": "renamed_Carrier",
            "DestWeather": "renamed_DestWeather",
        } == get_renames

    def test_multi_rename(self):
        oml_field_mappings = FieldMappings(
            client=OPENSEARCH_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME
        )

        pd_flights_column_series = self.pd_flights().columns.to_series()

        assert (
            pd_flights_column_series.index.to_list() == oml_field_mappings.display_names
        )

        renames = {
            "DestWeather": "renamed_DestWeather",
            "renamed_DestWeather": "renamed_renamed_DestWeather",
        }

        # inplace rename - only first rename gets renamed
        oml_field_mappings.rename(renames)

        assert (
            pd_flights_column_series.rename(renames).index.to_list()
            == oml_field_mappings.display_names
        )

        get_renames = oml_field_mappings.get_renames()

        assert {"DestWeather": "renamed_DestWeather"} == get_renames
