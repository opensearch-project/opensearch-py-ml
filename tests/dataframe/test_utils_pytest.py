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

import numpy as np
import pandas as pd
import pytest

import opensearch_py_ml as oml
from opensearch_py_ml.field_mappings import FieldMappings
from tests.common import (
    OPENSEARCH_TEST_CLIENT,
    TestData,
    assert_pandas_opensearch_py_ml_frame_equal,
)


class TestDataFrameUtils(TestData):
    def test_generate_os_mappings(self):
        df = pd.DataFrame(
            data={
                "A": np.random.rand(3),
                "B": 1,
                "C": "foo",
                "D": pd.to_datetime("2019-01-02"),
                "E": [1.0, 2.0, 3.0],
                "F": False,
                "G": [1, 2, 3],
                "H": pd.to_datetime("2019-01-02", utc=True),
            },
            index=["0", "1", "2"],
        )

        expected_mappings = {
            "mappings": {
                "properties": {
                    "A": {"type": "double"},
                    "B": {"type": "long"},
                    "C": {"type": "keyword"},
                    "D": {"type": "date"},
                    "E": {"type": "double"},
                    "F": {"type": "boolean"},
                    "G": {"type": "long"},
                    "H": {"type": "date"},
                }
            }
        }

        mappings = FieldMappings._generate_os_mappings(df)

        assert expected_mappings == mappings

        # Now create index
        index_name = "eland_test_generate_es_mappings"

        oml_df = oml.pandas_to_opensearch(
            df,
            OPENSEARCH_TEST_CLIENT,
            index_name,
            os_if_exists="replace",
            os_refresh=True,
        )
        oml_df_head = oml_df.head()

        assert_pandas_opensearch_py_ml_frame_equal(df, oml_df_head)

        OPENSEARCH_TEST_CLIENT.indices.delete(index=index_name)

    def test_pandas_to_oml_ignore_index(self):
        df = pd.DataFrame(
            data={
                "A": np.random.rand(3),
                "B": 1,
                "C": "foo",
                "D": pd.to_datetime("2019-01-02"),
                "E": [1.0, 2.0, 3.0],
                "F": False,
                "G": [1, 2, 3],
                "H": "Long text",  # text
                "I": "52.36,4.83",  # geo point
            },
            index=["0", "1", "2"],
        )

        # Now create index
        index_name = "test_pandas_to_eland_ignore_index"

        oml_df = oml.pandas_to_opensearch(
            df,
            OPENSEARCH_TEST_CLIENT,
            index_name,
            os_if_exists="replace",
            os_refresh=True,
            use_pandas_index_for_os_ids=False,
            os_type_overrides={"H": "text", "I": "geo_point"},
        )

        # Check types
        expected_mapping = {
            "test_pandas_to_eland_ignore_index": {
                "mappings": {
                    "properties": {
                        "A": {"type": "double"},
                        "B": {"type": "long"},
                        "C": {"type": "keyword"},
                        "D": {"type": "date"},
                        "E": {"type": "double"},
                        "F": {"type": "boolean"},
                        "G": {"type": "long"},
                        "H": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword"}},
                        },
                        "I": {"type": "geo_point"},
                    }
                }
            }
        }

        mapping = OPENSEARCH_TEST_CLIENT.indices.get_mapping(index=index_name)

        assert expected_mapping == mapping

        # Convert back to pandas and compare with original
        pd_df = oml.opensearch_to_pandas(oml_df)

        # Compare values excluding index
        assert df.values.all() == pd_df.values.all()

        # Ensure that index is populated by ES.
        assert not (df.index == pd_df.index).any()

        OPENSEARCH_TEST_CLIENT.indices.delete(index=index_name)

    def tests_to_pandas_performance(self):
        # TODO quantify this
        oml.opensearch_to_pandas(self.oml_flights(), show_progress=True)

        # This test calls the same method so is redundant
        # assert_pandas_eland_frame_equal(pd_df, self.oml_flights())

    def test_es_type_override_error(self):
        df = self.pd_flights().filter(
            ["AvgTicketPrice", "Cancelled", "dayOfWeek", "timestamp", "DestCountry"]
        )

        index_name = "test_es_type_override"

        match = "'DistanceKilometers', 'DistanceMiles' column(s) not in given dataframe"
        with pytest.raises(KeyError) as e:
            oml.pandas_to_opensearch(
                df,
                OPENSEARCH_TEST_CLIENT,
                index_name,
                os_if_exists="replace",
                os_refresh=True,
                use_pandas_index_for_os_ids=False,
                os_type_overrides={
                    "AvgTicketPrice": "long",
                    "DistanceKilometers": "text",
                    "DistanceMiles": "text",
                },
            )
            assert str(e.value) == match
            OPENSEARCH_TEST_CLIENT.indices.delete(index=index_name)
