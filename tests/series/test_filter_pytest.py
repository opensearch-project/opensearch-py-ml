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

from tests.common import TestData, assert_pandas_opensearch_py_ml_series_equal


class TestSeriesFilter(TestData):
    def test_filter_arguments_mutually_exclusive(self):
        oml_flights_small = self.oml_flights_small()["FlightDelayType"]

        with pytest.raises(TypeError):
            oml_flights_small.filter(items=[], like="!", regex="!")
        with pytest.raises(TypeError):
            oml_flights_small.filter(items=[], regex="!")
        with pytest.raises(TypeError):
            oml_flights_small.filter(items=[], like="!")
        with pytest.raises(TypeError):
            oml_flights_small.filter(like="!", regex="!")
        with pytest.raises(TypeError):
            oml_flights_small.filter()

    def test_filter_columns_not_allowed_for_series(self):
        oml_flights_small = self.oml_flights_small()["FlightDelayType"]
        pd_flights_small = self.pd_flights_small()["FlightDelayType"]

        with pytest.raises(ValueError):
            oml_flights_small.filter(regex=".*", axis="columns")
        with pytest.raises(ValueError):
            oml_flights_small.filter(regex=".*", axis=1)
        with pytest.raises(ValueError):
            pd_flights_small.filter(regex=".*", axis="columns")
        with pytest.raises(ValueError):
            pd_flights_small.filter(regex=".*", axis=1)

    @pytest.mark.parametrize("items", [[], ["20"], [str(x) for x in range(30)]])
    def test_flights_filter_index_items(self, items):
        oml_flights_small = self.oml_flights_small()["FlightDelayType"]
        pd_flights_small = self.pd_flights_small()["FlightDelayType"]

        oml_ser = oml_flights_small.filter(items=items, axis=0)
        pd_ser = pd_flights_small.filter(items=items, axis=0)

        assert_pandas_opensearch_py_ml_series_equal(pd_ser, oml_ser)

    def test_flights_filter_index_like_and_regex(self):
        oml_flights_small = self.oml_flights_small()["FlightDelayType"]

        with pytest.raises(NotImplementedError):
            oml_flights_small.filter(like="2", axis=0)
        with pytest.raises(NotImplementedError):
            oml_flights_small.filter(regex="^2", axis=0)
