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

from tests.common import TestData


class TestDataFrameIterrowsItertuples(TestData):
    def test_iterrows(self):
        oml_flights = self.oml_flights()
        pd_flights = self.pd_flights()

        oml_flights_iterrows = oml_flights.iterrows()
        pd_flights_iterrows = pd_flights.iterrows()

        for oml_index, oml_row in oml_flights_iterrows:
            pd_index, pd_row = next(pd_flights_iterrows)

            assert oml_index == pd_index
            assert_series_equal(oml_row, pd_row)

        # Assert that both are the same length and are exhausted.
        with pytest.raises(StopIteration):
            next(oml_flights_iterrows)
        with pytest.raises(StopIteration):
            next(pd_flights_iterrows)

    def test_itertuples(self):
        oml_flights = self.oml_flights()
        pd_flights = self.pd_flights()

        oml_flights_itertuples = list(oml_flights.itertuples(name=None))
        pd_flights_itertuples = list(pd_flights.itertuples(name=None))

        def assert_tuples_almost_equal(left, right):
            # Shim which uses pytest.approx() for floating point values inside tuples.
            assert len(left) == len(right)
            assert all(
                (lt == rt)  # Not floats? Use ==
                if not isinstance(lt, float) and not isinstance(rt, float)
                else (lt == pytest.approx(rt))  # If both are floats use pytest.approx()
                for lt, rt in zip(left, right)
            )

        for oml_tuple, pd_tuple in zip(oml_flights_itertuples, pd_flights_itertuples):
            assert_tuples_almost_equal(oml_tuple, pd_tuple)
