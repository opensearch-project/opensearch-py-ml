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

from tests.common import TestData, assert_pandas_opensearch_py_ml_frame_equal


class TestDataFrameSelectDTypes(TestData):
    def test_select_dtypes_include_number(self):
        oml_flights = self.oml_flights()
        pd_flights = self.pd_flights()

        oml_flights_numeric = oml_flights.select_dtypes(include=[np.number])
        pd_flights_numeric = pd_flights.select_dtypes(include=[np.number])

        assert_pandas_opensearch_py_ml_frame_equal(
            pd_flights_numeric.head(103), oml_flights_numeric.head(103)
        )

    def test_select_dtypes_exclude_number(self):
        oml_flights = self.oml_flights()
        pd_flights = self.pd_flights()

        oml_flights_non_numeric = oml_flights.select_dtypes(exclude=[np.number])
        pd_flights_non_numeric = pd_flights.select_dtypes(exclude=[np.number])

        assert_pandas_opensearch_py_ml_frame_equal(
            pd_flights_non_numeric.head(103), oml_flights_non_numeric.head(103)
        )
