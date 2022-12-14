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
import pandas as pd

import opensearch_py_ml as oml
from tests import FLIGHTS_INDEX_NAME, OPENSEARCH_TEST_CLIENT
from tests.common import TestData


class TestSeriesRepr(TestData):
    def test_repr_flights_carrier(self):
        pd_s = self.pd_flights()["Carrier"]
        oml_s = oml.Series(OPENSEARCH_TEST_CLIENT, FLIGHTS_INDEX_NAME, "Carrier")

        pd_repr = repr(pd_s)
        oml_repr = repr(oml_s)

        assert pd_repr == oml_repr

    def test_repr_flights_carrier_5(self):
        pd_s = self.pd_flights()["Carrier"].head(5)
        oml_s = oml.Series(OPENSEARCH_TEST_CLIENT, FLIGHTS_INDEX_NAME, "Carrier").head(
            5
        )

        pd_repr = repr(pd_s)
        oml_repr = repr(oml_s)

        assert pd_repr == oml_repr

    def test_repr_empty_series(self):
        pd_s = self.pd_flights()["Carrier"].head(0)
        oml_s = oml.Series(OPENSEARCH_TEST_CLIENT, FLIGHTS_INDEX_NAME, "Carrier").head(
            0
        )
        assert repr(pd_s) == repr(oml_s)

    def test_series_repr_pd_get_option_none(self):
        show_dimensions = pd.get_option("display.show_dimensions")
        show_rows = pd.get_option("display.max_rows")
        try:
            pd.set_option("display.show_dimensions", False)
            pd.set_option("display.max_rows", None)

            oml_flights = self.oml_flights()["Cancelled"].head(40).__repr__()
            pd_flights = self.pd_flights()["Cancelled"].head(40).__repr__()

            assert oml_flights == pd_flights
        finally:
            pd.set_option("display.max_rows", show_rows)
            pd.set_option("display.show_dimensions", show_dimensions)
