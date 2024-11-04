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
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from opensearch_py_ml.constants import STANDARD_DEVIATION, VARIANCE
from tests.common import TestData


class TestDataFrameAggs(TestData):
    def test_basic_aggs(self):
        pd_flights = self.pd_flights()
        oml_flights = self.oml_flights()

        pd_sum_min = pd_flights.select_dtypes(include=[np.number]).agg(["sum", "min"])
        oml_sum_min = oml_flights.select_dtypes(include=[np.number]).agg(
            ["sum", "min"], numeric_only=True
        )

        # Opensearch_py_ml returns all float values for all metric aggs, pandas can return int
        # TODO - investigate this more
        pd_sum_min = pd_sum_min.astype("float64")
        assert_frame_equal(pd_sum_min, oml_sum_min, check_exact=False)

        pd_sum_min_std = pd_flights.select_dtypes(include=[np.number]).agg(
            ["sum", "min", STANDARD_DEVIATION]
        )
        oml_sum_min_std = oml_flights.select_dtypes(include=[np.number]).agg(
            ["sum", "min", STANDARD_DEVIATION], numeric_only=True
        )

        print(pd_sum_min_std.dtypes)
        print(oml_sum_min_std.dtypes)

        assert_frame_equal(
            pd_sum_min_std, oml_sum_min_std, check_exact=False, rtol=True
        )

    def test_terms_aggs(self):
        pd_flights = self.pd_flights()
        oml_flights = self.oml_flights()

        pd_sum_min = pd_flights.select_dtypes(include=[np.number]).agg(["sum", "min"])
        oml_sum_min = oml_flights.select_dtypes(include=[np.number]).agg(
            ["sum", "min"], numeric_only=True
        )

        # Eland returns all float values for all metric aggs, pandas can return int
        # TODO - investigate this more
        pd_sum_min = pd_sum_min.astype("float64")
        assert_frame_equal(pd_sum_min, oml_sum_min, check_exact=False)

        pd_sum_min_std = pd_flights.select_dtypes(include=[np.number]).agg(
            ["sum", "min", STANDARD_DEVIATION]
        )
        oml_sum_min_std = oml_flights.select_dtypes(include=[np.number]).agg(
            ["sum", "min", STANDARD_DEVIATION], numeric_only=True
        )

        print(pd_sum_min_std.dtypes)
        print(oml_sum_min_std.dtypes)

        assert_frame_equal(
            pd_sum_min_std, oml_sum_min_std, check_exact=False, rtol=True
        )

    def test_aggs_median_var(self):
        pd_ecommerce = self.pd_ecommerce()
        oml_ecommerce = self.oml_ecommerce()

        pd_aggs = pd_ecommerce[
            ["taxful_total_price", "taxless_total_price", "total_quantity"]
        ].agg(["median", VARIANCE])
        oml_aggs = oml_ecommerce[
            ["taxful_total_price", "taxless_total_price", "total_quantity"]
        ].agg(["median", VARIANCE], numeric_only=True)

        print(pd_aggs, pd_aggs.dtypes)
        print(oml_aggs, oml_aggs.dtypes)

        # Eland returns all float values for all metric aggs, pandas can return int
        # TODO - investigate this more
        pd_aggs = pd_aggs.astype("float64")
        assert_frame_equal(pd_aggs, oml_aggs, check_exact=False, rtol=2)

    # If Aggregate is given a string then series is returned.
    @pytest.mark.parametrize("agg", ["mean", "min", "max"])
    def test_terms_aggs_series(self, agg):
        pd_flights = self.pd_flights()
        oml_flights = self.oml_flights()

        pd_sum_min_std = pd_flights.select_dtypes(include=[np.number]).agg(agg)
        oml_sum_min_std = oml_flights.select_dtypes(include=[np.number]).agg(
            agg, numeric_only=True
        )

        assert_series_equal(pd_sum_min_std, oml_sum_min_std)

    def test_terms_aggs_series_with_single_list_agg(self):
        # aggs list with single agg should return dataframe.
        pd_flights = self.pd_flights()
        oml_flights = self.oml_flights()

        pd_sum_min = pd_flights.select_dtypes(include=[np.number]).agg(["mean"])
        oml_sum_min = oml_flights.select_dtypes(include=[np.number]).agg(
            ["mean"], numeric_only=True
        )

        assert_frame_equal(pd_sum_min, oml_sum_min)

    # If Wrong Aggregate value is given.
    def test_terms_wrongaggs(self):
        oml_flights = self.oml_flights()[["FlightDelayMin"]]

        match = "('abc', ' not currently implemented')"
        with pytest.raises(NotImplementedError, match=match):
            oml_flights.select_dtypes(include=[np.number]).agg("abc")
