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


class TestSeriesValueCounts(TestData):
    def test_value_counts(self):
        pd_s = self.pd_flights()["Carrier"]
        oml_s = self.oml_flights()["Carrier"]

        pd_vc = pd_s.value_counts()
        oml_vc = oml_s.value_counts()

        assert_series_equal(pd_vc, oml_vc)

    def test_value_counts_size(self):
        pd_s = self.pd_flights()["Carrier"]
        oml_s = self.oml_flights()["Carrier"]

        pd_vc = pd_s.value_counts()[:1]
        oml_vc = oml_s.value_counts(os_size=1)

        assert_series_equal(pd_vc, oml_vc)

    def test_value_counts_keyerror(self):
        oml_f = self.oml_flights()
        with pytest.raises(KeyError):
            assert oml_f["not_a_column"].value_counts()

    def test_value_counts_dataframe(self):
        # value_counts() is a series method, should raise AttributeError if called on a DataFrame
        oml_f = self.oml_flights()
        with pytest.raises(AttributeError):
            assert oml_f.value_counts()

    def test_value_counts_non_int(self):
        oml_s = self.oml_flights()["Carrier"]
        with pytest.raises(TypeError):
            assert oml_s.value_counts(os_size="foo")

    def test_value_counts_non_positive_int(self):
        oml_s = self.oml_flights()["Carrier"]
        with pytest.raises(ValueError):
            assert oml_s.value_counts(os_size=-9)

    @pytest.mark.filterwarnings("ignore:Aggregations not supported")
    def test_value_counts_non_aggregatable(self):
        oml_s = self.oml_ecommerce()["customer_first_name"]
        pd_s = self.pd_ecommerce()["customer_first_name"]

        pd_vc = pd_s.value_counts().head(20).sort_index()
        oml_vc = oml_s.value_counts(os_size=20).sort_index()

        assert_series_equal(pd_vc, oml_vc)

        oml_s = self.oml_ecommerce()["customer_gender"]
        with pytest.raises(ValueError):
            assert oml_s.value_counts()
