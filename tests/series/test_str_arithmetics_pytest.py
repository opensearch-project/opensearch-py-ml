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


class TestSeriesArithmetics(TestData):
    def test_invalid_add_num(self):
        with pytest.raises(TypeError):
            assert 2 + self.oml_ecommerce()["currency"]

        with pytest.raises(TypeError):
            assert self.oml_ecommerce()["currency"] + 2

        with pytest.raises(TypeError):
            assert (
                self.oml_ecommerce()["currency"]
                + self.oml_ecommerce()["total_quantity"]
            )

        with pytest.raises(TypeError):
            assert (
                self.oml_ecommerce()["total_quantity"]
                + self.oml_ecommerce()["currency"]
            )

    def test_ser_add_ser(self):
        omladd = (
            self.oml_ecommerce()["customer_first_name"]
            + self.oml_ecommerce()["customer_last_name"]
        )
        pdadd = (
            self.pd_ecommerce()["customer_first_name"]
            + self.pd_ecommerce()["customer_last_name"]
        )

        assert_pandas_opensearch_py_ml_series_equal(pdadd, omladd)

    def test_ser_add_str(self):
        omladd = self.oml_ecommerce()["customer_first_name"] + " is the first name."
        pdadd = self.pd_ecommerce()["customer_first_name"] + " is the first name."

        assert_pandas_opensearch_py_ml_series_equal(pdadd, omladd)

    def test_frame_add_str(self):
        pdadd = (
            self.pd_ecommerce()[["customer_first_name", "customer_last_name"]]
            + "_steve"
        )
        print(pdadd.head())
        print(pdadd.columns)

    def test_str_add_ser(self):
        omladd = "The last name is: " + self.oml_ecommerce()["customer_last_name"]
        pdadd = "The last name is: " + self.pd_ecommerce()["customer_last_name"]

        assert_pandas_opensearch_py_ml_series_equal(pdadd, omladd)

    def test_bad_str_add_ser(self):
        # TODO encode special characters better
        #      Elasticsearch accepts this, but it will cause problems
        omladd = " *" + self.oml_ecommerce()["customer_last_name"]
        pdadd = " *" + self.pd_ecommerce()["customer_last_name"]

        assert_pandas_opensearch_py_ml_series_equal(pdadd, omladd)

    def test_ser_add_str_add_ser(self):
        pdadd = (
            self.pd_ecommerce()["customer_first_name"]
            + " "
            + self.pd_ecommerce()["customer_last_name"]
        )
        omladd = (
            self.oml_ecommerce()["customer_first_name"]
            + " "
            + self.oml_ecommerce()["customer_last_name"]
        )

        assert_pandas_opensearch_py_ml_series_equal(pdadd, omladd)

    @pytest.mark.filterwarnings("ignore:Aggregations not supported")
    def test_non_aggregatable_add_str(self):
        with pytest.raises(ValueError):
            assert self.oml_ecommerce()["customer_gender"] + "is the gender"

    @pytest.mark.filterwarnings("ignore:Aggregations not supported")
    def teststr_add_non_aggregatable(self):
        with pytest.raises(ValueError):
            assert "The gender is: " + self.oml_ecommerce()["customer_gender"]

    @pytest.mark.filterwarnings("ignore:Aggregations not supported")
    def test_non_aggregatable_add_aggregatable(self):
        with pytest.raises(ValueError):
            assert (
                self.oml_ecommerce()["customer_gender"]
                + self.oml_ecommerce()["customer_first_name"]
            )

    @pytest.mark.filterwarnings("ignore:Aggregations not supported")
    def test_aggregatable_add_non_aggregatable(self):
        with pytest.raises(ValueError):
            assert (
                self.oml_ecommerce()["customer_first_name"]
                + self.oml_ecommerce()["customer_gender"]
            )
