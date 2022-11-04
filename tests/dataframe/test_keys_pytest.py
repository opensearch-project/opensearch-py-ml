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

from pandas.testing import assert_index_equal

from tests.common import TestData


class TestDataFrameKeys(TestData):
    def test_ecommerce_keys(self):
        pd_ecommerce = self.pd_ecommerce()
        oml_ecommerce = self.oml_ecommerce()

        pd_keys = pd_ecommerce.keys()
        oml_keys = oml_ecommerce.keys()

        assert_index_equal(pd_keys, oml_keys)

    def test_flights_keys(self):
        pd_flights = self.pd_flights()
        oml_flights = self.oml_flights()

        pd_keys = pd_flights.keys()
        oml_keys = oml_flights.keys()

        assert_index_equal(pd_keys, oml_keys)
