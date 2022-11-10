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

from pandas.testing import assert_series_equal

from tests.common import TestData


class TestDataFrameNUnique(TestData):
    def test_flights_nunique(self):
        # Note pandas.nunique fails for dict columns (e.g. DestLocation)
        columns = [
            "AvgTicketPrice",
            "Cancelled",
            "Carrier",
            "Dest",
            "DestAirportID",
            "DestCityName",
        ]
        pd_flights = self.pd_flights()[columns]
        oml_flights = self.oml_flights()[columns]

        pd_flights.nunique()
        oml_flights.nunique()

        # TODO - OS is approximate counts so these aren't equal...
        # E[left]: [13059, 2, 4, 156, 156, 143]
        # E[right]: [13132, 2, 4, 156, 156, 143]
        # assert_series_equal(pd_nunique, oml_nunique)

    def test_ecommerce_nunique(self):
        columns = ["customer_first_name", "customer_last_name", "day_of_week_i"]
        pd_ecommerce = self.pd_ecommerce()[columns]
        oml_ecommerce = self.oml_ecommerce()[columns]

        pd_nunique = pd_ecommerce.nunique()
        oml_nunique = oml_ecommerce.nunique()

        assert_series_equal(pd_nunique, oml_nunique)
