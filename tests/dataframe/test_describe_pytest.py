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

from pandas.testing import assert_frame_equal

from tests.common import TestData


class TestDataFrameDescribe(TestData):
    def test_flights_describe(self):
        pd_flights = self.pd_flights()
        oml_flights = self.oml_flights()

        pd_describe = pd_flights.describe().drop(["timestamp"], axis=1)
        # We remove bool columns to match pandas output
        oml_describe = oml_flights.describe().drop(
            ["Cancelled", "FlightDelay"], axis="columns"
        )

        assert_frame_equal(
            pd_describe.drop(["25%", "50%", "75%"], axis="index"),
            oml_describe.drop(["25%", "50%", "75%"], axis="index"),
            check_exact=False,
            rtol=True,
        )

        # TODO - this fails for percentile fields as OS aggregations are approximate
        #        if OS percentile agg uses
        #        "hdr": {
        #           "number_of_significant_value_digits": 3
        #         }
        #        this works

        # pd_ecommerce_describe = self.pd_ecommerce().describe()
        # oml_ecommerce_describe = self.oml_ecommerce().describe()
        # We don't compare ecommerce here as the default dtypes in pandas from read_json
        # don't match the mapping types. This is mainly because the products field is
        # nested and so can be treated as a multi-field in ES, but not in pandas

        # We can not also run 'describe' on a truncate oml dataframe
