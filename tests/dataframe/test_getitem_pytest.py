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


class TestDataFrameGetItem:
    def test_getitem_one_attribute(self, df):
        df.load_dataset("flights")
        print(df.head(103)["OriginAirportID"])

    def test_getitem_attribute_list(self, df):
        print(df[["OriginAirportID", "AvgTicketPrice", "Carrier"]])

    def test_getitem_one_argument(self, df):
        print(df.OriginAirportID)

    def test_getitem_multiple_calls(self, df):
        df = df[["DestCityName", "DestCountry", "DestLocation", "DestRegion"]]
        with pytest.raises(KeyError):
            df["Carrier"]

        df["DestCountry"]
