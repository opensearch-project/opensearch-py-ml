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
from tests.common import (
    OPENSEARCH_TEST_CLIENT,
    TestData,
    assert_pandas_opensearch_py_ml_frame_equal,
)


class TestDataFrameQuery(TestData):
    def test_getitem_query(self):
        # Examples from:
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html
        pd_df = pd.DataFrame(
            {"A": range(1, 6), "B": range(10, 0, -2), "C": range(10, 5, -1)},
            index=["0", "1", "2", "3", "4"],
        )
        """
        >>> pd_df
           A   B   C
        0  1  10  10
        1  2   8   9
        2  3   6   8
        3  4   4   7
        4  5   2   6
        """
        # Now create index
        index_name = "oml_test_query"

        oml_df = oml.pandas_to_opensearch(
            pd_df,
            OPENSEARCH_TEST_CLIENT,
            index_name,
            os_if_exists="replace",
            os_refresh=True,
        )

        assert_pandas_opensearch_py_ml_frame_equal(pd_df, oml_df)

        pd_df.info()
        oml_df.info()

        pd_q1 = pd_df[pd_df.A > 2]
        pd_q2 = pd_df[pd_df.A > pd_df.B]
        pd_q3 = pd_df[pd_df.B == pd_df.C]

        oml_q1 = oml_df[oml_df.A > 2]
        oml_q2 = oml_df[oml_df.A > oml_df.B]
        oml_q3 = oml_df[oml_df.B == oml_df.C]

        assert_pandas_opensearch_py_ml_frame_equal(pd_q1, oml_q1)
        assert_pandas_opensearch_py_ml_frame_equal(pd_q2, oml_q2)
        assert_pandas_opensearch_py_ml_frame_equal(pd_q3, oml_q3)

        pd_q4 = pd_df[(pd_df.A > 2) & (pd_df.B > 3)]
        oml_q4 = oml_df[(oml_df.A > 2) & (oml_df.B > 3)]

        assert_pandas_opensearch_py_ml_frame_equal(pd_q4, oml_q4)

        OPENSEARCH_TEST_CLIENT.indices.delete(index=index_name)

    def test_simple_query(self):
        oml_flights = self.oml_flights()
        pd_flights = self.pd_flights()

        assert (
            pd_flights.query("FlightDelayMin > 60").shape
            == oml_flights.query("FlightDelayMin > 60").shape
        )

    def test_isin_query(self):
        oml_flights = self.oml_flights()
        pd_flights = self.pd_flights()

        for obj in (["LHR", "SYD"], ("LHR", "SYD"), pd.Series(data=["LHR", "SYD"])):
            assert (
                pd_flights[pd_flights.OriginAirportID.isin(obj)].shape
                == oml_flights[oml_flights.OriginAirportID.isin(obj)].shape
            )

    def test_multiitem_query(self):
        # Examples from:
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html
        pd_df = pd.DataFrame(
            {"A": range(1, 6), "B": range(10, 0, -2), "C": range(10, 5, -1)},
            index=["0", "1", "2", "3", "4"],
        )
        """
        >>> pd_df
           A   B   C
        0  1  10  10
        1  2   8   9
        2  3   6   8
        3  4   4   7
        4  5   2   6
        """
        # Now create index
        index_name = "oml_test_query"

        oml_df = oml.pandas_to_opensearch(
            pd_df,
            OPENSEARCH_TEST_CLIENT,
            index_name,
            os_if_exists="replace",
            os_refresh=True,
        )

        assert_pandas_opensearch_py_ml_frame_equal(pd_df, oml_df)

        pd_df.info()
        oml_df.info()

        pd_q1 = pd_df[pd_df.A > 2]
        pd_q2 = pd_df[pd_df.A > pd_df.B]
        pd_q3 = pd_df[pd_df.B == pd_df.C]

        oml_q1 = oml_df[oml_df.A > 2]
        oml_q2 = oml_df[oml_df.A > oml_df.B]
        oml_q3 = oml_df[oml_df.B == oml_df.C]

        assert_pandas_opensearch_py_ml_frame_equal(pd_q1, oml_q1)
        assert_pandas_opensearch_py_ml_frame_equal(pd_q2, oml_q2)
        assert_pandas_opensearch_py_ml_frame_equal(pd_q3, oml_q3)

        oml_q4 = oml_q1.query("B > 2")
        pd_q4 = pd_q1.query("B > 2")

        assert_pandas_opensearch_py_ml_frame_equal(pd_q4, oml_q4)

        # Drop rows by index
        oml_q4 = oml_q4.drop(["2"])
        pd_q4 = pd_q4.drop(["2"])

        assert_pandas_opensearch_py_ml_frame_equal(pd_q4, oml_q4)

        OPENSEARCH_TEST_CLIENT.indices.delete(index=index_name)
