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

import pytest

# File called _pytest for PyCharm compatability
from tests.common import TestData


class TestDataFrameDrop(TestData):
    def test_drop_rows(self, df):
        # to properly test dropping of indices we must account for differently named indexes
        pd_drop_many = ["1", "2"]
        pd_drop_one = "3"
        ed_index = df.ed.head().to_pandas().index
        ed_drop_many = [ed_index[1], ed_index[2]]
        ed_drop_one = ed_index[3]

        df.check_values(df.ed.drop(ed_drop_many), df.pd.drop(pd_drop_many))
        df.check_values(df.ed.drop(labels=ed_drop_many, axis=0), df.pd.drop(labels=pd_drop_many, axis=0))
        df.check_values(df.ed.drop(index=ed_drop_many), df.pd.drop(index=pd_drop_many))
        df.check_values(df.ed.drop(labels=ed_drop_one, axis=0), df.pd.drop(labels=pd_drop_one, axis=0))

    def test_drop_columns(self, df):
        df.drop(labels=["Carrier", "DestCityName"], axis=1)
        df.drop(columns=["Carrier", "DestCityName"])

        df.drop(columns="Carrier")
        df.drop(columns=["Carrier", "Carrier_1"], errors="ignore")
        df.drop(columns=["Carrier_1"], errors="ignore")

    def test_drop_all_columns(self, df):
        all_columns = list(df.columns)
        rows = df.shape[0]

        for dropped in (
            df.drop(labels=all_columns, axis=1),
            df.drop(columns=all_columns),
            df.drop(all_columns, axis=1),
        ):
            assert dropped.shape == (rows, 0)
            assert list(dropped.columns) == []

    def test_drop_all_index(self, df):
        all_index_pd = list(df.pd.index)
        all_index_ed = list(df.ed.to_pandas().index)
        cols = df.shape[1]

        for dropped_pd, dropped_ed in (
                (df.pd.drop(all_index_pd), df.ed.drop(all_index_ed)),
                (df.pd.drop(all_index_pd, axis=0), df.ed.drop(all_index_ed, axis=0)),
                (df.pd.drop(index=all_index_pd), df.ed.drop(index=all_index_ed)),
        ):
            assert dropped_pd.shape == (0, cols)
            assert dropped_ed.shape == (0, cols)
            assert list(dropped_pd.index) == []
            assert list(dropped_ed.to_pandas().index) == []

    def test_drop_raises(self):
        ed_flights = self.ed_flights()

        with pytest.raises(
            ValueError, match="Cannot specify both 'labels' and 'index'/'columns'"
        ):
            ed_flights.drop(
                labels=["Carrier", "DestCityName"], columns=["Carrier", "DestCityName"]
            )

        with pytest.raises(
            ValueError, match="Cannot specify both 'labels' and 'index'/'columns'"
        ):
            ed_flights.drop(labels=["Carrier", "DestCityName"], index=[0, 1, 2])

        with pytest.raises(
            ValueError,
            match="Need to specify at least one of 'labels', 'index' or 'columns'",
        ):
            ed_flights.drop()

        with pytest.raises(
            ValueError,
            match="number of labels 0!=2 not contained in axis",
        ):
            ed_flights.drop(errors="raise", axis=0, labels=["-1", "-2"])

        with pytest.raises(ValueError) as error:
            ed_flights.drop(columns=["Carrier_1"], errors="raise")
            assert str(error.value) == "labels ['Carrier_1'] not contained in axis"
