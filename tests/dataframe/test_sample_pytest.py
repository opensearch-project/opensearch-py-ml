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

# File called _pytest for PyCharm compatibility
import pytest
from pandas.testing import assert_frame_equal

from opensearch_py_ml import opensearch_to_pandas
from tests.common import TestData


class TestDataFrameSample(TestData):
    SEED = 42

    def build_from_index(self, sample_oml_flights):
        sample_pd_flights = self.pd_flights_small().loc[
            sample_oml_flights.index, sample_oml_flights.columns
        ]
        return sample_pd_flights

    def test_sample(self):
        oml_flights_small = self.oml_flights_small()
        first_sample = oml_flights_small.sample(n=10, random_state=self.SEED)
        second_sample = oml_flights_small.sample(n=10, random_state=self.SEED)

        assert_frame_equal(
            opensearch_to_pandas(first_sample), opensearch_to_pandas(second_sample)
        )

    @pytest.mark.parametrize(
        ["opts", "message"],
        [
            (
                {"n": 10, "frac": 0.1},
                "Please enter a value for `frac` OR `n`, not both",
            ),
            ({"frac": 1.5}, "`frac` must be between 0. and 1."),
            (
                {"n": -1},
                "A negative number of rows requested. Please provide positive value.",
            ),
            ({"n": 1.5}, "Only integers accepted as `n` values"),
        ],
    )
    def test_sample_raises(self, opts, message):
        oml_flights_small = self.oml_flights_small()

        with pytest.raises(ValueError, match=message):
            oml_flights_small.sample(**opts)

    def test_sample_basic(self):
        oml_flights_small = self.oml_flights_small()
        sample_oml_flights = oml_flights_small.sample(n=10, random_state=self.SEED)
        pd_from_eland = opensearch_to_pandas(sample_oml_flights)

        # build using index
        sample_pd_flights = self.build_from_index(pd_from_eland)

        assert_frame_equal(sample_pd_flights, pd_from_eland)

    def test_sample_frac_01(self):
        frac = 0.15
        oml_flights = self.oml_flights_small().sample(frac=frac, random_state=self.SEED)
        pd_from_eland = opensearch_to_pandas(oml_flights)
        pd_flights = self.build_from_index(pd_from_eland)

        assert_frame_equal(pd_flights, pd_from_eland)

        # assert right size from pd_flights
        size = len(self.pd_flights_small())
        assert len(pd_flights) == int(round(frac * size))

    def test_sample_on_boolean_filter(self):
        oml_flights = self.oml_flights_small()
        columns = ["timestamp", "OriginAirportID", "DestAirportID", "FlightDelayMin"]
        sample_oml_flights = oml_flights[columns].sample(n=5, random_state=self.SEED)
        pd_from_eland = opensearch_to_pandas(sample_oml_flights)
        sample_pd_flights = self.build_from_index(pd_from_eland)

        assert_frame_equal(sample_pd_flights, pd_from_eland)

    def test_sample_head(self):
        oml_flights = self.oml_flights_small()
        sample_oml_flights = oml_flights.sample(n=10, random_state=self.SEED)
        sample_pd_flights = self.build_from_index(
            opensearch_to_pandas(sample_oml_flights)
        )

        pd_head_5 = sample_pd_flights.head(5)
        oml_head_5 = sample_oml_flights.head(5)
        assert_frame_equal(pd_head_5, opensearch_to_pandas(oml_head_5))

    def test_sample_shape(self):
        oml_flights = self.oml_flights_small()
        sample_oml_flights = oml_flights.sample(n=10, random_state=self.SEED)
        sample_pd_flights = self.build_from_index(
            opensearch_to_pandas(sample_oml_flights)
        )

        assert sample_pd_flights.shape == sample_oml_flights.shape
