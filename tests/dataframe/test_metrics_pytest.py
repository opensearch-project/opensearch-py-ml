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

import numpy as np
import pandas as pd

# File called _pytest for PyCharm compatibility
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from tests.common import TestData, assert_almost_equal


class TestDataFrameMetrics(TestData):
    funcs = ["max", "min", "mean", "sum"]
    extended_funcs = ["median", "mad", "var", "std"]
    filter_data = [
        "AvgTicketPrice",
        "Cancelled",
        "dayOfWeek",
        "timestamp",
        "DestCountry",
    ]

    @pytest.mark.parametrize("numeric_only", [False, None])
    def test_flights_metrics(self, numeric_only):
        pd_flights = self.pd_flights()
        oml_flights = self.oml_flights()

        for func in self.funcs:
            # Pandas v1.0 doesn't support mean() on datetime
            # Pandas and opensearch_py_ml don't support sum() on datetime
            if not numeric_only:
                dtype_include = (
                    [np.number, np.datetime64]
                    if func not in ("mean", "sum")
                    else [np.number]
                )
                pd_flights = pd_flights.select_dtypes(include=dtype_include)
                oml_flights = oml_flights.select_dtypes(include=dtype_include)

            pd_metric = getattr(pd_flights, func)(numeric_only=numeric_only)
            oml_metric = getattr(oml_flights, func)(numeric_only=numeric_only)

            assert_series_equal(pd_metric, oml_metric, check_dtype=False)

    def test_flights_extended_metrics(self):
        pd_flights = self.pd_flights()
        oml_flights = self.oml_flights()

        # Test on reduced set of data for more consistent
        # median behaviour + better var, std test for sample vs population
        pd_flights = pd_flights[["AvgTicketPrice"]]
        oml_flights = oml_flights[["AvgTicketPrice"]]

        import logging

        logger = logging.getLogger("elasticsearch")
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)

        for func in self.extended_funcs:
            if func == "mad":
                pd_metric = (pd_flights - pd_flights.mean()).abs().mean()
            else:
                pd_metric = getattr(pd_flights, func)(**({"numeric_only": True}))
            oml_metric = getattr(oml_flights, func)(numeric_only=True)

            pd_value = pd_metric["AvgTicketPrice"]
            oml_value = oml_metric["AvgTicketPrice"]
            assert (oml_value * 0.9) <= pd_value <= (oml_value * 1.1)  # +/-10%

    def test_flights_extended_metrics_nan(self):
        pd_flights = self.pd_flights()
        oml_flights = self.oml_flights()

        # Test on single row to test NaN behaviour of sample std/variance
        pd_flights_1 = pd_flights[pd_flights.FlightNum == "9HY9SWR"][["AvgTicketPrice"]]
        oml_flights_1 = oml_flights[oml_flights.FlightNum == "9HY9SWR"][
            ["AvgTicketPrice"]
        ]

        for func in self.extended_funcs:
            if func == "mad":
                pd_metric = (pd_flights_1 - pd_flights_1.mean()).abs().mean()
            else:
                pd_metric = getattr(pd_flights_1, func)()
            oml_metric = getattr(oml_flights_1, func)(numeric_only=False)

            assert_series_equal(pd_metric, oml_metric, check_exact=False)

        # Test on zero rows to test NaN behaviour of sample std/variance
        pd_flights_0 = pd_flights[pd_flights.FlightNum == "XXX"][["AvgTicketPrice"]]
        oml_flights_0 = oml_flights[oml_flights.FlightNum == "XXX"][["AvgTicketPrice"]]

        for func in self.extended_funcs:
            if func == "mad":
                pd_metric = (pd_flights_0 - pd_flights_0.mean()).abs().mean()
            else:
                pd_metric = getattr(pd_flights_0, func)()
            oml_metric = getattr(oml_flights_0, func)(numeric_only=False)

            assert_series_equal(pd_metric, oml_metric, check_exact=False)

    def test_ecommerce_selected_non_numeric_source_fields(self):
        # None of these are numeric
        columns = [
            "category",
            "currency",
            "customer_first_name",
            "user",
        ]

        pd_ecommerce = self.pd_ecommerce()[columns]
        oml_ecommerce = self.oml_ecommerce()[columns]

        for func in self.funcs:
            assert_series_equal(
                getattr(pd_ecommerce, func)(numeric_only=True),
                getattr(oml_ecommerce, func)(numeric_only=True),
                check_exact=False,
            )

    def test_ecommerce_selected_mixed_numeric_source_fields(self):
        # Some of these are numeric
        columns = [
            "category",
            "currency",
            "taxless_total_price",
            "total_quantity",
            "customer_first_name",
            "user",
        ]

        pd_ecommerce = self.pd_ecommerce()[columns]
        oml_ecommerce = self.oml_ecommerce()[columns]

        for func in self.funcs:
            assert_series_equal(
                getattr(pd_ecommerce, func)(numeric_only=True),
                getattr(oml_ecommerce, func)(numeric_only=True),
                check_exact=False,
            )

    def test_ecommerce_selected_all_numeric_source_fields(self):
        # All of these are numeric
        columns = ["total_quantity", "taxful_total_price", "taxless_total_price"]

        pd_ecommerce = self.pd_ecommerce()[columns]
        oml_ecommerce = self.oml_ecommerce()[columns]

        for func in self.funcs:
            assert_series_equal(
                getattr(pd_ecommerce, func)(numeric_only=True),
                getattr(oml_ecommerce, func)(numeric_only=True),
                check_exact=False,
            )

    def test_flights_datetime_metrics_agg(self):
        oml_timestamps = self.oml_flights()[["timestamp"]]
        expected_values = {
            "max": pd.Timestamp("2018-02-11 23:50:12"),
            "min": pd.Timestamp("2018-01-01 00:00:00"),
            "mean": pd.Timestamp("2018-01-21 19:20:45.564438232"),
            "sum": pd.NaT,
            "mad": pd.NaT,
            "var": pd.NaT,
            "std": pd.NaT,
            "nunique": 12236,
        }

        oml_metrics = oml_timestamps.agg(
            self.funcs + self.extended_funcs + ["nunique"], numeric_only=False
        )
        oml_metrics_dict = oml_metrics["timestamp"].to_dict()
        oml_metrics_dict.pop("median")  # Median is tested below.

        for key, expected_value in expected_values.items():
            assert_almost_equal(oml_metrics_dict[key], expected_value)

    @pytest.mark.parametrize("agg", ["mean", "min", "max", "nunique"])
    def test_flights_datetime_metrics_single_agg(self, agg):
        oml_timestamps = self.oml_flights()[["timestamp"]]
        expected_values = {
            "min": pd.Timestamp("2018-01-01 00:00:00"),
            "mean": pd.Timestamp("2018-01-21 19:20:45.564438232"),
            "max": pd.Timestamp("2018-02-11 23:50:12"),
            "nunique": 12236,
        }
        oml_metric = oml_timestamps.agg([agg])

        if agg == "nunique":
            # df with timestamp column should return int64
            assert oml_metric.dtypes["timestamp"] == np.int64
        else:
            # df with timestamp column should return datetime64[ns]
            assert oml_metric.dtypes["timestamp"] == np.dtype("datetime64[ns]")
        assert_almost_equal(oml_metric["timestamp"][0], expected_values[agg])

    @pytest.mark.parametrize("agg", ["mean", "min", "max"])
    def test_flights_datetime_metrics_agg_func(self, agg):
        oml_timestamps = self.oml_flights()[["timestamp"]]
        expected_values = {
            "min": pd.Timestamp("2018-01-01 00:00:00"),
            "mean": pd.Timestamp("2018-01-21 19:20:45.564438232"),
            "max": pd.Timestamp("2018-02-11 23:50:12"),
        }
        oml_metric = getattr(oml_timestamps, agg)(numeric_only=False)

        assert oml_metric.dtype == np.dtype("datetime64[ns]")
        assert_almost_equal(oml_metric[0], expected_values[agg])

    @pytest.mark.parametrize("agg", ["median", "quantile"])
    def test_flights_datetime_metrics_median_quantile(self, agg):
        oml_df = self.oml_flights_small()[["timestamp"]]

        median = oml_df.median(numeric_only=False)[0]
        assert isinstance(median, pd.Timestamp)
        assert (
            pd.to_datetime("2018-01-01 10:00:00.000")
            <= median
            <= pd.to_datetime("2018-01-01 12:00:00.000")
        )

        agg_value = oml_df.agg([agg])["timestamp"][0]
        assert isinstance(agg_value, pd.Timestamp)
        assert (
            pd.to_datetime("2018-01-01 10:00:00.000")
            <= agg_value
            <= pd.to_datetime("2018-01-01 12:00:00.000")
        )

    def test_metric_agg_keep_dtypes(self):
        # max, min and median maintain their dtypes
        df = self.oml_flights_small()[["AvgTicketPrice", "Cancelled", "dayOfWeek"]]
        assert df.min().tolist() == [131.81910705566406, False, 0]
        assert df.max().tolist() == [989.9527587890625, True, 0]
        assert df.median().tolist() == [550.276123046875, False, 0]
        all_agg = df.agg(["min", "max", "median"])
        assert all_agg.dtypes.tolist() == [
            np.dtype("float64"),
            np.dtype("bool"),
            np.dtype("int64"),
        ]
        assert all_agg.to_dict() == {
            "AvgTicketPrice": {
                "max": 989.9527587890625,
                "median": 550.276123046875,
                "min": 131.81910705566406,
            },
            "Cancelled": {"max": True, "median": False, "min": False},
            "dayOfWeek": {"max": 0, "median": 0, "min": 0},
        }
        # sum should always be the same dtype as the input, except for bool where the sum of bools should be an int64.
        sum_agg = df.agg(["sum"])
        assert sum_agg.dtypes.to_list() == [
            np.dtype("float64"),
            np.dtype("int64"),
            np.dtype("int64"),
        ]
        assert sum_agg.to_dict() == {
            "AvgTicketPrice": {"sum": 26521.624084472656},
            "Cancelled": {"sum": 6},
            "dayOfWeek": {"sum": 0},
        }

    def test_flights_numeric_only(self):
        # All Aggregations Data Check
        oml_flights = self.oml_flights().filter(self.filter_data)
        pd_flights = self.pd_flights().filter(self.filter_data)
        # agg => numeric_only True returns float64 values
        # We compare it with individual single agg functions of pandas with numeric_only=True
        filtered_aggs = self.funcs + self.extended_funcs
        agg_data = oml_flights.agg(filtered_aggs, numeric_only=True).transpose()
        for agg in filtered_aggs:
            # Explicitly check for mad because it returns nan for bools
            if agg == "mad":
                assert np.isnan(agg_data[agg]["Cancelled"])
            else:
                assert_series_equal(
                    agg_data[agg].rename(None),
                    getattr(pd_flights, agg)(numeric_only=True).astype(float),
                    check_exact=False,
                    rtol=True,
                )

    # all single aggs return float64 for numeric_only=True
    def test_numeric_only_true_single_aggs(self):
        oml_flights = self.oml_flights().filter(self.filter_data)
        for agg in self.funcs + self.extended_funcs:
            result = getattr(oml_flights, agg)(numeric_only=True)
            assert result.dtype == np.dtype("float64")
            assert result.shape == ((3,) if agg != "mad" else (2,))

    # check dtypes and shape of min, max and median for numeric_only=False | None
    @pytest.mark.parametrize("agg", ["min", "max", "median"])
    @pytest.mark.parametrize("numeric_only", [False, None])
    def test_min_max_median_numeric_only(self, agg, numeric_only):
        oml_flights = self.oml_flights().filter(self.filter_data)
        if numeric_only is False:
            calculated_values = getattr(oml_flights, agg)(numeric_only=numeric_only)
            assert isinstance(calculated_values["AvgTicketPrice"], np.float64)
            assert isinstance(calculated_values["Cancelled"], np.bool_)
            assert isinstance(calculated_values["dayOfWeek"], np.int64)
            assert isinstance(calculated_values["timestamp"], pd.Timestamp)
            assert np.isnan(calculated_values["DestCountry"])
            assert calculated_values.shape == (5,)
        elif numeric_only is None:
            calculated_values = getattr(oml_flights, agg)(numeric_only=numeric_only)
            assert isinstance(calculated_values["AvgTicketPrice"], np.float64)
            assert isinstance(calculated_values["Cancelled"], np.bool_)
            assert isinstance(calculated_values["dayOfWeek"], np.int64)
            assert isinstance(calculated_values["timestamp"], pd.Timestamp)
            assert calculated_values.shape == (4,)

    # check dtypes and shape for sum
    @pytest.mark.parametrize("numeric_only", [False, None])
    def test_sum_numeric_only(self, numeric_only):
        oml_flights = self.oml_flights().filter(self.filter_data)
        if numeric_only is False:
            calculated_values = oml_flights.sum(numeric_only=numeric_only)
            assert isinstance(calculated_values["AvgTicketPrice"], np.float64)
            assert isinstance(calculated_values["dayOfWeek"], np.int64)
            assert isinstance(calculated_values["Cancelled"], np.int64)
            assert pd.isnull(calculated_values["timestamp"])
            assert np.isnan(calculated_values["DestCountry"])
            assert calculated_values.shape == (5,)
        elif numeric_only is None:
            calculated_values = oml_flights.sum(numeric_only=numeric_only)
            dtype_list = [calculated_values[i].dtype for i in calculated_values.index]
            assert dtype_list == [
                np.dtype("float64"),
                np.dtype("int64"),
                np.dtype("int64"),
            ]
            assert calculated_values.shape == (3,)

    # check dtypes and shape for std
    @pytest.mark.parametrize("numeric_only", [False, None])
    def test_std_numeric_only(self, numeric_only):
        oml_flights = self.oml_flights().filter(self.filter_data)
        if numeric_only is False:
            calculated_values = oml_flights.std(numeric_only=numeric_only)
            assert isinstance(calculated_values["AvgTicketPrice"], float)
            assert isinstance(calculated_values["Cancelled"], float)
            assert isinstance(calculated_values["dayOfWeek"], float)
            assert pd.isnull(calculated_values["timestamp"])
            assert np.isnan(calculated_values["DestCountry"])
            assert calculated_values.shape == (5,)
        elif numeric_only is None:
            calculated_values = oml_flights.std(numeric_only=numeric_only)
            assert isinstance(calculated_values["AvgTicketPrice"], float)
            assert isinstance(calculated_values["Cancelled"], float)
            assert isinstance(calculated_values["dayOfWeek"], float)
            assert calculated_values.shape == (3,)

    # check dtypes and shape for var
    @pytest.mark.parametrize("numeric_only", [False, None])
    def test_var_numeric_only(self, numeric_only):
        oml_flights = self.oml_flights().filter(self.filter_data)
        if numeric_only is False:
            calculated_values = oml_flights.var(numeric_only=numeric_only)
            assert isinstance(calculated_values["AvgTicketPrice"], np.float64)
            assert isinstance(calculated_values["dayOfWeek"], np.float64)
            assert isinstance(calculated_values["Cancelled"], np.float64)
            assert pd.isnull(calculated_values["timestamp"])
            assert np.isnan(calculated_values["DestCountry"])
            assert calculated_values.shape == (5,)
        elif numeric_only is None:
            calculated_values = oml_flights.var(numeric_only=numeric_only)
            assert isinstance(calculated_values["AvgTicketPrice"], float)
            assert isinstance(calculated_values["Cancelled"], float)
            assert isinstance(calculated_values["dayOfWeek"], float)
            assert calculated_values.shape == (3,)

    # check dtypes and shape for mean
    @pytest.mark.parametrize("numeric_only", [False, None])
    def test_mean_numeric_only(self, numeric_only):
        oml_flights = self.oml_flights().filter(self.filter_data)
        if numeric_only is False:
            calculated_values = oml_flights.mean(numeric_only=numeric_only)
            assert isinstance(calculated_values["AvgTicketPrice"], float)
            assert isinstance(calculated_values["dayOfWeek"], float)
            assert isinstance(calculated_values["Cancelled"], float)
            assert isinstance(calculated_values["timestamp"], pd.Timestamp)
            assert np.isnan(calculated_values["DestCountry"])
            assert calculated_values.shape == (5,)
        elif numeric_only is None:
            calculated_values = oml_flights.mean(numeric_only=numeric_only)
            assert isinstance(calculated_values["AvgTicketPrice"], float)
            assert isinstance(calculated_values["Cancelled"], float)
            assert isinstance(calculated_values["dayOfWeek"], float)
            assert isinstance(calculated_values["timestamp"], pd.Timestamp)
            assert calculated_values.shape == (4,)

    # check dtypes and shape for mad
    @pytest.mark.parametrize("numeric_only", [False, None])
    def test_mad_numeric_only(self, numeric_only):
        oml_flights = self.oml_flights().filter(self.filter_data)
        if numeric_only is False:
            calculated_values = oml_flights.mad(numeric_only=numeric_only)
            assert isinstance(calculated_values["AvgTicketPrice"], float)
            assert isinstance(calculated_values["Cancelled"], np.float64)
            assert isinstance(calculated_values["dayOfWeek"], float)
            assert pd.isnull(calculated_values["timestamp"])
            assert np.isnan(calculated_values["DestCountry"])
            assert calculated_values.shape == (5,)
        elif numeric_only is None:
            calculated_values = oml_flights.mad(numeric_only=numeric_only)
            assert isinstance(calculated_values["AvgTicketPrice"], float)
            assert isinstance(calculated_values["dayOfWeek"], float)
            assert calculated_values.shape == (2,)

    def test_aggs_count(self):
        pd_flights = self.pd_flights().filter(self.filter_data)
        oml_flights = self.oml_flights().filter(self.filter_data)

        pd_count = pd_flights.agg(["count"])
        oml_count = oml_flights.agg(["count"])

        assert_frame_equal(pd_count, oml_count)

    @pytest.mark.parametrize("numeric_only", [True, False])
    @pytest.mark.parametrize("os_size", [1, 2, 20, 100, 5000, 3000])
    def test_aggs_mode(self, os_size, numeric_only):
        # FlightNum has unique values, so we can test `fill` NaN/NaT for remaining columns
        pd_flights = self.pd_flights().filter(
            ["Cancelled", "dayOfWeek", "timestamp", "DestCountry", "FlightNum"]
        )
        oml_flights = self.oml_flights().filter(
            ["Cancelled", "dayOfWeek", "timestamp", "DestCountry", "FlightNum"]
        )

        pd_mode = pd_flights.mode(numeric_only=numeric_only)[:os_size]
        oml_mode = oml_flights.mode(numeric_only=numeric_only, os_size=os_size)

        # Skipping dtype check because opensearch_py_ml is giving Cancelled dtype as bool
        # but pandas is referring it as object
        assert_frame_equal(
            pd_mode, oml_mode, check_dtype=(False if os_size == 1 else True)
        )

    @pytest.mark.parametrize("quantiles", [[0.2, 0.5], [0, 1], [0.75, 0.2, 0.1, 0.5]])
    @pytest.mark.parametrize("numeric_only", [False, None])
    def test_flights_quantile(self, quantiles, numeric_only):
        pd_flights = self.pd_flights()
        oml_flights = self.oml_flights()

        pd_quantile = pd_flights.filter(
            ["AvgTicketPrice", "FlightDelayMin", "dayOfWeek"]
        ).quantile(q=quantiles, numeric_only=numeric_only)
        oml_quantile = oml_flights.filter(
            ["AvgTicketPrice", "FlightDelayMin", "dayOfWeek"]
        ).quantile(q=quantiles, numeric_only=numeric_only)

        assert_frame_equal(pd_quantile, oml_quantile, check_exact=False, rtol=2)

        pd_quantile = pd_flights[["timestamp"]].quantile(
            q=quantiles, numeric_only=numeric_only
        )
        oml_quantile = oml_flights[["timestamp"]].quantile(
            q=quantiles, numeric_only=numeric_only
        )

        pd_timestamp = pd.to_numeric(pd_quantile.squeeze(), downcast="float")
        oml_timestamp = pd.to_numeric(oml_quantile.squeeze(), downcast="float")

        assert_series_equal(pd_timestamp, oml_timestamp, check_exact=False, rtol=2)

    @pytest.mark.parametrize("quantiles", [5, [2, 1], -1.5, [1.2, 0.2]])
    def test_flights_quantile_error(self, quantiles):
        oml_flights = self.oml_flights().filter(self.filter_data)

        match = f"quantile should be in range of 0 and 1, given {quantiles[0] if isinstance(quantiles, list) else quantiles}"
        with pytest.raises(ValueError, match=match):
            oml_flights[["timestamp"]].quantile(q=quantiles)

    @pytest.mark.parametrize("numeric_only", [True, False, None])
    def test_flights_agg_quantile(self, numeric_only):
        pd_flights = self.pd_flights().filter(
            ["AvgTicketPrice", "FlightDelayMin", "dayOfWeek"]
        )
        oml_flights = self.oml_flights().filter(
            ["AvgTicketPrice", "FlightDelayMin", "dayOfWeek"]
        )

        pd_quantile = pd_flights.agg([lambda x: x.quantile(0.5), lambda x: x.min()])
        pd_quantile.index = ["quantile", "min"]
        oml_quantile = oml_flights.agg(["quantile", "min"], numeric_only=numeric_only)

        assert_frame_equal(
            pd_quantile, oml_quantile, check_exact=False, rtol=4, check_dtype=False
        )

    def test_flights_idx_on_index(self):
        pd_flights = self.pd_flights().filter(
            ["AvgTicketPrice", "FlightDelayMin", "dayOfWeek"]
        )
        oml_flights = self.oml_flights().filter(
            ["AvgTicketPrice", "FlightDelayMin", "dayOfWeek"]
        )

        pd_idxmax = list(pd_flights.idxmax())
        oml_idxmax = list(oml_flights.idxmax())
        assert_frame_equal(
            pd_flights.filter(items=pd_idxmax, axis=0).reset_index(),
            oml_flights.filter(items=oml_idxmax, axis=0).to_pandas().reset_index(),
        )

        pd_idxmin = list(pd_flights.idxmin())
        oml_idxmin = list(oml_flights.idxmin())
        assert_frame_equal(
            pd_flights.filter(items=pd_idxmin, axis=0).reset_index(),
            oml_flights.filter(items=oml_idxmin, axis=0).to_pandas().reset_index(),
        )

    def test_flights_idx_on_columns(self):
        match = "This feature is not implemented yet for 'axis = 1'"
        with pytest.raises(NotImplementedError, match=match):
            oml_flights = self.oml_flights().filter(
                ["AvgTicketPrice", "FlightDelayMin", "dayOfWeek"]
            )
            oml_flights.idxmax(axis=1)
