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

from datetime import datetime

# File called _pytest for PyCharm compatability
import numpy as np
import pytest

from opensearch_py_ml import Series
from tests.common import TestData, assert_pandas_opensearch_py_ml_series_equal


class TestSeriesArithmetics(TestData):
    def test_ecommerce_datetime_comparisons(self):
        pd_df = self.pd_ecommerce()
        oml_df = self.oml_ecommerce()

        ops = ["__le__", "__lt__", "__gt__", "__ge__", "__eq__", "__ne__"]

        # this datetime object is timezone naive
        datetime_obj = datetime(2016, 12, 18)

        # FIXME: the following timezone conversions are just a temporary fix
        # to run the datetime comparison tests
        #
        # The problem:
        # - the datetime objects of the pandas DataFrame are timezone aware and
        #   can't be compared with timezone naive datetime objects
        # - the datetime objects of the opensearch_py_ml DataFrame are timezone naive (which
        #   should be fixed)
        # - however if the opensearch_py_ml DataFrame is converted to a pandas DataFrame
        #   (using the `to_pandas` function) the datetime objects become timezone aware
        #
        # This tests converts the datetime objects of the pandas Series to
        # timezone naive ones and utilizes a class to make the datetime objects of the
        # opensearch_py_ml Series timezone naive before the result of `to_pandas` is returned.
        # The `to_pandas` function is executed by the `assert_pandas_opensearch_py_ml_series_equal`
        # function, which compares the opensearch_py_ml and pandas Series

        # convert to timezone naive datetime object
        pd_df["order_date"] = pd_df["order_date"].dt.tz_localize(None)

        class ModifiedElandSeries(Series):
            def to_pandas(self):
                """remove timezone awareness before returning the pandas dataframe"""
                series = super().to_pandas()
                series = series.dt.tz_localize(None)
                return series

        for op in ops:
            pd_series = pd_df[getattr(pd_df["order_date"], op)(datetime_obj)][
                "order_date"
            ]
            oml_series = oml_df[getattr(oml_df["order_date"], op)(datetime_obj)][
                "order_date"
            ]

            # "type cast" to modified class (inherits from ed.Series) that overrides the `to_pandas` function
            oml_series.__class__ = ModifiedElandSeries

            assert_pandas_opensearch_py_ml_series_equal(
                pd_series, oml_series, check_less_precise=True
            )

    def test_ecommerce_series_invalid_div(self):
        pd_df = self.pd_ecommerce()
        oml_df = self.oml_ecommerce()

        # opensearch_py_ml / pandas == error
        with pytest.raises(TypeError):
            _ = oml_df["total_quantity"] / pd_df["taxful_total_price"]

    def test_ecommerce_series_simple_arithmetics(self):
        pd_df = self.pd_ecommerce().head(100)
        oml_df = self.oml_ecommerce().head(100)

        pd_series = (
            pd_df["taxful_total_price"]
            + 5
            + pd_df["total_quantity"] / pd_df["taxless_total_price"]
            - pd_df["total_unique_products"] * 10.0
            + pd_df["total_quantity"]
        )
        oml_series = (
            oml_df["taxful_total_price"]
            + 5
            + oml_df["total_quantity"] / oml_df["taxless_total_price"]
            - oml_df["total_unique_products"] * 10.0
            + oml_df["total_quantity"]
        )

        assert_pandas_opensearch_py_ml_series_equal(pd_series, oml_series, rtol=True)

    def test_ecommerce_series_simple_integer_addition(self):
        pd_df = self.pd_ecommerce().head(100)
        oml_df = self.oml_ecommerce().head(100)

        pd_series = pd_df["taxful_total_price"] + 5
        oml_series = oml_df["taxful_total_price"] + 5

        assert_pandas_opensearch_py_ml_series_equal(pd_series, oml_series, rtol=True)

    def test_ecommerce_series_simple_series_addition(self):
        pd_df = self.pd_ecommerce().head(100)
        oml_df = self.oml_ecommerce().head(100)

        pd_series = pd_df["taxful_total_price"] + pd_df["total_quantity"]
        oml_series = oml_df["taxful_total_price"] + oml_df["total_quantity"]

        assert_pandas_opensearch_py_ml_series_equal(pd_series, oml_series, rtol=True)

    def test_ecommerce_series_basic_arithmetics(self):
        pd_df = self.pd_ecommerce().head(100)
        oml_df = self.oml_ecommerce().head(100)

        ops = [
            "__add__",
            "__truediv__",
            "__floordiv__",
            "__pow__",
            "__mod__",
            "__mul__",
            "__sub__",
            "add",
            "truediv",
            "floordiv",
            "pow",
            "mod",
            "mul",
            "sub",
        ]

        for op in ops:
            pd_series = getattr(pd_df["taxful_total_price"], op)(
                pd_df["total_quantity"]
            )
            oml_series = getattr(oml_df["taxful_total_price"], op)(
                oml_df["total_quantity"]
            )
            assert_pandas_opensearch_py_ml_series_equal(
                pd_series, oml_series, rtol=True
            )

            pd_series = getattr(pd_df["taxful_total_price"], op)(10.56)
            oml_series = getattr(oml_df["taxful_total_price"], op)(10.56)
            assert_pandas_opensearch_py_ml_series_equal(
                pd_series, oml_series, rtol=True
            )

            pd_series = getattr(pd_df["taxful_total_price"], op)(np.float32(1.879))
            oml_series = getattr(oml_df["taxful_total_price"], op)(np.float32(1.879))
            assert_pandas_opensearch_py_ml_series_equal(
                pd_series, oml_series, rtol=True
            )

            pd_series = getattr(pd_df["taxful_total_price"], op)(int(8))
            oml_series = getattr(oml_df["taxful_total_price"], op)(int(8))
            assert_pandas_opensearch_py_ml_series_equal(
                pd_series, oml_series, rtol=True
            )

    def test_supported_series_dtypes_ops(self):
        pd_df = self.pd_ecommerce().head(100)
        oml_df = self.oml_ecommerce().head(100)

        # Test some specific operations that are and aren't supported
        numeric_ops = [
            "__add__",
            "__truediv__",
            "__floordiv__",
            "__pow__",
            "__mod__",
            "__mul__",
            "__sub__",
        ]

        non_string_numeric_ops = [
            "__add__",
            "__truediv__",
            "__floordiv__",
            "__pow__",
            "__mod__",
            "__sub__",
        ]
        # __mul__ is supported for int * str in pandas

        # float op float
        for op in numeric_ops:
            pd_series = getattr(pd_df["taxful_total_price"], op)(
                pd_df["taxless_total_price"]
            )
            oml_series = getattr(oml_df["taxful_total_price"], op)(
                oml_df["taxless_total_price"]
            )
            assert_pandas_opensearch_py_ml_series_equal(
                pd_series, oml_series, rtol=True
            )

        # int op float
        for op in numeric_ops:
            pd_series = getattr(pd_df["total_quantity"], op)(
                pd_df["taxless_total_price"]
            )
            oml_series = getattr(oml_df["total_quantity"], op)(
                oml_df["taxless_total_price"]
            )
            assert_pandas_opensearch_py_ml_series_equal(
                pd_series, oml_series, rtol=True
            )

        # float op int
        for op in numeric_ops:
            pd_series = getattr(pd_df["taxful_total_price"], op)(
                pd_df["total_quantity"]
            )
            oml_series = getattr(oml_df["taxful_total_price"], op)(
                oml_df["total_quantity"]
            )
            assert_pandas_opensearch_py_ml_series_equal(
                pd_series, oml_series, rtol=True
            )

        # str op int (throws)
        for op in non_string_numeric_ops:
            with pytest.raises(TypeError):
                pd_series = getattr(pd_df["currency"], op)(pd_df["total_quantity"])
            with pytest.raises(TypeError):
                oml_series = getattr(oml_df["currency"], op)(oml_df["total_quantity"])
            with pytest.raises(TypeError):
                pd_series = getattr(pd_df["currency"], op)(1)
            with pytest.raises(TypeError):
                oml_series = getattr(oml_df["currency"], op)(1)

        # int op str (throws)
        for op in non_string_numeric_ops:
            with pytest.raises(TypeError):
                pd_series = getattr(pd_df["total_quantity"], op)(pd_df["currency"])
            with pytest.raises(TypeError):
                oml_series = getattr(oml_df["total_quantity"], op)(oml_df["currency"])

    def test_ecommerce_series_basic_rarithmetics(self):
        pd_df = self.pd_ecommerce().head(10)
        oml_df = self.oml_ecommerce().head(10)

        ops = [
            "__radd__",
            "__rtruediv__",
            "__rfloordiv__",
            "__rpow__",
            "__rmod__",
            "__rmul__",
            "__rsub__",
            "radd",
            "rtruediv",
            "rfloordiv",
            "rpow",
            "rmod",
            "rmul",
            "rsub",
        ]

        for op in ops:
            pd_series = getattr(pd_df["taxful_total_price"], op)(
                pd_df["total_quantity"]
            )
            oml_series = getattr(oml_df["taxful_total_price"], op)(
                oml_df["total_quantity"]
            )
            assert_pandas_opensearch_py_ml_series_equal(
                pd_series, oml_series, rtol=True
            )

            pd_series = getattr(pd_df["taxful_total_price"], op)(3.141)
            oml_series = getattr(oml_df["taxful_total_price"], op)(3.141)
            assert_pandas_opensearch_py_ml_series_equal(
                pd_series, oml_series, rtol=True
            )

            pd_series = getattr(pd_df["taxful_total_price"], op)(np.float32(2.879))
            oml_series = getattr(oml_df["taxful_total_price"], op)(np.float32(2.879))
            assert_pandas_opensearch_py_ml_series_equal(
                pd_series, oml_series, rtol=True
            )

            pd_series = getattr(pd_df["taxful_total_price"], op)(int(6))
            oml_series = getattr(oml_df["taxful_total_price"], op)(int(6))
            assert_pandas_opensearch_py_ml_series_equal(
                pd_series, oml_series, rtol=True
            )

    def test_supported_series_dtypes_rops(self):
        pd_df = self.pd_ecommerce().head(100)
        oml_df = self.oml_ecommerce().head(100)

        # Test some specific operations that are and aren't supported
        numeric_ops = [
            "__radd__",
            "__rtruediv__",
            "__rfloordiv__",
            "__rpow__",
            "__rmod__",
            "__rmul__",
            "__rsub__",
        ]

        non_string_numeric_ops = [
            "__radd__",
            "__rtruediv__",
            "__rfloordiv__",
            "__rpow__",
            "__rmod__",
            "__rsub__",
        ]
        # __rmul__ is supported for int * str in pandas

        # float op float
        for op in numeric_ops:
            pd_series = getattr(pd_df["taxful_total_price"], op)(
                pd_df["taxless_total_price"]
            )
            oml_series = getattr(oml_df["taxful_total_price"], op)(
                oml_df["taxless_total_price"]
            )
            assert_pandas_opensearch_py_ml_series_equal(
                pd_series, oml_series, rtol=True
            )

        # int op float
        for op in numeric_ops:
            pd_series = getattr(pd_df["total_quantity"], op)(
                pd_df["taxless_total_price"]
            )
            oml_series = getattr(oml_df["total_quantity"], op)(
                oml_df["taxless_total_price"]
            )
            assert_pandas_opensearch_py_ml_series_equal(
                pd_series, oml_series, rtol=True
            )

        # float op int
        for op in numeric_ops:
            pd_series = getattr(pd_df["taxful_total_price"], op)(
                pd_df["total_quantity"]
            )
            oml_series = getattr(oml_df["taxful_total_price"], op)(
                oml_df["total_quantity"]
            )
            assert_pandas_opensearch_py_ml_series_equal(
                pd_series, oml_series, rtol=True
            )

        # str op int (throws)
        for op in non_string_numeric_ops:
            with pytest.raises(TypeError):
                pd_series = getattr(pd_df["currency"], op)(pd_df["total_quantity"])
            with pytest.raises(TypeError):
                oml_series = getattr(oml_df["currency"], op)(oml_df["total_quantity"])
            with pytest.raises(TypeError):
                pd_series = getattr(pd_df["currency"], op)(10.0)
            with pytest.raises(TypeError):
                oml_series = getattr(oml_df["currency"], op)(10.0)

        # int op str (throws)
        for op in non_string_numeric_ops:
            with pytest.raises(TypeError):
                pd_series = getattr(pd_df["total_quantity"], op)(pd_df["currency"])
            with pytest.raises(TypeError):
                oml_series = getattr(oml_df["total_quantity"], op)(oml_df["currency"])
