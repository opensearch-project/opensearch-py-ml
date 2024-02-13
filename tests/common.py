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

import os
from datetime import timedelta

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

import opensearch_py_ml as oml

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Create pandas and opensearch_py_ml data frames
from tests import (
    ECOMMERCE_DF_FILE_NAME,
    ECOMMERCE_INDEX_NAME,
    FLIGHTS_DF_FILE_NAME,
    FLIGHTS_INDEX_NAME,
    FLIGHTS_SMALL_INDEX_NAME,
    OPENSEARCH_TEST_CLIENT,
)

_pd_flights = pd.read_json(FLIGHTS_DF_FILE_NAME).sort_index()
_pd_flights["timestamp"] = pd.to_datetime(_pd_flights["timestamp"])
_pd_flights.index = _pd_flights.index.map(str)  # make index 'object' not int
_oml_flights = oml.DataFrame(OPENSEARCH_TEST_CLIENT, FLIGHTS_INDEX_NAME)

_pd_flights_small = _pd_flights.head(48)
_oml_flights_small = oml.DataFrame(OPENSEARCH_TEST_CLIENT, FLIGHTS_SMALL_INDEX_NAME)

_pd_ecommerce = pd.read_json(ECOMMERCE_DF_FILE_NAME).sort_index()
_pd_ecommerce["order_date"] = pd.to_datetime(_pd_ecommerce["order_date"])
_pd_ecommerce["products.created_on"] = _pd_ecommerce["products.created_on"].apply(
    lambda x: pd.to_datetime(x)
)
_pd_ecommerce.insert(2, "customer_birth_date", None)
_pd_ecommerce.index = _pd_ecommerce.index.map(str)  # make index 'object' not int
_pd_ecommerce["customer_birth_date"].astype("datetime64[ns]")
_oml_ecommerce = oml.DataFrame(OPENSEARCH_TEST_CLIENT, ECOMMERCE_INDEX_NAME)


class TestData:
    client = OPENSEARCH_TEST_CLIENT

    def pd_flights(self):
        return _pd_flights

    def oml_flights(self):
        return _oml_flights

    def pd_flights_small(self):
        return _pd_flights_small

    def oml_flights_small(self):
        return _oml_flights_small

    def pd_ecommerce(self):
        return _pd_ecommerce

    def oml_ecommerce(self):
        return _oml_ecommerce


def assert_pandas_opensearch_py_ml_frame_equal(left, right, **kwargs):
    if not isinstance(left, pd.DataFrame):
        raise AssertionError(f"Expected type pd.DataFrame, found {type(left)} instead")

    if not isinstance(right, oml.DataFrame):
        raise AssertionError(f"Expected type ed.DataFrame, found {type(right)} instead")

    # Use pandas tests to check similarity
    assert_frame_equal(
        left.reset_index(drop=True), right.to_pandas().reset_index(drop=True), **kwargs
    )


def assert_opensearch_py_ml_frame_equal(left, right, **kwargs):
    if not isinstance(left, oml.DataFrame):
        raise AssertionError(f"Expected type oml.DataFrame, found {type(left)} instead")

    if not isinstance(right, oml.DataFrame):
        raise AssertionError(
            f"Expected type oml.DataFrame, found {type(right)} instead"
        )

    # Use pandas tests to check similarity
    assert_frame_equal(
        left.to_pandas().reset_index(drop=True),
        right.to_pandas().reset_index(drop=True),
        **kwargs,
    )


def assert_pandas_opensearch_py_ml_series_equal(left, right, **kwargs):
    if not isinstance(left, pd.Series):
        raise AssertionError(f"Expected type pd.Series, found {type(left)} instead")

    if not isinstance(right, oml.Series):
        raise AssertionError(f"Expected type oml.Series, found {type(right)} instead")

    # Use pandas tests to check similarity
    assert_series_equal(
        left.reset_index(drop=True), right.to_pandas().reset_index(drop=True), **kwargs
    )


def assert_almost_equal(left, right, **kwargs):
    """Asserts left and right are almost equal. Left and right
    can be scalars, series, dataframes, etc
    """
    if isinstance(left, (oml.DataFrame, oml.Series)):
        left = left.to_pandas().reset_index(drop=True)
    if isinstance(right, (oml.DataFrame, oml.Series)):
        right = right.to_pandas().reset_index(drop=True)

    if isinstance(right, pd.DataFrame):
        kwargs.setdefault("check_exact", True)
        assert_frame_equal(left, right)
    elif isinstance(right, pd.Series):
        kwargs.setdefault("check_exact", True)
        assert_series_equal(left, right)
    elif isinstance(right, float):
        assert right * 0.99 <= left <= right * 1.01
    elif isinstance(right, pd.Timestamp):
        assert isinstance(left, pd.Timestamp) and right - timedelta(
            seconds=0.1
        ) < left < right + timedelta(seconds=0.1)
    elif right is pd.NaT:
        assert left is pd.NaT
    else:
        assert left == right, f"{left} != {right}"


def mad(x):
    if isinstance(x, pd.Series):
        if x.dtype == "<M8[ns]":
            return pd.Timestamp("NaT")
        elif x.dtype == object:
            return np.nan
    else:
        numeric_columns = x.select_dtypes(include=["number", "bool"]).columns
        x = x[numeric_columns]
    return np.fabs(x - x.mean()).mean()

def quantile(x, numeric_only=None):
    return x.quantile()
