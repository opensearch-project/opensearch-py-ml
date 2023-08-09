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

import inspect

import pandas as pd
import pytest

import opensearch_py_ml as oml

from .common import (
    TestData,
    _oml_ecommerce,
    _oml_flights,
    _oml_flights_small,
    _pd_ecommerce,
    _pd_flights,
    _pd_flights_small,
    assert_frame_equal,
    assert_pandas_opensearch_py_ml_frame_equal,
    assert_pandas_opensearch_py_ml_series_equal,
    assert_series_equal,
)


class SymmetricAPIChecker:
    def __init__(self, oml_obj, pd_obj):
        self.oml = oml_obj
        self.pd = pd_obj

    def load_dataset(self, dataset):
        if dataset == "flights":
            self.oml = _oml_flights
            self.pd = _pd_flights.copy()
        elif dataset == "flights_small":
            self.oml = _oml_flights_small
            self.pd = _pd_flights_small.copy()
        elif dataset == "ecommerce":
            self.oml = _oml_ecommerce
            self.pd = _pd_ecommerce.copy()
        else:
            raise ValueError(f"Unknown dataset {dataset!r}")

    def return_value_checker(self, func_name):
        """Returns a function which wraps the requested function
        and checks the return value when that function is inevitably
        called.
        """

        def f(*args, **kwargs):
            oml_exc = None
            try:
                oml_obj = getattr(self.oml, func_name)(*args, **kwargs)
            except Exception as e:
                oml_exc = e
            pd_exc = None
            try:
                if func_name == "to_pandas":
                    pd_obj = self.pd
                else:
                    pd_obj = getattr(self.pd, func_name)(*args, **kwargs)
            except Exception as e:
                pd_exc = e

            self.check_exception(oml_exc, pd_exc)
            self.check_values(oml_obj, pd_obj)

            if isinstance(oml_obj, (oml.DataFrame, oml.Series)):
                return SymmetricAPIChecker(oml_obj, pd_obj)
            return pd_obj

        return f

    def check_values(self, oml_obj, pd_obj):
        """Checks that any two values coming from opensearch_py_ml and pandas are equal"""
        if isinstance(oml_obj, oml.DataFrame):
            assert_pandas_opensearch_py_ml_frame_equal(pd_obj, oml_obj)
        elif isinstance(oml_obj, oml.Series):
            assert_pandas_opensearch_py_ml_series_equal(pd_obj, oml_obj)
        elif isinstance(oml_obj, pd.DataFrame):
            assert_frame_equal(oml_obj, pd_obj)
        elif isinstance(oml_obj, pd.Series):
            assert_series_equal(oml_obj, pd_obj)
        elif isinstance(oml_obj, pd.Index):
            assert oml_obj.equals(pd_obj)
        else:
            assert oml_obj == pd_obj

    def check_exception(self, ed_exc, pd_exc):
        """Checks that either an exception was raised or not from both opensearch_py_ml and pandas"""
        assert (ed_exc is None) == (pd_exc is None) and isinstance(ed_exc, type(pd_exc))
        if pd_exc is not None:
            raise pd_exc

    def __getitem__(self, item):
        if isinstance(item, SymmetricAPIChecker):
            pd_item = item.pd
            oml_item = item.oml
        else:
            pd_item = oml_item = item

        oml_exc = None
        pd_exc = None
        try:
            pd_obj = self.pd[pd_item]
        except Exception as e:
            pd_exc = e
        try:
            oml_obj = self.oml[oml_item]
        except Exception as e:
            oml_exc = e

        self.check_exception(oml_exc, pd_exc)
        if isinstance(oml_obj, (oml.DataFrame, oml.Series)):
            return SymmetricAPIChecker(oml_obj, pd_obj)
        return pd_obj

    def __getattr__(self, item):
        if item == "to_pandas":
            return self.return_value_checker("to_pandas")

        pd_obj = getattr(self.pd, item)
        if inspect.isfunction(pd_obj) or inspect.ismethod(pd_obj):
            return self.return_value_checker(item)
        oml_obj = getattr(self.oml, item)

        self.check_values(oml_obj, pd_obj)

        if isinstance(oml_obj, (oml.DataFrame, oml.Series)):
            return SymmetricAPIChecker(oml_obj, pd_obj)
        return pd_obj


@pytest.fixture(scope="function")
def df():
    return SymmetricAPIChecker(
        oml_obj=_oml_flights_small, pd_obj=_pd_flights_small.copy()
    )


@pytest.fixture(scope="session")
def testdata():
    return TestData()
