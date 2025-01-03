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

import pandas as pd
import pytest

from opensearch_py_ml.utils import is_valid_attr_name, to_list, to_list_if_needed


class TestUtils:
    def test_is_valid_attr_name(self):
        assert is_valid_attr_name("_piZZa")
        assert is_valid_attr_name("nice_pizza_with_2_mushrooms")
        assert is_valid_attr_name("_2_pizze")
        assert is_valid_attr_name("_")
        assert is_valid_attr_name("___")

        assert not is_valid_attr_name("4")
        assert not is_valid_attr_name(4)
        assert not is_valid_attr_name(None)
        assert not is_valid_attr_name("4pizze")
        assert not is_valid_attr_name("pizza+")

    def test_to_list_if_needed_index(self):
        index = pd.Index([1, 2, 3])
        result = to_list_if_needed(index)
        assert result == [1, 2, 3], "Expected Index to convert to list correctly"

    def test_to_list_if_needed_single_value(self):
        result = to_list_if_needed(42)
        assert result == [42], "Expected single value to be wrapped in list"

    def test_to_list_if_needed_list(self):
        result = to_list_if_needed([1, 2, 3])
        assert result == [1, 2, 3], "Expected list-like input to return as-is"

    def test_to_list_if_needed_none(self):
        result = to_list_if_needed(None)
        assert result is None, "Expected None to return None"

    def test_to_list_series(self):
        series = pd.Series([4, 5, 6])
        result = to_list(series)
        assert result == [4, 5, 6], "Expected Series to convert to list"

    def test_to_list_collection(self):
        collection = {7, 8, 9}
        result = to_list(collection)
        assert sorted(result) == [7, 8, 9], "Expected Collection to convert to list"

    def test_to_list_not_implemented(self):
        with pytest.raises(NotImplementedError):
            to_list(12345)  # Passing an integer should raise an error
