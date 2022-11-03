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

from typing import Any, Dict

import numpy as np
import pandas as pd  # type: ignore
import pytest  # type: ignore

import opensearch_py_ml as oml

# Fix console size for consistent test results
pd.set_option("display.max_rows", 10)
pd.set_option("display.max_columns", 5)
pd.set_option("display.width", 100)


@pytest.fixture(autouse=True)
def add_imports(doctest_namespace: Dict[str, Any]) -> None:
    doctest_namespace["np"] = np
    doctest_namespace["pd"] = pd
    doctest_namespace["oml"] = oml
