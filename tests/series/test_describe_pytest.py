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

import pandas as pd

from opensearch_py_ml.utils import STANDARD_DEVIATION
from tests.common import TestData, assert_series_equal


class TestSeriesDescribe(TestData):
    def test_series_describe(self):
        oml_df = self.oml_flights_small()
        pd_df = self.pd_flights_small()

        oml_desc = oml_df.AvgTicketPrice.describe()
        pd_desc = pd_df.AvgTicketPrice.describe()

        assert isinstance(oml_desc, pd.Series)
        assert oml_desc.shape == pd_desc.shape
        assert oml_desc.dtype == pd_desc.dtype
        assert oml_desc.index.equals(pd_desc.index)

        # Percentiles calculations vary for Elasticsearch
        assert_series_equal(
            oml_desc[["count", "mean", STANDARD_DEVIATION, "min", "max"]],
            pd_desc[["count", "mean", STANDARD_DEVIATION, "min", "max"]],
            rtol=0.2,
        )
