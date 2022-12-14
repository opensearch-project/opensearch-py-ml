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

from opensearch_py_ml import opensearch_to_pandas
from tests.common import TestData, assert_pandas_opensearch_py_ml_frame_equal


class TestSeriesNA(TestData):
    columns = [
        "currency",
        "customer_full_name",
        "geoip.country_iso_code",
        "geoip.region_name",
    ]

    def test_not_isna(self):
        oml_ecommerce = self.oml_ecommerce()
        pd_ecommerce = opensearch_to_pandas(oml_ecommerce)

        for column in self.columns:
            not_isna_oml_ecommerce = oml_ecommerce[~oml_ecommerce[column].isna()]
            not_isna_pd_ecommerce = pd_ecommerce[~pd_ecommerce[column].isna()]
            assert_pandas_opensearch_py_ml_frame_equal(
                not_isna_pd_ecommerce, not_isna_oml_ecommerce
            )

    def test_isna(self):
        oml_ecommerce = self.oml_ecommerce()
        pd_ecommerce = opensearch_to_pandas(oml_ecommerce)

        isna_oml_ecommerce = oml_ecommerce[oml_ecommerce["geoip.region_name"].isna()]
        isna_pd_ecommerce = pd_ecommerce[pd_ecommerce["geoip.region_name"].isna()]
        assert_pandas_opensearch_py_ml_frame_equal(
            isna_pd_ecommerce, isna_oml_ecommerce
        )

    def test_notna(self):
        oml_ecommerce = self.oml_ecommerce()
        pd_ecommerce = opensearch_to_pandas(oml_ecommerce)

        for column in self.columns:
            notna_oml_ecommerce = oml_ecommerce[oml_ecommerce[column].notna()]
            notna_pd_ecommerce = pd_ecommerce[pd_ecommerce[column].notna()]
            assert_pandas_opensearch_py_ml_frame_equal(
                notna_pd_ecommerce, notna_oml_ecommerce
            )
